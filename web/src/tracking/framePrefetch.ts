/**
 * Pipelining for the tracking loop: while one `/api/track/frame` request is in flight, the NEXT frame is
 * already being captured and encoded, so when the answer arrives a ready frame goes out at once.
 *
 * Without it the loop is strictly serial — capture, send, answer, capture… — and its period is capture +
 * round trip (measured on a Windows Edge laptop: 129 ms = capture 62 + round trip 55 + render). With it,
 * the period approaches max(capture, round trip). Requests to the server stay ONE in flight: this module
 * never sends anything; it only prepares the frame the loop will send next, and the loop still sends only
 * after the previous answer.
 *
 * Rules (each one keeps a prepared frame from making a box older than the serial loop would):
 * - The prepared frame is always a NEWER camera frame than the one just sent (`waitForNewerFrame` after the
 *   sent frame's presented-frame count), so the same decoded frame is never sent twice.
 * - Preparation starts `prefetchDelayMs` after the send — timed to finish about when the answer is due —
 *   not immediately: a frame prepared early only waits, ageing, while the request is still out.
 * - When the answer arrives, `take` wakes a preparation still in its delay, and awaits one already
 *   capturing (that frame is frozen already; finishing its encode is quicker than starting a new one).
 * - A prepared frame older than `staleLimitMs` (the answer came much later than expected) is returned as
 *   `stale` and NOT sent; the loop captures a fresh one instead.
 * - Stop, fence changes (scene, source, mirror, task, consent, camera), a new run or a retired run all
 *   abort the run's signal, which discards whatever is being prepared; its late result is dropped.
 *
 * Pure (no DOM or React), so `node --test` runs it.
 */

/** One camera frame at 30 fps, the rate the demo cameras present at. */
export const CAMERA_FRAME_MS = 1000 / 30;

/**
 * The oldest a prepared frame may be when the answer arrives and still be sent: twice the capture cost,
 * and never less than one camera frame.
 *
 * What the user sees is the box's age, which peaks just before the next answer replaces it. Sending a
 * prepared frame of age A now, versus capturing a fresh one (cost C) instead:
 * - the fresh capture delays this send, so the CURRENT box stays on screen C longer — its peak grows by C;
 * - the new box then starts C old instead of A old — the NEXT peak changes by C − A.
 * Together the fresh capture helps only once A > 2C, so that is the limit. The one-frame floor keeps a
 * frame that is as new as the camera can deliver from being thrown away while C is still unmeasured (0)
 * or very small.
 */
export function staleLimitMs(captureCostMs: number | null): number {
  return Math.max(CAMERA_FRAME_MS, 2 * (captureCostMs ?? 0));
}

/**
 * How long after a send to start preparing the next frame, so it is ready when the answer is due: the
 * expected round trip less the capture cost. Every millisecond off costs the same either way — starting
 * early ages the prepared frame by that much, starting late makes the answer wait that much on the
 * encode — so the target is the expected answer itself, with no margin. The wait for a newly presented
 * frame after the delay usually returns at once (the camera has moved on since the sent frame was
 * frozen). 0 (start at once) until both durations have been measured.
 */
export function prefetchDelayMs(roundTripMs: number | null, captureCostMs: number | null): number {
  if (roundTripMs === null || captureCostMs === null) return 0;
  return Math.max(0, roundTripMs - captureCostMs);
}

/**
 * Exponentially weighted moving average of a duration. α = 0.3 follows a sustained change within a few
 * frames while one slow outlier moves the estimate by under a third of its excess.
 */
export class DurationEstimate {
  private value: number | null = null;
  private readonly alpha: number;

  constructor(alpha = 0.3) {
    this.alpha = alpha;
  }

  add(sampleMs: number): void {
    if (!Number.isFinite(sampleMs) || sampleMs < 0) return;
    this.value = this.value === null ? sampleMs : this.value + this.alpha * (sampleMs - this.value);
  }

  get(): number | null {
    return this.value;
  }
}

export interface FramePrefetchDeps<F extends { capturedAt: number }> {
  /** Resolves once the camera has presented a frame newer than `afterIdx` (or the signal aborted). */
  waitForNewerFrame(afterIdx: number | undefined, signal: AbortSignal): Promise<void>;
  /** The camera's presented-frame count now; read beside the capture, it marks the captured frame. */
  presentedIdx(): number;
  /** Freeze and encode the current frame. `capturedAt` is its wall-clock freeze time. */
  capture(): Promise<F>;
  /** Resolves after `ms` (or once the signal aborted); must not reject. */
  sleep(ms: number, signal: AbortSignal): Promise<void>;
}

export type PrefetchTake<F> =
  | { kind: 'ready'; frame: F; presentedIdx: number }
  /** Prepared but too old to send; `presentedIdx` marks it, so the fresh capture waits for a newer frame. */
  | { kind: 'stale'; presentedIdx: number; ageMs: number }
  | { kind: 'none' };

type Settled<F> =
  | { kind: 'ok'; frame: F; presentedIdx: number }
  | { kind: 'error'; error: unknown }
  | { kind: 'cancelled' };

interface Slot<F> {
  controller: AbortController;
  wake: () => void;
  settled: Promise<Settled<F>>;
}

/** The one frame being prepared for a run. Owned by the run; dies with the run's signal. */
export class FramePrefetcher<F extends { capturedAt: number }> {
  private slot: Slot<F> | null = null;
  private readonly deps: FramePrefetchDeps<F>;
  private readonly signal: AbortSignal;

  constructor(deps: FramePrefetchDeps<F>, signal: AbortSignal) {
    this.deps = deps;
    this.signal = signal;
    signal.addEventListener('abort', () => this.discard(), { once: true });
  }

  /** Whether a frame is being prepared (or is ready and not yet taken). */
  get active(): boolean {
    return this.slot !== null;
  }

  /**
   * Start preparing the frame after `afterIdx` (the presented-frame count of the frame just sent),
   * `delayMs` from now. Replaces anything prepared before. At most one capture is ever started per call.
   */
  start(afterIdx: number | undefined, delayMs: number): void {
    this.discard();
    if (this.signal.aborted) return;
    const controller = new AbortController();
    const { promise: woken, resolve: wake } = Promise.withResolvers<void>();
    const { signal } = controller;
    const prepare = async (): Promise<Settled<F>> => {
      try {
        if (delayMs > 0) await Promise.race([this.deps.sleep(delayMs, signal), woken]);
        if (signal.aborted) return { kind: 'cancelled' };
        await this.deps.waitForNewerFrame(afterIdx, signal);
        if (signal.aborted) return { kind: 'cancelled' };
        // The marker of the frame about to be frozen: read before the capture, as the serial loop does.
        const presentedIdx = this.deps.presentedIdx();
        const frame = await this.deps.capture();
        return { kind: 'ok', frame, presentedIdx };
      } catch (error) {
        return { kind: 'error', error };
      }
    };
    this.slot = { controller, wake, settled: prepare() };
  }

  /**
   * The prepared frame, for sending now. Wakes a preparation still in its delay and awaits one in
   * progress. A capture failure is rethrown here, to be handled exactly like a failure of a fresh capture.
   */
  async take(staleAfterMs: number, now: () => number): Promise<PrefetchTake<F>> {
    const slot = this.slot;
    if (!slot) return { kind: 'none' };
    slot.wake();
    const settled = await slot.settled;
    if (this.slot === slot) this.slot = null;
    // Discarded while it was being awaited (the run was retired): nothing to send.
    if (slot.controller.signal.aborted || settled.kind === 'cancelled') return { kind: 'none' };
    if (settled.kind === 'error') throw settled.error;
    const ageMs = now() - settled.frame.capturedAt;
    if (ageMs > staleAfterMs) return { kind: 'stale', presentedIdx: settled.presentedIdx, ageMs };
    return { kind: 'ready', frame: settled.frame, presentedIdx: settled.presentedIdx };
  }

  /** Drop whatever is being prepared; a capture already running finishes, and its result is ignored. */
  discard(): void {
    const slot = this.slot;
    this.slot = null;
    if (!slot) return;
    slot.controller.abort();
    slot.wake();
  }
}
