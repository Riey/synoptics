/**
 * Owns one tracking run: mints it, drives the frame loop, and keeps the display a **complete pair** —
 * a captured image together with the response computed for that exact image.
 *
 * Pairing rule: the last complete pair is retained while the next frame is in flight, and the new image
 * + its result are swapped in atomically when the matching response arrives. The panel therefore never
 * shows an older box over a newer image, never strobes, and never holds geometry over a live image. The
 * age shown is this hook's own capture→now clock (the server's `ingest_age_ms` is a diagnostic, not a
 * freshness gate); past `TRACK_FRESHNESS_LIMIT_MS` the box is hidden.
 *
 * Cancellation model: every await is re-checked against `runRef.current`/abort/mounted/fence, so a
 * retired run's late answer can never be applied. Hard fences (stop, source, mirror, task, consent,
 * camera loss, a new run) retire the run immediately and synchronously; nothing auto-restarts.
 *
 * Pipelining: while a frame is in flight the run prepares the next one (`framePrefetch.ts`), so a ready
 * frame goes out the moment the answer arrives. It is still exactly one request in flight, the prepared
 * frame is always a newer camera frame than the one sent, a prepared frame grown too old is replaced by a
 * fresh capture, and everything being prepared dies with the run (every fence above aborts it).
 *
 * Two outputs per accepted answer: every accepted frame is published to the hook's `liveTrack` store
 * (the fast path the live overlay reads outside React), while the React `shown` pair — now a diagnostic
 * still — is committed at most every `SHOWN_COMMIT_INTERVAL_MS` unless the run/track/generation/state
 * changes. The hook lives in `App`, so a per-frame React commit would re-render the whole page per frame.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { Box } from '@visual-coach/visual-tools';
import type { TrackFrameResponse, TrackSelectResponse } from '../generated/api.generated';

import { ApiError } from '../coach/api';
import {
  CameraFrameUnavailableError,
  captureTrackFrameAsync,
  captureVideoFrame,
  trackFramesFromStream,
  type TrackFrameCapture,
} from '../coach/image-processor';
import { newOpaqueId } from '../ids';
import { decideError, decideUpdate, isBoxDrawable, isTerminalTrackState, runFenceKey } from './trackFence';
import { selectTarget, trackControl, trackFrame, trackFrameBinaryAvailable } from './trackClient';
import { createLiveTrackStore, type LiveTrackStore } from './liveTrackStore';
import { DurationEstimate, FramePrefetcher, prefetchDelayMs, staleLimitMs } from './framePrefetch';

/** Past this age the retained pair is no longer a usable position, so its box is hidden. */
export const TRACK_FRESHNESS_LIMIT_MS = 3000;
/**
 * Minimum spacing of React commits of the diagnostic still pair while nothing but the box moves. Each
 * commit is still a complete pair (image + its own answer); only intermediate pairs are skipped. The live
 * overlay never depends on this — it reads every accepted frame from the `liveTrack` store.
 */
export const SHOWN_COMMIT_INTERVAL_MS = 250;
/**
 * The target metadata a manually drawn (seeded) run carries. It is deliberately a neutral, honest token
 * — never a class guess — because the user's own box, not a text prompt, selects the object
 * (`source="user"`, no grounder). The visible label is the UI's own `직접 지정한 물체`.
 */
export const MANUAL_SELECTION_TARGET = 'user-selected object';
/**
 * Strictly increasing per app session. Module scope on purpose: the counter must survive a TrackPanel
 * remount so two mounts on the same session never reuse a `start_seq`.
 */
let startSeqCounter = 0;

/** The video element could not produce a frame yet — transient, not an outage. */
class CaptureUnavailableError extends Error {}

export interface TrackPair {
  frameId: string;
  frameSeq: number;
  /** The still of a frame uploaded as JSON (a `data:` URL); null when it went as raw JPEG (`jpeg`). */
  imageDataUrl: string | null;
  /** Exact uploaded bytes as base64 (JSON upload), retained so a manual seed reuses this frame. */
  rawBase64: string | null;
  /** Exact uploaded bytes when the frame went as raw JPEG (`?capture=v2`); null for a JSON upload. */
  jpeg: Blob | null;
  width: number;
  height: number;
  capturedAt: number;
}

/** A frame captured on the main thread for a startup analysis or a manual drawing: always base64. */
type StillPair = TrackPair & { rawBase64: string; imageDataUrl: string };

/** The atomic display unit: an image and the one response computed for it. */
export interface TrackShown {
  pair: TrackPair;
  response: TrackFrameResponse;
}

export interface TrackRunError {
  status: number;
  code: string;
  message: string;
}

/** A frame captured only for manual drawing, its own scene fence, and no tracker response. */
interface ManualCapture {
  pair: StillPair;
  fenceKey: string;
}

/** The honest outcome of a startup analysis that did not choose an object. */
export interface SelectionOutcome {
  status: Extract<TrackSelectResponse['status'], 'no_target' | 'uncertain'>;
  rationale: string | null;
}

interface SeedFrame {
  imageBase64: string;
  imageDataUrl: string;
  width: number;
  height: number;
  capturedAt: number;
}

interface ActiveRun {
  /** A startup analysis in flight, or a tracking run. One fence object serves both. */
  kind: 'select' | 'track';
  runId: string;
  target: string | null;
  sessionId: string;
  fenceKey: string;
  frameSeq: number;
  lastVersion: number;
  trackingId: string | null;
  controller: AbortController;
  /** First frame to send instead of capturing (manual seed); consumed once. */
  seedFrame: SeedFrame | undefined;
  seedBox: Box | undefined;
  /**
   * The camera's presented-frame count at the moment the last frame was CAPTURED. The loop compares it
   * against the live count after each response: a higher count means a newer frame already exists, so
   * the next capture starts immediately instead of waiting for yet another presented frame.
   */
  lastPresentedIdx: number | undefined;
  /**
   * The next frame, prepared while a request is in flight (tracking runs only; see `framePrefetch.ts`).
   * Bound to `controller.signal`, so every way a run is retired also discards what it was preparing.
   */
  prefetch: FramePrefetcher<TrackFrameCapture> | null;
  /** Capture start → encoded frame, and send → answer, as this run has measured them. */
  captureCost: DurationEstimate;
  roundTrip: DurationEstimate;
}

export interface UseObjectTrackOptions {
  videoRef: RefObject<HTMLVideoElement | null>;
  ensureSession: () => Promise<string>;
  sessionId: string | null;
  consent: boolean;
  cameraLive: boolean;
  mirror: boolean;
  sourceEpoch: number;
  sceneMode: string;
  cameraFrameSizeKey: string;
  taskKey: string;
  /** The current task inputs, sent with the startup analysis so the model sees what the user is doing. */
  userGoal: string;
  context: string;
  /**
   * Called exactly once per startup analysis attempt so the page-lifetime owner can fold the one paid
   * call's usage. `billed` carries a received answer's usage; `unknown` is a dispatched attempt whose
   * usage never came back. A pre-provider refusal is neither.
   */
  onSelectionBilling?: (billing: SelectionBilling) => void;
}

/** One startup analysis attempt's billing outcome, reported once to the page-lifetime owner. */
export type SelectionBilling =
  | { kind: 'billed'; provider: string; model: string | null; usage: TrackSelectResponse['usage'] }
  | { kind: 'unknown' };

interface FrameCallbackVideo extends Omit<HTMLVideoElement, 'requestVideoFrameCallback' | 'cancelVideoFrameCallback'> {
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: { presentedFrames: number; mediaTime: number }) => void
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
}

/**
 * One live rVFC observer. The object identity is the run's ownership token: a callback may only
 * re-register while its own pacer is still the installed one, so a late callback from a retired pacer
 * can never re-arm itself over the pacer that replaced it.
 */
interface Pacer {
  video: HTMLVideoElement;
  handle: number;
}

/**
 * Wait for the camera to present a NEW frame. This is the loop's slow-path pacer: with one request in
 * flight and this wait between requests, the rate is set by the tracker plus fresh video frames — never
 * a fixed sleep, and never the same decoded frame captured twice.
 */
function nextVideoFrame(video: HTMLVideoElement | null, signal: AbortSignal): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  // One listener per call, removed the moment the frame wins — otherwise a long run accumulates a
  // listener per dispatched frame and an abort resolves them all at once.
  let onAbort: (() => void) | undefined;
  const settle = () => {
    if (onAbort) signal.removeEventListener('abort', onAbort);
    onAbort = undefined;
    resolve();
  };
  if (signal.aborted) {
    resolve();
    return promise;
  }
  const target = video as FrameCallbackVideo | null;
  if (target && typeof target.requestVideoFrameCallback === 'function') {
    const handle = target.requestVideoFrameCallback(() => settle());
    onAbort = () => {
      target.cancelVideoFrameCallback?.(handle);
      settle();
    };
    signal.addEventListener('abort', onAbort, { once: true });
    return promise;
  }
  const raf = window.requestAnimationFrame(() => settle());
  onAbort = () => {
    window.cancelAnimationFrame(raf);
    settle();
  };
  signal.addEventListener('abort', onAbort, { once: true });
  return promise;
}

/**
 * The latest-frame pacer's fast path.
 *
 * `lastPresentedIdx` is the count of frames the camera had presented when the previous frame was
 * CAPTURED. If the camera has presented another frame since, that frame is already the newest one and
 * there is nothing to wait for — the next capture starts immediately. Only when no new frame has
 * arrived (the request returned faster than the camera presents, e.g. a low-latency tracker) does the
 * loop wait for one. Skipping that wait is what keeps the dispatch rate set by the camera instead of
 * by camera-plus-one-frame, which is the difference between capturing every frame and every other one.
 *
 * `presented` is read at call time and again when the wait arms, so a frame presented between the
 * capture and this call is never missed.
 */
function waitForPresentedFrame(
  video: HTMLVideoElement | null,
  presented: () => number,
  lastPresentedIdx: number | undefined,
  signal: AbortSignal
): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  if (signal.aborted) {
    resolve();
    return promise;
  }
  if (lastPresentedIdx !== undefined && presented() > lastPresentedIdx) {
    resolve();
    return promise;
  }
  // v2 with the camera track in the tracking worker: the worker itself takes a camera frame newer than the
  // last one it sent (`distinct`) the moment the camera delivers it — before the <video> presents it, so
  // waiting for the presentation here would only age the frame.
  if (video && trackFramesFromStream(video)) {
    resolve();
    return promise;
  }
  return nextVideoFrame(video, signal);
}

/** Resolves after `ms`, or at once when the signal aborts (the timer is cleared). Never rejects. */
function sleepUnlessAborted(ms: number, signal: AbortSignal): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  if (signal.aborted) {
    resolve();
    return promise;
  }
  const onAbort = () => {
    window.clearTimeout(timer);
    resolve();
  };
  const timer = window.setTimeout(() => {
    signal.removeEventListener('abort', onAbort);
    resolve();
  }, ms);
  signal.addEventListener('abort', onAbort, { once: true });
  return promise;
}

export function useObjectTrack(options: UseObjectTrackOptions) {
  const {
    videoRef,
    ensureSession,
    sessionId,
    consent,
    cameraLive,
    mirror,
    sourceEpoch,
    sceneMode,
    cameraFrameSizeKey,
    taskKey,
    userGoal,
    context,
    onSelectionBilling,
  } = options;

  const [phase, setPhase] = useState<'idle' | 'analyzing' | 'starting' | 'running' | 'error'>('idle');
  const [shown, setShown] = useState<TrackShown | null>(null);
  const [error, setError] = useState<TrackRunError | null>(null);
  /**
   * Bumped only to re-render the age readout. The age itself is DERIVED at render from the pair that is
   * actually shown. Storing it in state would let the first paint after a pair is published reuse the
   * previous pair's age, so a delayed frame's box could be drawn as fresh (or a fresh one as stale) for
   * one commit — the freshness limit must hold from the very first paint.
   */
  const [, ageTick] = useState(0);
  /** The honest outcome of the last startup analysis that found no object; null once a run starts. */
  const [selection, setSelection] = useState<SelectionOutcome | null>(null);
  /** The read-only target label of the current/last run; the neutral token for a manually drawn run. */
  const [target, setTarget] = useState<string | null>(null);
  /** A frame captured only for manual drawing, shown without pretending to be a tracker response. */
  const [manualCapture, setManualCapture] = useState<ManualCapture | null>(null);

  /** One store per hook instance: the per-frame fast path for the live overlay and status chip. */
  const [liveTrack] = useState<LiveTrackStore>(createLiveTrackStore);
  /** The last React commit of `shown`: when, and for which run/track/generation/state. */
  const shownCommitRef = useRef<{ at: number; key: string } | null>(null);

  const runRef = useRef<ActiveRun | null>(null);
  /** Synchronous click suppression: a second Start cannot mint a second paid analysis. */
  const selectingRef = useRef(false);
  const mountedRef = useRef(true);
  const mirrorRef = useRef(mirror);
  mirrorRef.current = mirror;
  const consentRef = useRef(consent);
  consentRef.current = consent;
  const cameraLiveRef = useRef(cameraLive);
  cameraLiveRef.current = cameraLive;
  const taskRef = useRef({ userGoal, context });
  taskRef.current = { userGoal, context };
  const billingRef = useRef(onSelectionBilling);
  billingRef.current = onSelectionBilling;
  const fenceRef = useRef({ sourceEpoch, sceneMode, cameraFrameSize: cameraFrameSizeKey, taskKey });
  fenceRef.current = { sourceEpoch, sceneMode, cameraFrameSize: cameraFrameSizeKey, taskKey };

  /**
   * Frames the camera has presented, kept current by the run's own rVFC observer. It is the pacer's
   * clock: the loop records it when it captures, and compares it after the response.
   */
  const presentedRef = useRef(0);
  const pacerRef = useRef<{ video: HTMLVideoElement; handle: number } | null>(null);

  /** Stop the rVFC observer. Safe to call when no observer is running. */
  const stopPacer = useCallback(() => {
    const pacer = pacerRef.current;
    if (!pacer) return;
    pacerRef.current = null;
    (pacer.video as FrameCallbackVideo).cancelVideoFrameCallback?.(pacer.handle);
  }, []);

  /**
   * Start the rVFC observer for this run: it re-registers itself every presented frame, so the count
   * advances for as long as the run is live and stops the moment it retires. Without a live observer
   * the loop falls back to the slow wait, which is correct but captures roughly every other frame.
   */
  const startPacer = useCallback(
    (video: HTMLVideoElement | null) => {
      stopPacer();
      if (!video) return;
      const target = video as FrameCallbackVideo;
      if (typeof target.requestVideoFrameCallback !== 'function') return;
      // This observer's own identity; the callback below closes over it.
      const pacer: Pacer = { video, handle: 0 };
      const onFrame = (_now: number, metadata: { presentedFrames: number }) => {
        // Fence by pacer INSTANCE, not by video element: the same <video> is reused across a restart or
        // a source change, so an element check would let a late callback from a retired pacer re-arm
        // itself and overwrite the live pacer's handle (and the clock) of the run that replaced it.
        if (pacerRef.current !== pacer) return;
        presentedRef.current = metadata.presentedFrames;
        pacer.handle = target.requestVideoFrameCallback!(onFrame);
      };
      pacerRef.current = pacer;
      pacer.handle = target.requestVideoFrameCallback(onFrame);
    },
    [stopPacer]
  );

  const clearVisible = useCallback(() => {
    shownCommitRef.current = null;
    setShown(null);
    liveTrack.clear();
  }, [liveTrack]);

  /** Synchronously retire the current attempt; `keepVisible` preserves the pair for manual seeding. */
  const retireRun = useCallback(
    (keepVisible: boolean) => {
      // The pacer belongs to the run: stop it even when there is no run left, so a run that already
      // cleared runRef cannot leave an observer re-registering forever.
      stopPacer();
      const run = runRef.current;
      runRef.current = null;
      if (!run) return;
      run.controller.abort();
      if (!keepVisible) clearVisible();
      // A kept still is for manual reseeding, not a live position: the live snapshot survives only when it
      // is a terminal state (no box by contract), so the status chip can still say why tracking ended.
      else if (!isTerminalTrackState(liveTrack.current()?.state ?? '')) liveTrack.clear();
      // Only a registered tracking run has a server-side run to release; a selection attempt never does.
      if (run.kind === 'track' && run.sessionId) {
        void trackControl({ session_id: run.sessionId, run_id: run.runId, action: 'stop' }).catch(() => {});
      }
    },
    [clearVisible, liveTrack, stopPacer]
  );

  /** Capture the current video frame into the exact uploaded bytes plus a display still. */
  const captureCurrentFrame = useCallback((): StillPair | null => {
    const video = videoRef.current;
    if (!video) return null;
    let snapshot;
    try {
      snapshot = captureVideoFrame(video, { mirror: mirrorRef.current });
    } catch {
      return null;
    }
    return {
      frameId: newOpaqueId('trk'),
      frameSeq: 0,
      imageDataUrl: snapshot.dataUrl,
      rawBase64: snapshot.rawBase64,
      jpeg: null,
      width: snapshot.width,
      height: snapshot.height,
      capturedAt: snapshot.capturedAt,
    };
  }, [videoRef]);

  /**
   * Freeze and encode the current frame for a tracking run, first in the capture worker's queue, and time
   * it for the run's pipelining estimates. Used for a fresh capture and for a prepared one alike, so both
   * fail the same way: "no usable frame yet" is transient, anything else is a real error.
   */
  const captureTrackFrame = useCallback(
    async (run: ActiveRun, video: HTMLVideoElement): Promise<TrackFrameCapture> => {
      const startedAt = performance.now();
      let snapshot: TrackFrameCapture;
      try {
        // Raw JPEG where the page and the server both can (v2), base64 for the JSON form otherwise.
        snapshot = await captureTrackFrameAsync(video, { mirror: mirrorRef.current, binary: trackFrameBinaryAvailable() });
      } catch (caught) {
        // Only "the camera has not produced a usable frame yet" is transient. A real capture failure
        // (a bounds rejection, or the encoder itself) must surface rather than being retried forever.
        if (caught instanceof CameraFrameUnavailableError) throw new CaptureUnavailableError();
        throw caught;
      }
      run.captureCost.add(performance.now() - startedAt);
      return snapshot;
    },
    []
  );

  const dispatchFrame = useCallback(
    async (run: ActiveRun): Promise<string | null> => {
      let imageBase64: string | null;
      let imageDataUrl: string | null;
      let jpeg: Blob | null = null;
      let width: number;
      let height: number;
      let capturedAt: number;
      let seedBox: Box | undefined;

      if (run.seedFrame) {
        ({ imageBase64, imageDataUrl, width, height, capturedAt } = run.seedFrame);
        // The seed was frozen when the user drew the box, so no frame is newer than it yet: the next
        // wait must be a real one rather than a fast-path skip.
        run.lastPresentedIdx = presentedRef.current;
        seedBox = run.seedBox;
        run.seedFrame = undefined;
        run.seedBox = undefined;
      } else {
        const video = videoRef.current;
        if (!video) throw new CaptureUnavailableError();
        let snapshot: TrackFrameCapture;
        let presentedAtCapture: number;
        // The frame prepared while the previous request was in flight, if it is still young enough.
        const taken = run.prefetch
          ? await run.prefetch.take(staleLimitMs(run.captureCost.get()), Date.now)
          : ({ kind: 'none' } as const);
        if (taken.kind === 'ready') {
          snapshot = taken.frame;
          presentedAtCapture = taken.presentedIdx;
        } else {
          if (taken.kind === 'stale') {
            // The prepared frame was too old to send. The fresh capture must still be NEWER than it.
            if (runRef.current !== run || run.controller.signal.aborted || !mountedRef.current) return null;
            await waitForPresentedFrame(video, () => presentedRef.current, taken.presentedIdx, run.controller.signal);
            if (runRef.current !== run || run.controller.signal.aborted || !mountedRef.current) return null;
          }
          // The pacer's clock as of the frame that is about to be frozen. Read HERE, before the encode:
          // reading it after would describe however far the camera advanced during the encode instead of
          // the frame actually captured.
          presentedAtCapture = presentedRef.current;
          snapshot = await captureTrackFrame(run, video);
        }

        // FENCE RE-CHECK IMMEDIATELY AFTER THE ASYNC ENCODE. The off-thread encode is the only long
        // await between "a frame was captured" and "bytes are minted and sent", so it can settle AFTER
        // this run was retired — a real Stop, a source/mirror/consent/task-text change, or a newer run
        // superseding this one. The encoded bytes are stale then and must be DISCARDED before a frameId
        // is minted, before any request is built, and before anything is sent. A prepared frame is held
        // to the same check: it was frozen under the run's fence, which may no longer hold.
        if (runRef.current !== run || run.controller.signal.aborted || !mountedRef.current) return null;
        if (runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }) !== run.fenceKey) return null;

        imageBase64 = snapshot.rawBase64;
        imageDataUrl = snapshot.rawBase64 === null ? null : `data:image/jpeg;base64,${snapshot.rawBase64}`;
        jpeg = snapshot.jpeg;
        width = snapshot.width;
        height = snapshot.height;
        // The freeze time the snapshot carries — not the moment its encode finished (nor, for a prepared
        // frame, the moment it is sent). A timestamp taken after the await would age a held or slow
        // encode as young as the encode was slow, so an old image would present as fresh and defeat the
        // freshness limit.
        capturedAt = snapshot.capturedAt;
        // The pacer's marker for THIS frame: after the response a higher count means a newer frame is
        // already available and the next capture need not wait for one.
        run.lastPresentedIdx = presentedAtCapture;
      }

      const frameId = newOpaqueId('trk');
      const frameSeq = run.frameSeq + 1;
      run.frameSeq = frameSeq;
      const pendingPair: TrackPair = { frameId, frameSeq, imageDataUrl, rawBase64: imageBase64, jpeg, width, height, capturedAt };

      // Pipelining: prepare the NEXT frame while this one is in flight, timed to be ready about when the
      // answer is due. It is only prepared here — it is sent by the next dispatch, after this answer, so
      // there is still exactly one request in flight.
      run.prefetch?.start(run.lastPresentedIdx, prefetchDelayMs(run.roundTrip.get(), run.captureCost.get()));
      const sentAt = performance.now();
      const answer = await trackFrame(
        { session_id: run.sessionId, run_id: run.runId, frame_id: frameId, frame_seq: frameSeq },
        jpeg ? { jpeg } : { base64: imageBase64 ?? '' },
        seedBox,
        run.controller.signal
      );
      run.roundTrip.add(performance.now() - sentAt);

      // Every post-await check: this run is still current, alive, mounted, and its scene fence holds.
      if (runRef.current !== run || run.controller.signal.aborted || !mountedRef.current) return null;
      if (runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }) !== run.fenceKey) return null;

      const decision = decideUpdate(
        { runId: answer.run_id, frameId: answer.frame_id, frameSeq: answer.frame_seq, version: answer.version },
        {
          activeRunId: run.runId,
          pairedFrameId: pendingPair.frameId,
          pairedFrameSeq: pendingPair.frameSeq,
          lastVersion: run.lastVersion,
        }
      );
      if (!decision.accept) return null;
      // A changed track_id is a new identity (re-acquisition is an explicit user action, never automatic):
      // the whole pair — image and box — is replaced from this one answer, nothing is carried across.
      run.trackingId = answer.track_id;
      run.lastVersion = answer.version;
      if (runRef.current === run && mountedRef.current) {
        // Fast path first, every accepted frame: the live overlay reads this outside React.
        liveTrack.publish({
          runId: answer.run_id,
          trackId: answer.track_id,
          generation: answer.generation,
          state: answer.state,
          box: isBoxDrawable(answer.box, answer.state) ? answer.box : null,
          capturedAt: pendingPair.capturedAt,
          version: answer.version,
          target: run.target,
        });
        // Diagnostic still: commit on any identity/state change, otherwise at most every interval.
        const key = `${answer.run_id}|${answer.track_id}|${answer.generation}|${answer.state}`;
        const now = performance.now();
        const last = shownCommitRef.current;
        if (!last || last.key !== key || now - last.at >= SHOWN_COMMIT_INTERVAL_MS) {
          shownCommitRef.current = { at: now, key };
          setShown({ pair: pendingPair, response: answer });
        }
      }
      return answer.state;
    },
    [captureTrackFrame, liveTrack, videoRef]
  );

  const runLoop = useCallback(
    async (run: ActiveRun) => {
      while (!run.controller.signal.aborted && runRef.current === run) {
        if (!cameraLiveRef.current || !consentRef.current) {
          retireRun(false);
          if (mountedRef.current) setPhase('idle');
          return;
        }
        let acceptedState: string | null = null;
        try {
          acceptedState = await dispatchFrame(run);
        } catch (caught) {
          if (run.controller.signal.aborted || runRef.current !== run) return;
          // A failed step restarts from a fresh capture, exactly as before pipelining: nothing prepared
          // around a failure is sent after it.
          run.prefetch?.discard();
          if (caught instanceof CaptureUnavailableError) {
            // The camera has not produced a usable frame yet: wait for the next one, do not fail the run.
            await nextVideoFrame(videoRef.current, run.controller.signal);
            continue;
          }
          const failure = toTrackError(caught);
          const action = decideError(failure.status, failure.code);
          if (action === 'drop_frame') {
            await nextVideoFrame(videoRef.current, run.controller.signal);
            continue;
          }
          // retire_run / unavailable / lost_start_race / unknown: the run cannot continue honestly.
          runRef.current = null;
          run.controller.abort();
          if (mountedRef.current) {
            setPhase('error');
            setError(failure);
            clearVisible();
          }
          return;
        }
        if (acceptedState !== null && isTerminalTrackState(acceptedState)) {
          // Sticky terminal state: the service will not continue this run on its own. Stop uploading,
          // release the server-side run, and keep the final honest pair visible for the user to reseed.
          retireRun(true);
          if (mountedRef.current) setPhase('idle');
          return;
        }
        // A frame being prepared already waits for a newer camera frame itself. Otherwise the latest-frame
        // pacer: skip the wait entirely when the camera presented a frame while the request was in
        // flight, so the rate stays set by the camera rather than by camera-plus-one.
        if (!run.prefetch?.active) {
          await waitForPresentedFrame(videoRef.current, () => presentedRef.current, run.lastPresentedIdx, run.controller.signal);
        }
      }
    },
    [clearVisible, dispatchFrame, retireRun]
  );

  /**
   * Mint a run and start it. The cancellation token is created and installed BEFORE the session await,
   * so a stop/consent-revoke/new-run landing during session creation already retires this attempt.
   */
  const beginRun = useCallback(
    async (target: string, seedFrame?: SeedFrame, seedBox?: Box) => {
      // Fail closed before any session creation or server run: consent and a live camera are the
      // preconditions, not something to discover after the awaits. Re-checked again below.
      if (!consentRef.current || !cameraLiveRef.current) {
        if (mountedRef.current) setPhase('idle');
        return;
      }
      retireRun(Boolean(seedFrame));
      if (!seedFrame) clearVisible();
      // A new run never inherits the previous run's live snapshot, even a kept terminal one.
      liveTrack.clear();
      setError(null);
      setPhase('starting');

      const runId = newOpaqueId('run');
      const controller = new AbortController();
      const run: ActiveRun = {
        kind: 'track',
        runId,
        target,
        sessionId: '',
        fenceKey: runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }),
        frameSeq: 0,
        lastVersion: 0,
        trackingId: null,
        controller,
        seedFrame,
        seedBox,
        lastPresentedIdx: presentedRef.current,
        prefetch: null,
        captureCost: new DurationEstimate(),
        roundTrip: new DurationEstimate(),
      };
      run.prefetch = new FramePrefetcher<TrackFrameCapture>(
        {
          waitForNewerFrame: (afterIdx, signal) =>
            waitForPresentedFrame(videoRef.current, () => presentedRef.current, afterIdx, signal),
          presentedIdx: () => presentedRef.current,
          capture: () => {
            const video = videoRef.current;
            return video ? captureTrackFrame(run, video) : Promise.reject(new CaptureUnavailableError());
          },
          sleep: (ms, signal) => sleepUnlessAborted(ms, signal),
        },
        controller.signal
      );
      runRef.current = run;
      // The observer lives exactly as long as the run, so the presented-frame clock advances only
      // while there is something to pace.
      startPacer(videoRef.current);
      setSelection(null);
      setTarget(target);

      let session = sessionId;
      if (!session) {
        try {
          session = await ensureSession();
        } catch (caught) {
          // Fenced exactly like fulfilment: a run retired while the session was being created (stop /
          // consent revoke / new run) must not surface a stale session error over the newer state.
          if (runRef.current !== run || controller.signal.aborted || !mountedRef.current) return;
          runRef.current = null;
          setPhase('error');
          setError(toTrackError(caught, 'session_failed'));
          return;
        }
      }
      // Re-check after BOTH awaits: consent, camera, fence, still-current, mounted.
      if (runRef.current !== run || controller.signal.aborted || !mountedRef.current) return;
      if (!consentRef.current || !cameraLiveRef.current) {
        runRef.current = null;
        controller.abort();
        if (mountedRef.current) setPhase('idle');
        if (!seedFrame) clearVisible();
        return;
      }
      if (runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }) !== run.fenceKey) {
        runRef.current = null;
        controller.abort();
        if (mountedRef.current) setPhase('idle');
        return;
      }
      run.sessionId = session;

      // Reserve the sequence number BEFORE dispatching, so two concurrent starts never share one.
      const startSeq = (startSeqCounter += 1);
      try {
        await trackControl(
          { session_id: session, run_id: runId, action: 'start', start_seq: startSeq, target },
          controller.signal
        );
      } catch (caught) {
        if (runRef.current !== run) return;
        runRef.current = null;
        controller.abort();
        if (mountedRef.current) {
          setPhase('error');
          setError(toTrackError(caught));
        }
        return;
      }
      if (runRef.current !== run || controller.signal.aborted || !mountedRef.current) return;
      setPhase('running');
      void runLoop(run);
    },
    [captureTrackFrame, clearVisible, ensureSession, liveTrack, retireRun, runLoop, sessionId, startPacer]
  );


  /**
   * Start a tracking run from a target the caller already knows (the guide lane's plan selected it with
   * its own paid call). Same fences and lifecycle as a run started through select(); no provider call.
   */
  const startWithTarget = useCallback(
    async (chosenTarget: string) => {
      const trimmed = chosenTarget.trim();
      if (!trimmed) return;
      setManualCapture(null);
      setSelection(null);
      await beginRun(trimmed);
    },
    [beginRun]
  );

  const stop = useCallback(() => {
    retireRun(false);
    setPhase('idle');
    setError(null);
  }, [retireRun]);

  /**
   * Startup analysis: one paid call that reads the captured frame plus the current task and chooses the
   * object to track. The attempt lives in the same fence slot as a run, so Stop / a source / mirror / task
   * / consent change aborts it and a late answer is discarded. A double click is suppressed synchronously
   * before any request, so at most one analysis is dispatched per click.
   */
  const select = useCallback(async () => {
    if (selectingRef.current) return;
    if (!consentRef.current || !cameraLiveRef.current) {
      if (mountedRef.current) setPhase('idle');
      return;
    }
    selectingRef.current = true;
    try {
      retireRun(false);
      clearVisible();
      setManualCapture(null);
      setError(null);
      setSelection(null);
      setTarget(null);
      setPhase('analyzing');

      const pair = captureCurrentFrame();
      if (!pair) {
        if (mountedRef.current) setPhase('idle');
        return;
      }
      const runId = newOpaqueId('run');
      const controller = new AbortController();
      const attempt: ActiveRun = {
        kind: 'select',
        runId,
        target: null,
        sessionId: '',
        fenceKey: runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }),
        frameSeq: 0,
        lastVersion: 0,
        trackingId: null,
        controller,
        seedFrame: undefined,
        seedBox: undefined,
        lastPresentedIdx: undefined,
        prefetch: null,
        captureCost: new DurationEstimate(),
        roundTrip: new DurationEstimate(),
      };
      runRef.current = attempt;

      let session = sessionId;
      let settled = false;
      let dispatched = false;
      try {
        if (!session) session = await ensureSession();
        // Re-check after the session await exactly like beginRun: stop / consent / fence / mounted.
        if (runRef.current !== attempt || controller.signal.aborted || !mountedRef.current) return;
        if (!consentRef.current || !cameraLiveRef.current) return;
        if (runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }) !== attempt.fenceKey) return;
        attempt.sessionId = session;

        dispatched = true;
        const answer = await selectTarget(
          {
            session_id: session,
            consent_ai: true,
            scene: { frame_id: pair.frameId, image_base64: pair.rawBase64 },
            user_goal: taskRef.current.userGoal.trim() || null,
            context: taskRef.current.context.trim() || null,
          },
          controller.signal
        );
        // A received answer was billed whether or not its result is used — fold it before any staleness
        // decision, so a late or fenced answer still settles once and never revives a run.
        settled = true;
        billingRef.current?.({
          kind: 'billed',
          provider: answer.provider,
          model: answer.model ?? null,
          usage: answer.usage ?? null,
        });
        if (runRef.current !== attempt || controller.signal.aborted || !mountedRef.current) return;
        if (!consentRef.current || !cameraLiveRef.current) return;
        if (runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }) !== attempt.fenceKey) return;
        // An answer to a frame this attempt is no longer about is discarded, and the attempt is retired
        // honestly instead of being left pending in `analyzing` forever.
        if (answer.frame_id !== pair.frameId) {
          runRef.current = null;
          controller.abort();
          if (mountedRef.current) setPhase('idle');
          return;
        }

        if (answer.status === 'selected' && answer.target) {
          const selectedTarget = answer.target;
          const seedFrame: SeedFrame = {
            imageBase64: pair.rawBase64,
            imageDataUrl: pair.imageDataUrl,
            width: pair.width,
            height: pair.height,
            capturedAt: pair.capturedAt,
          };
          runRef.current = null; // the attempt is spent; the run installs its own fence object
          await beginRun(selectedTarget, seedFrame);
          return;
        }
        runRef.current = null;
        controller.abort();
        if (mountedRef.current) {
          setSelection({
            status: answer.status === 'selected' ? 'uncertain' : answer.status,
            rationale: answer.rationale ?? null,
          });
          setPhase('idle');
        }
      } catch (caught) {
        // A dispatched attempt that never produced usage is unknown spend; a refusal before the provider
        // ran (no provider, consent, session, rate, validation) is not. A session failure before the
        // request left the browser is never billed.
        if (!settled && dispatched && failureMayHaveBilled(caught)) billingRef.current?.({ kind: 'unknown' });
        if (runRef.current !== attempt || controller.signal.aborted) return;
        if (!mountedRef.current) return;
        runRef.current = null;
        controller.abort();
        setPhase('error');
        setError(toTrackError(caught));
      }
    } finally {
      selectingRef.current = false;
    }
  }, [beginRun, captureCurrentFrame, clearVisible, ensureSession, retireRun, sessionId]);

  /** Manual selection entry: capture a fresh frame to draw on, whether or not a run ever existed. */
  const openManualSelect = useCallback(() => {
    retireRun(true);
    setPhase('idle');
    setError(null);
    setSelection(null);
    const pair = captureCurrentFrame();
    setManualCapture(
      pair ? { pair, fenceKey: runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }) } : null
    );
  }, [captureCurrentFrame, retireRun]);

  const cancelManualSelect = useCallback(() => setManualCapture(null), []);

  /** Manual seed: the drawn box on its own captured frame is the run's first frame (source="user"). */
  const reseed = useCallback(
    async (seedBox: Box) => {
      const capture = manualCapture;
      if (!capture) return;
      // The scene must be the one this frame was captured from; a source/mirror/task change since the
      // capture makes it stale, so the run is refused rather than seeded from the wrong scene.
      if (runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current }) !== capture.fenceKey) {
        setManualCapture(null);
        return;
      }
      setManualCapture(null);
      await beginRun(
        MANUAL_SELECTION_TARGET,
        {
          imageBase64: capture.pair.rawBase64,
          imageDataUrl: capture.pair.imageDataUrl,
          width: capture.pair.width,
          height: capture.pair.height,
          capturedAt: capture.pair.capturedAt,
        },
        seedBox
      );
    },
    [beginRun, manualCapture]
  );

  // Age readout: this hook's own capture→now clock over the retained pair. A 250 ms tick only re-renders;
  // the value is derived below, so it is already correct on the first paint that shows a new pair.
  useEffect(() => {
    if (!shown) return undefined;
    const timer = window.setInterval(() => ageTick((n) => n + 1), 250);
    return () => window.clearInterval(timer);
  }, [shown]);

  /** Derived, never stored: the retained pair's capture→now age, correct from the first render. */
  const ageMs = shown ? Math.max(0, Date.now() - shown.pair.capturedAt) : 0;

  // Consent revocation hides tracking immediately and retires the server run.
  useEffect(() => {
    if (consent) return;
    retireRun(false);
    // Also when no run is left (a kept terminal snapshot): revocation hides every tracking readout.
    liveTrack.clear();
    setManualCapture(null);
    setPhase('idle');
    setError(null);
  }, [consent, liveTrack, retireRun]);

  // A source / mirror / task / camera change retires the run immediately and clears the pair: the
  // geometry belonged to a scene that no longer exists, and nothing auto-restarts. A pending manual
  // drawing belongs to one scene too, so it is dropped by the same change even when no run exists.
  useEffect(() => {
    const current = runFenceKey({ ...fenceRef.current, mirrorView: mirrorRef.current });
    setManualCapture((previous) =>
      previous && (previous.fenceKey !== current || !cameraLive) ? null : previous
    );
    const run = runRef.current;
    if (!run) return;
    if (!cameraLive) {
      retireRun(false);
      setPhase('idle');
      return;
    }
    if (current === run.fenceKey) return;
    retireRun(false);
    setPhase('idle');
    setError(null);
  }, [sourceEpoch, sceneMode, cameraFrameSizeKey, taskKey, mirror, cameraLive, retireRun]);

  // Leaving the panel ends the run; the retireRun helper also sends the best-effort stop.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopPacer();
      liveTrack.clear();
      const run = runRef.current;
      runRef.current = null;
      run?.controller.abort();
      if (run?.kind === 'track' && run.sessionId) {
        void trackControl({ session_id: run.sessionId, run_id: run.runId, action: 'stop' }).catch(() => {});
      }
    };
  }, [liveTrack, stopPacer]);

  return {
    phase,
    shown,
    processing: phase === 'running',
    ageMs,
    error,
    selection,
    target,
    manualCapture,
    select,
    startWithTarget,
    reseed,
    openManualSelect,
    cancelManualSelect,
    stop,
    /** Per-frame live snapshot store (fast path); stable for the hook's lifetime. */
    liveTrack,
  };
}

/** The hook's result, passed from `App` to the panel, the camera stage and later consumers. */
export type ObjectTrack = ReturnType<typeof useObjectTrack>;

/**
 * Whether a failed startup analysis got far enough that a provider may have been billed. A transport or
 * abort failure, and the server's post-provider failure codes, are unknown spend; a refusal the server
 * makes before it calls a provider (no configured key, consent, session, rate, validation) is not.
 */
function failureMayHaveBilled(caught: unknown): boolean {
  // A non-API failure never reached a server verdict (transport error, abort): the request was dispatched,
  // so its spend is unknown.
  if (!(caught instanceof ApiError)) return true;
  // A session expiry can be returned by the post-await fence AFTER the provider answered, so it is
  // ambiguous rather than known-free.
  if (caught.status === 401) return true;
  if (caught.status === 502 || caught.status === 504) return true;
  return caught.status >= 500 && caught.status !== 503;
}

/**
 * One place that turns a thrown transport error into the panel's error state. A status-bearing error
 * (the API's own refusal) gets the described message; a plain `Error` (a precondition such as a missing
 * access code) keeps its own message instead of being relabelled as a network failure.
 */
function toTrackError(caught: unknown, fallbackCode = 'network'): TrackRunError {
  const hasStatus = typeof caught === 'object' && caught !== null && 'status' in caught;
  const status = hasStatus ? (caught as { status: number }).status : 0;
  const code =
    typeof caught === 'object' && caught !== null && 'code' in caught
      ? String((caught as { code: unknown }).code)
      : fallbackCode;
  const message = hasStatus
    ? describeTrackError(status, code)
    : caught instanceof Error
      ? caught.message
      : '추적 요청이 실패했습니다.';
  return { status, code, message };
}

function describeTrackError(status: number, code: string): string {
  if (code === 'session_timeout') return '세션 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요.';
  if (code === 'service_unavailable') {
    return '장면 분석을 사용할 수 없습니다(제공자 미설정). “대상 직접 지정”으로 대상을 그려주세요.';
  }
  if (code.startsWith('provider_') || code === 'invalid_provider_output') {
    return '장면 분석에 실패했습니다. 잠시 후 다시 시도하거나 “대상 직접 지정”으로 그려주세요.';
  }
  if (code === 'rate_limited') return '장면 분석 요청이 너무 잦습니다. 잠시 후 다시 시도해 주세요.';
  if (code === 'target_required') return '추적할 대상이 지정되지 않았습니다. 장면 분석을 먼저 실행해 주세요.';
  if (code === 'ai_consent_required') return 'AI 동의를 켠 뒤 다시 시도해 주세요.';
  if (code === 'session_mismatch' || code === 'session_required' || code === 'session_expired') {
    return '세션이 만료되었습니다. 페이지를 새로고침한 뒤 다시 시도해 주세요.';
  }
  if (status === 503 || status === 504) {
    return '추적기 서버에 연결할 수 없습니다. 추적기가 준비되면 페이지를 새로고침해 주세요.';
  }
  if (status === 0) {
    return '추적기 서버에 연결할 수 없습니다(네트워크 오류). 연결이 복구되면 페이지를 새로고침해 주세요.';
  }
  if (code === 'stale_start') return '추적 시작 요청이 뒤늦게 도착했습니다. 다시 시도해 주세요.';
  if (code === 'stale_run' || code === 'not_active_run') return '이 추적 실행은 더 이상 유효하지 않습니다.';
  if (code === 'seed_not_first_frame') return '대상 지정은 새 추적 실행의 첫 프레임에서만 적용됩니다.';
  if (code === 'session_failed') return '세션을 만들지 못했습니다.';
  return '추적 요청이 실패했습니다.';
}
