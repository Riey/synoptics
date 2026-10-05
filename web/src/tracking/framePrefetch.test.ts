/**
 * The tracking loop's pipelining (`framePrefetch.ts`): what a prepared frame may be, when it is thrown
 * away, and — with a loop that runs the same steps in the same order as `useObjectTrack`'s
 * `dispatchFrame`/`runLoop` on a simulated clock — that there is never more than one request in flight
 * and that the period approaches max(capture, round trip).
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CAMERA_FRAME_MS,
  DurationEstimate,
  FramePrefetcher,
  prefetchDelayMs,
  staleLimitMs,
  type FramePrefetchDeps,
} from './framePrefetch.ts';

interface Frame {
  capturedAt: number;
  idx: number;
}

const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

/** A controllable dependency set: every wait resolves only when the test says so. */
function manualDeps() {
  let presented = 0;
  let captures = 0;
  const waits: Array<{ afterIdx: number | undefined; resolve: () => void }> = [];
  const sleeps: Array<{ ms: number; resolve: () => void }> = [];
  const pendingCaptures: Array<{ resolve: (frame: Frame) => void; reject: (error: unknown) => void }> = [];
  const deps: FramePrefetchDeps<Frame> = {
    waitForNewerFrame: (afterIdx, signal) =>
      new Promise<void>((resolve) => {
        if (afterIdx !== undefined && presented > afterIdx) return resolve();
        waits.push({ afterIdx, resolve });
        signal.addEventListener('abort', () => resolve(), { once: true });
      }),
    presentedIdx: () => presented,
    capture: () => {
      captures += 1;
      return new Promise<Frame>((resolve, reject) => pendingCaptures.push({ resolve, reject }));
    },
    sleep: (ms, signal) =>
      new Promise<void>((resolve) => {
        sleeps.push({ ms, resolve });
        signal.addEventListener('abort', () => resolve(), { once: true });
      }),
  };
  return {
    deps,
    setPresented: (value: number) => {
      presented = value;
    },
    captures: () => captures,
    waits,
    sleeps,
    pendingCaptures,
  };
}

test('the stale limit is twice the capture cost, never under one camera frame', () => {
  assert.equal(staleLimitMs(null), CAMERA_FRAME_MS);
  assert.equal(staleLimitMs(5), CAMERA_FRAME_MS);
  assert.equal(staleLimitMs(62), 124);
});

test('the prefetch delay aims the end of the capture at the expected answer', () => {
  assert.equal(prefetchDelayMs(null, 40), 0, 'no round trip measured yet: start at once');
  assert.equal(prefetchDelayMs(55, null), 0, 'no capture measured yet: start at once');
  assert.equal(prefetchDelayMs(55, 62), 0, 'capture slower than the round trip: start at once');
  assert.equal(prefetchDelayMs(200, 40), 160);
});

test('the duration estimate starts at the first sample and moves 0.3 of the way to each next one', () => {
  const estimate = new DurationEstimate();
  assert.equal(estimate.get(), null);
  estimate.add(100);
  assert.equal(estimate.get(), 100);
  estimate.add(200);
  assert.equal(estimate.get(), 130);
  estimate.add(Number.NaN);
  estimate.add(-5);
  assert.equal(estimate.get(), 130, 'nonsense samples are ignored');
});

test('a prepared frame is a newer camera frame than the one sent, captured once, and returned ready', async () => {
  const m = manualDeps();
  const prefetch = new FramePrefetcher(m.deps, new AbortController().signal);
  m.setPresented(5);
  prefetch.start(5, 0);
  await flush();
  assert.equal(m.captures(), 0, 'no newer frame yet: nothing is captured');
  assert.equal(m.waits[0].afterIdx, 5);
  m.setPresented(6);
  m.waits[0].resolve();
  await flush();
  assert.equal(m.captures(), 1);
  m.pendingCaptures[0].resolve({ capturedAt: 1000, idx: 6 });
  const taken = await prefetch.take(staleLimitMs(40), () => 1050);
  assert.deepEqual(taken, { kind: 'ready', frame: { capturedAt: 1000, idx: 6 }, presentedIdx: 6 });
  assert.equal(prefetch.active, false, 'a taken frame is consumed');
  assert.deepEqual(await prefetch.take(1000, () => 1050), { kind: 'none' });
  assert.equal(m.captures(), 1, 'one start, one capture');
});

test('a prepared frame older than the limit is returned as stale, never as a frame to send', async () => {
  const m = manualDeps();
  const prefetch = new FramePrefetcher(m.deps, new AbortController().signal);
  m.setPresented(9);
  prefetch.start(8, 0);
  await flush();
  m.pendingCaptures[0].resolve({ capturedAt: 1000, idx: 9 });
  const limit = staleLimitMs(40);
  const taken = await prefetch.take(limit, () => 1000 + limit + 1);
  assert.equal(taken.kind, 'stale');
  assert.equal(taken.kind === 'stale' && taken.presentedIdx, 9, 'the fresh capture must be newer than this frame');
});

test('the answer arriving during the delay wakes the preparation instead of waiting the delay out', async () => {
  const m = manualDeps();
  const prefetch = new FramePrefetcher(m.deps, new AbortController().signal);
  m.setPresented(3);
  prefetch.start(2, 500);
  await flush();
  assert.equal(m.sleeps.length, 1);
  assert.equal(m.captures(), 0, 'still in the delay');
  const taking = prefetch.take(1000, () => 10);
  await flush();
  assert.equal(m.captures(), 1, 'woken: the capture starts without the sleep resolving');
  m.pendingCaptures[0].resolve({ capturedAt: 5, idx: 3 });
  assert.equal((await taking).kind, 'ready');
});

test('retiring the run (abort) discards the preparation: no capture after it, a running capture is dropped', async () => {
  const m = manualDeps();
  const controller = new AbortController();
  const prefetch = new FramePrefetcher(m.deps, controller.signal);
  m.setPresented(1);
  prefetch.start(0, 500);
  controller.abort();
  await flush();
  assert.equal(m.captures(), 0, 'aborted during the delay: nothing is captured');
  assert.equal(prefetch.active, false);
  prefetch.start(0, 0);
  await flush();
  assert.equal(m.captures(), 0, 'a retired run cannot start another preparation');

  const live = new AbortController();
  const running = new FramePrefetcher(m.deps, live.signal);
  running.start(0, 0);
  await flush();
  assert.equal(m.captures(), 1);
  const taking = running.take(1000, () => 0);
  live.abort();
  m.pendingCaptures[0].resolve({ capturedAt: 0, idx: 1 });
  assert.deepEqual(await taking, { kind: 'none' }, 'the late result of a retired run is never handed out');
});

test('a new start replaces the previous preparation and its late result is ignored', async () => {
  const m = manualDeps();
  const prefetch = new FramePrefetcher(m.deps, new AbortController().signal);
  m.setPresented(1);
  prefetch.start(0, 0);
  await flush();
  m.setPresented(2);
  prefetch.start(1, 0);
  await flush();
  assert.equal(m.captures(), 2);
  m.pendingCaptures[0].resolve({ capturedAt: 0, idx: 1 });
  m.pendingCaptures[1].resolve({ capturedAt: 1, idx: 2 });
  const taken = await prefetch.take(1000, () => 2);
  assert.equal(taken.kind === 'ready' && taken.frame.idx, 2);
});

test('a capture failure is rethrown by take, to be handled like a failed fresh capture', async () => {
  const m = manualDeps();
  const prefetch = new FramePrefetcher(m.deps, new AbortController().signal);
  m.setPresented(1);
  prefetch.start(0, 0);
  await flush();
  m.pendingCaptures[0].reject(new Error('encoder broke'));
  await assert.rejects(prefetch.take(1000, () => 0), /encoder broke/);
  assert.equal(prefetch.active, false);
});

// ------------------------------------------------------------------------------- the loop, simulated

/** Discrete-event clock: timers fire in time order, with all microtasks flushed between them. */
class SimClock {
  now = 0;
  private timers: Array<{ at: number; seq: number; fire: () => void }> = [];
  private seq = 0;

  sleep(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise<void>((resolve) => {
      const timer = { at: this.now + Math.max(0, ms), seq: (this.seq += 1), fire: resolve };
      this.timers.push(timer);
      signal?.addEventListener(
        'abort',
        () => {
          this.timers = this.timers.filter((t) => t !== timer);
          resolve();
        },
        { once: true }
      );
    });
  }

  async runUntil(end: number): Promise<void> {
    await flush();
    while (this.timers.length) {
      this.timers.sort((a, b) => a.at - b.at || a.seq - b.seq);
      const next = this.timers[0];
      if (next.at > end) break;
      this.timers.shift();
      this.now = next.at;
      next.fire();
      await flush();
    }
    this.now = end;
  }
}

interface SimResult {
  maxInFlight: number;
  /** Prepared frames thrown away as too old. */
  stale: number;
  sends: Array<{ sentAt: number; answeredAt: number; capturedAt: number; idx: number }>;
}

/**
 * The loop of `useObjectTrack`, step for step: take the prepared frame (or capture fresh, waiting for a
 * newer camera frame after a stale one), start preparing the next, send and await the answer, and wait
 * for a newer camera frame only when nothing is being prepared.
 */
async function simulate(
  captureMs: number,
  roundTripMs: number | ((sendIndex: number) => number),
  pipelined: boolean,
  durationMs = 6000
): Promise<SimResult> {
  const rttOf = typeof roundTripMs === 'number' ? () => roundTripMs : roundTripMs;
  const clock = new SimClock();
  const presented = () => Math.floor(clock.now / CAMERA_FRAME_MS);
  const waitForNewerFrame = async (afterIdx: number | undefined, signal?: AbortSignal) => {
    if (afterIdx !== undefined && presented() > afterIdx) return;
    const nextFrameAt = (presented() + 1) * CAMERA_FRAME_MS;
    await clock.sleep(nextFrameAt - clock.now + 1e-6, signal);
  };
  const captureCost = new DurationEstimate();
  const roundTrip = new DurationEstimate();
  const capture = async (): Promise<Frame> => {
    const startedAt = clock.now;
    const frame = { capturedAt: clock.now, idx: presented() };
    await clock.sleep(captureMs);
    captureCost.add(clock.now - startedAt);
    return frame;
  };
  const controller = new AbortController();
  const prefetch = pipelined
    ? new FramePrefetcher<Frame>(
        { waitForNewerFrame, presentedIdx: presented, capture, sleep: (ms, signal) => clock.sleep(ms, signal) },
        controller.signal
      )
    : null;

  const result: SimResult = { maxInFlight: 0, stale: 0, sends: [] };
  let inFlight = 0;
  let lastIdx: number | undefined = presented();
  const loop = async () => {
    while (!controller.signal.aborted) {
      const taken = prefetch ? await prefetch.take(staleLimitMs(captureCost.get()), () => clock.now) : { kind: 'none' as const };
      let frame: Frame;
      if (taken.kind === 'ready') {
        frame = taken.frame;
      } else {
        if (taken.kind === 'stale') result.stale += 1;
        if (taken.kind === 'stale') await waitForNewerFrame(taken.presentedIdx);
        frame = await capture();
      }
      if (controller.signal.aborted) return;
      lastIdx = frame.idx;
      prefetch?.start(lastIdx, prefetchDelayMs(roundTrip.get(), captureCost.get()));
      inFlight += 1;
      result.maxInFlight = Math.max(result.maxInFlight, inFlight);
      const sentAt = clock.now;
      await clock.sleep(rttOf(result.sends.length));
      inFlight -= 1;
      roundTrip.add(clock.now - sentAt);
      result.sends.push({ sentAt, answeredAt: clock.now, capturedAt: frame.capturedAt, idx: frame.idx });
      if (!prefetch?.active) await waitForNewerFrame(lastIdx);
    }
  };
  const done = loop();
  await clock.runUntil(durationMs);
  controller.abort();
  await clock.runUntil(durationMs + 1000);
  await done;
  return result;
}

function steadyPeriod(result: SimResult): number {
  const sends = result.sends.slice(10); // past the estimates' warm-up
  return (sends[sends.length - 1].sentAt - sends[0].sentAt) / (sends.length - 1);
}

function peakBoxAge(result: SimResult): number {
  // What the status chip sees: a box's age just before the NEXT answer replaces it (freeze → next answer).
  const sends = result.sends.slice(10);
  return Math.max(...sends.slice(1).map((s, i) => s.answeredAt - sends[i].capturedAt));
}

for (const [label, captureMs, roundTripMs] of [
  ['Windows Edge laptop (capture 62, round trip 55)', 62, 55],
  ['MacBook (capture 18, round trip 55)', 18, 55],
  ['slow tracker (capture 40, round trip 200)', 40, 200],
] as const) {
  test(`pipelined loop, ${label}: one request in flight, newer frames only, period ≈ max(capture, round trip)`, async () => {
    const serial = await simulate(captureMs, roundTripMs, false);
    const pipelined = await simulate(captureMs, roundTripMs, true);

    assert.equal(pipelined.maxInFlight, 1, 'never more than one request in flight');
    for (let i = 1; i < pipelined.sends.length; i += 1) {
      assert.ok(pipelined.sends[i].idx > pipelined.sends[i - 1].idx, 'every sent frame is newer than the last');
      assert.ok(pipelined.sends[i].sentAt >= pipelined.sends[i - 1].answeredAt, 'a send waits for the previous answer');
    }

    const period = steadyPeriod(pipelined);
    const bound = Math.max(captureMs, roundTripMs);
    // Up to one camera frame on top: the prepared frame waits for a newly presented frame.
    assert.ok(period <= bound + CAMERA_FRAME_MS + 1, `period ${period.toFixed(1)} ms vs max(capture, rtt) ${bound} ms`);
    assert.ok(period < steadyPeriod(serial), `pipelined ${period.toFixed(1)} < serial ${steadyPeriod(serial).toFixed(1)}`);
    // And the box the user sees is never older than with the serial loop.
    assert.ok(
      peakBoxAge(pipelined) <= peakBoxAge(serial),
      `peak box age ${peakBoxAge(pipelined).toFixed(1)} vs serial ${peakBoxAge(serial).toFixed(1)}`
    );
  });
}

test('pipelined loop with a jittery tracker (every 5th answer takes 300 ms): stale frames are re-captured, never sent', async () => {
  const jitter = (i: number) => (i % 5 === 4 ? 300 : 55);
  const serial = await simulate(62, jitter, false);
  const pipelined = await simulate(62, jitter, true);
  assert.equal(pipelined.maxInFlight, 1);
  assert.ok(pipelined.stale > 0, 'the slow answers leave the prepared frame too old');
  for (const send of pipelined.sends.slice(10)) {
    // The estimate has settled near the true 62 ms capture cost by now.
    assert.ok(send.sentAt - send.capturedAt <= staleLimitMs(62) + 1, 'no frame older than the limit is sent');
  }
  assert.ok(steadyPeriod(pipelined) < steadyPeriod(serial));
  assert.ok(peakBoxAge(pipelined) <= peakBoxAge(serial), `${peakBoxAge(pipelined)} vs ${peakBoxAge(serial)}`);
});
