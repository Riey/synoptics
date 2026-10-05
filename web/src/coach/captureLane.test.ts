/**
 * The pure parts of the v2 capture path: the `?capture=` switch, the per-worker job order (priority by
 * kind, latest-wins for appearance samples), and the adaptive tracking upload size.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (/^\.\.?\//.test(specifier) && !/\.[cm]?[jt]sx?$/.test(specifier)) {
      try {
        return nextResolve(`${specifier}.ts`, context);
      } catch {
        // not a .ts file: resolve as written
      }
    }
    return nextResolve(specifier, context);
  },
});

const { parseCaptureSettings, DEFAULT_CAPTURE_SETTINGS } = await import('./captureMode.ts');
const { LaneQueue } = await import('./laneQueue.ts');
const { TrackScaleController, MIN_SAMPLES, MIN_DWELL_MS, SCALE_DOWN_ABOVE_MS, SCALE_UP_BELOW_MS } = await import(
  './trackScale.ts'
);

test('the capture switch defaults to v2 and ignores unknown values', () => {
  assert.deepEqual(parseCaptureSettings(''), DEFAULT_CAPTURE_SETTINGS);
  assert.deepEqual(DEFAULT_CAPTURE_SETTINGS, { mode: 'v2', trackScale: 'off', trackUpload: 'binary', v2Source: 'auto' });
  assert.equal(parseCaptureSettings('?capture=legacy').mode, 'legacy');
  assert.equal(parseCaptureSettings('?capture=WORKER1').mode, 'worker1');
  assert.equal(parseCaptureSettings('?capture=v3').mode, 'v2');
  assert.deepEqual(parseCaptureSettings('?capture=v2&trackscale=auto&trackupload=json&v2source=bitmap'), {
    mode: 'v2',
    trackScale: 'auto',
    trackUpload: 'json',
    v2Source: 'bitmap',
  });
  assert.equal(parseCaptureSettings('?trackscale=720').trackScale, 'off');
});

interface Job {
  kind: 'frame' | 'appearance';
  name: string;
}

function harness(latestWins: Array<Job['kind']>) {
  const ran: string[] = [];
  const superseded: string[] = [];
  const gates: Array<() => void> = [];
  const queue = new LaneQueue<Job['kind'], Job>(
    { order: ['frame', 'appearance'], latestWins },
    (job) =>
      new Promise<void>((resolve) => {
        ran.push(job.name);
        gates.push(resolve);
      }),
    (job) => superseded.push(job.name)
  );
  const release = async () => {
    gates.shift()?.();
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
  return { queue, ran, superseded, release };
}

test('a lane runs one job at a time, frames before samples, FIFO within a kind', async () => {
  const { queue, ran, release } = harness([]);
  queue.push({ kind: 'appearance', name: 'a1' });
  queue.push({ kind: 'appearance', name: 'a2' });
  queue.push({ kind: 'frame', name: 'f1' });
  queue.push({ kind: 'frame', name: 'f2' });
  assert.deepEqual(ran, ['a1'], 'the first job starts at once and is never preempted');
  assert.equal(queue.waiting, 3);
  for (let i = 0; i < 4; i += 1) await release();
  assert.deepEqual(ran, ['a1', 'f1', 'f2', 'a2']);
  assert.equal(queue.busy, false);
});

test('latest-wins kinds keep only the newest waiting job; the running one is never superseded', async () => {
  const { queue, ran, superseded, release } = harness(['appearance']);
  queue.push({ kind: 'appearance', name: 'a1' });
  queue.push({ kind: 'appearance', name: 'a2' });
  queue.push({ kind: 'frame', name: 'f1' });
  queue.push({ kind: 'appearance', name: 'a3' });
  assert.deepEqual(superseded, ['a2']);
  queue.push({ kind: 'frame', name: 'f2' });
  assert.deepEqual(superseded, ['a2'], 'frames are never superseded');
  for (let i = 0; i < 4; i += 1) await release();
  assert.deepEqual(ran, ['a1', 'f1', 'f2', 'a3']);
});

test('a runner that throws does not wedge the lane', async () => {
  const ran: string[] = [];
  const queue = new LaneQueue<Job['kind'], Job>(
    { order: ['frame'], latestWins: [] },
    async (job) => {
      ran.push(job.name);
      throw new Error('bug');
    },
    () => {}
  );
  queue.push({ kind: 'frame', name: 'f1' });
  queue.push({ kind: 'frame', name: 'f2' });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(ran, ['f1', 'f2']);
});

test('the adaptive tracking size is off unless asked, and 960 forces it', () => {
  const off = new TrackScaleController('off');
  for (let i = 0; i < 50; i += 1) off.observe(200, i * 1000, false);
  assert.equal(off.longSide(), null);
  assert.equal(new TrackScaleController('960').longSide(), 960);
});

test('auto scales down after a sustained slow capture and back up only when clearly cheap', () => {
  const auto = new TrackScaleController('auto');
  let t = 10_000;
  // One slow outlier among fast captures changes nothing.
  for (let i = 0; i < MIN_SAMPLES * 2; i += 1) auto.observe(i === 3 ? 400 : 10, (t += 100), false);
  assert.equal(auto.longSide(), null);
  // Sustained work above the threshold: down once enough samples agree.
  for (let i = 0; i < MIN_SAMPLES; i += 1) auto.observe(SCALE_DOWN_ABOVE_MS + 30, (t += 100), false);
  assert.equal(auto.longSide(), 960);
  // A sample that was taken at the old size (in flight across the change) is ignored.
  auto.observe(1, (t += 100), false);
  // Between the two thresholds, before and after the dwell time: stays (hysteresis).
  for (let i = 0; i < MIN_SAMPLES; i += 1) auto.observe(SCALE_UP_BELOW_MS + 5, (t += 100), true);
  t += MIN_DWELL_MS;
  for (let i = 0; i < MIN_SAMPLES * 2; i += 1) auto.observe(SCALE_UP_BELOW_MS + 5, (t += 100), true);
  assert.equal(auto.longSide(), 960);
  // Clearly cheap, after the dwell: back to full size.
  for (let i = 0; i < MIN_SAMPLES * 3; i += 1) auto.observe(2, (t += 100), true);
  assert.equal(auto.longSide(), null);
});

test('no change within the dwell time after a change, however clear the samples', () => {
  const auto = new TrackScaleController('auto');
  let t = 0;
  for (let i = 0; i < MIN_SAMPLES; i += 1) auto.observe(SCALE_DOWN_ABOVE_MS * 2, (t += 100), false);
  assert.equal(auto.longSide(), 960);
  for (let i = 0; i < MIN_SAMPLES * 2; i += 1) auto.observe(0.5, (t += 100), true);
  assert.ok(t < MIN_DWELL_MS + MIN_SAMPLES * 100);
  assert.equal(auto.longSide(), 960);
});
