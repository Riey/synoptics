/**
 * The v2 capture path's routing and contracts (`captureV2.ts`), with fake workers, bitmaps and camera
 * stream: a worker per lane; frames read by the worker itself where the engine can hand it the camera track
 * (`MediaStreamTrackProcessor`), a main-thread bitmap otherwise; the fallbacks between the two; the
 * appearance sample's raw-coordinate crop and its back-pressure skip; `capturedAt` as the freeze time.
 *
 * Pixel equivalence of the worker's output with the other paths needs a real browser (measured in
 * headless Chromium, see the commit message / report).
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

type Message = Record<string, unknown> & { id?: string; kind: string };
type Reply = Record<string, unknown> | null;

const env = {
  workers: [] as FakeWorker[],
  bitmapCalls: [] as Array<{ args: unknown[]; at: number }>,
  reply: (message: Message, _worker: FakeWorker): Reply => defaultReply(message),
  replyDelayMs: 0,
};

const STAGES = { postMs: 1, queueMs: 0, waitMs: 0, drawMs: 2, appearanceMs: 0, encodeMs: 3, b64Ms: 0, workMs: 5 };

function defaultReply(message: Message): Reply {
  const stages = { ...STAGES, doneAbs: performance.timeOrigin + performance.now() };
  if (message.kind === 'frame') {
    return {
      id: message.id,
      ok: 'frame',
      blob: { size: 3 },
      rawBase64: message.base64 ? 'QUJD' : null,
      width: 1280,
      height: 720,
      byteLength: 3,
      appearanceGrid: message.appearance ? new Float32Array(24 * 24).fill(0.75) : null,
      frameWall: message.from === 'stream' ? Date.now() - 20 : null,
      frameAbs: message.from === 'stream' ? performance.timeOrigin + performance.now() - 20 : null,
      stages,
    };
  }
  if (message.kind === 'appearance') return { id: message.id, ok: 'appearance', grid: new Float32Array(24 * 24), stages };
  return null;
}

class FakeWorker {
  readonly name: string;
  readonly posted: Message[] = [];
  readonly transfers: unknown[][] = [];
  terminated = false;
  private readonly listeners = new Map<string, Set<(event: unknown) => void>>();
  constructor(_url: unknown, options: { name?: string }) {
    this.name = options.name ?? '';
    env.workers.push(this);
  }
  addEventListener(type: string, listener: (event: unknown) => void): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(listener);
  }
  postMessage(message: Message, transfer: unknown[]): void {
    this.posted.push(message);
    this.transfers.push(transfer);
    const reply = env.reply(message, this);
    if (!reply) return;
    setTimeout(() => this.emit({ data: reply }), env.replyDelayMs);
  }
  emit(event: unknown): void {
    for (const listener of [...(this.listeners.get('message') ?? [])]) listener(event);
  }
  terminate(): void {
    this.terminated = true;
  }
}

const globals = globalThis as unknown as Record<string, unknown>;
const saved = new Map<string, unknown>();
function setGlobal(key: string, value: unknown): void {
  if (!saved.has(key)) saved.set(key, globals[key]);
  globals[key] = value;
}
function clearGlobal(key: string): void {
  if (!saved.has(key)) saved.set(key, globals[key]);
  delete globals[key];
}

function OffscreenCanvasStub() {}
OffscreenCanvasStub.prototype.convertToBlob = () => {};

class FakeMediaStream {
  readonly tracks: Array<{ id: string; readyState: string }>;
  constructor(tracks: Array<{ id: string; readyState: string }>) {
    this.tracks = tracks;
  }
  getVideoTracks() {
    return this.tracks;
  }
}
class FakeProcessor {
  readonly readable: { stream: string };
  constructor(init: { track: { id: string } }) {
    this.readable = { stream: init.track.id };
  }
}

setGlobal('Worker', FakeWorker);
setGlobal('OffscreenCanvas', OffscreenCanvasStub);
setGlobal('createImageBitmap', async (...args: unknown[]) => {
  env.bitmapCalls.push({ args, at: Date.now() });
  return { width: 1280, height: 720, close() {} };
});

test.after(() => {
  for (const [key, value] of saved) {
    if (value === undefined) delete globals[key];
    else globals[key] = value;
  }
});

let moduleCase = 0;
async function fresh() {
  env.workers = [];
  env.bitmapCalls = [];
  env.reply = (message) => defaultReply(message);
  env.replyDelayMs = 0;
  moduleCase += 1;
  return import(`./captureV2.ts?case=${moduleCase}`);
}

function withStream(on: boolean, trackId = 'cam-1') {
  if (on) {
    setGlobal('MediaStream', FakeMediaStream);
    setGlobal('MediaStreamTrackProcessor', FakeProcessor);
    setGlobal('VideoFrame', function VideoFrame() {});
  } else {
    clearGlobal('MediaStream');
    clearGlobal('MediaStreamTrackProcessor');
    clearGlobal('VideoFrame');
  }
  const track = { id: trackId, readyState: 'live' };
  return {
    readyState: 4,
    videoWidth: 1280,
    videoHeight: 720,
    srcObject: on ? new FakeMediaStream([track]) : null,
    track,
  } as unknown as HTMLVideoElement & { track: { id: string; readyState: string } };
}

const BOX = { x: 0.1, y: 0.2, width: 0.3, height: 0.4 };

test('tracking and guide captures go to two different workers; no luma grid, base64 only when asked', async () => {
  const v2 = await fresh();
  const video = withStream(false);
  const track = await v2.captureFrameV2(video, { lane: 'track', mirror: true, base64: false });
  const guide = await v2.captureFrameV2(video, { lane: 'guide', mirror: true, base64: true, appearanceBox: BOX });
  assert.deepEqual(env.workers.map((worker) => worker.name), ['track', 'guide']);
  const [trackJob] = env.workers[0].posted;
  const [guideJob] = env.workers[1].posted;
  assert.equal(trackJob.kind, 'frame');
  assert.equal(trackJob.from, 'bitmap');
  assert.equal(trackJob.distinct, true, 'a tracking frame is never the previous one again');
  assert.equal(trackJob.base64, false);
  assert.equal(trackJob.maxLongSide, null, 'full size unless the adaptive size is on');
  assert.equal(trackJob.mirror, true);
  assert.equal(trackJob.quality, 0.88);
  assert.equal('luma' in trackJob || 'lumaGrid' in trackJob, false);
  assert.equal(env.workers[0].transfers[0].length, 1, 'the bitmap is transferred');
  assert.equal(guideJob.distinct, false);
  assert.equal(guideJob.base64, true);
  assert.deepEqual((guideJob.appearance as { box: unknown }).box, BOX);
  assert.equal(track.rawBase64, null);
  assert.equal(guide.rawBase64, 'QUJD');
  assert.equal(guide.appearanceGrid?.length, 24 * 24);
  // The freeze time is read beside createImageBitmap, not when the reply lands.
  assert.ok(track.capturedAt <= env.bitmapCalls[0].at);
  assert.equal(track.source, 'bitmap');
});

test('where the engine can, the worker reads the camera track itself: no main-thread bitmap', async () => {
  const v2 = await fresh();
  const video = withStream(true);
  const before = Date.now();
  const frame = await v2.captureFrameV2(video, { lane: 'track', mirror: false, base64: false });
  const [source, job] = env.workers[0].posted;
  assert.equal(source.kind, 'source');
  assert.equal(source.sourceId, 'cam-1');
  assert.deepEqual(env.workers[0].transfers[0], [source.readable], 'the readable is transferred');
  assert.equal(job.from, 'stream');
  assert.equal(env.bitmapCalls.length, 0);
  assert.equal(frame.source, 'stream');
  // capturedAt is when the frame reached the worker (20 ms before its reply here), not the reply time.
  assert.ok(frame.capturedAt <= before);
  assert.ok(frame.frozenAt <= performance.now() - 15);

  await v2.captureFrameV2(video, { lane: 'track', mirror: false, base64: false });
  assert.equal(env.workers[0].posted.filter((message) => message.kind === 'source').length, 1, 'attached once per track');
  await v2.captureFrameV2(video, { lane: 'guide', mirror: false, base64: true });
  assert.equal(env.workers[1].posted[0].kind, 'source', 'each lane reads its own processor');
});

test('a stream with no frame falls back to a bitmap, and a new camera track re-attaches', async () => {
  const v2 = await fresh();
  const video = withStream(true);
  env.reply = (message) => (message.kind === 'frame' && message.from === 'stream' ? { id: message.id, unavailable: 'ended' } : defaultReply(message));
  const frame = await v2.captureFrameV2(video, { lane: 'guide', mirror: false, base64: true });
  assert.equal(frame.source, 'bitmap');
  assert.equal(env.bitmapCalls.length, 1);
  // The ended source is forgotten: the next capture attaches again (here: to a new camera's track).
  env.reply = (message) => defaultReply(message);
  (video as unknown as { srcObject: FakeMediaStream }).srcObject = new FakeMediaStream([{ id: 'cam-2', readyState: 'live' }]);
  await v2.captureFrameV2(video, { lane: 'guide', mirror: false, base64: true });
  const sources = env.workers[0].posted.filter((message) => message.kind === 'source').map((message) => message.sourceId);
  assert.deepEqual(sources, ['cam-1', 'cam-2']);
});

test('a tracking frame with no newer camera frame in time is transient, never a repeat', async () => {
  const v2 = await fresh();
  const video = withStream(true);
  env.reply = (message) => (message.kind === 'frame' ? { id: message.id, unavailable: 'no_frame' } : defaultReply(message));
  await assert.rejects(v2.captureFrameV2(video, { lane: 'track', mirror: false, base64: false }), v2.V2FrameUnavailableError);
  assert.equal(env.bitmapCalls.length, 0, 'no bitmap of the same displayed frame is sent instead');
  // A guide capture is not held to "newer": it falls back to a bitmap.
  env.reply = (message) =>
    message.kind === 'frame' && message.from === 'stream' ? { id: message.id, unavailable: 'no_frame' } : defaultReply(message);
  const guide = await v2.captureFrameV2(video, { lane: 'guide', mirror: false, base64: true });
  assert.equal(guide.source, 'bitmap');
});

test('an appearance sample without the stream snapshots only the crop, in raw (unmirrored) pixels', async () => {
  const v2 = await fresh();
  const video = withStream(false);
  await v2.sampleAppearanceV2(video, BOX, true);
  const [, x, y, width, height] = env.bitmapCalls[0].args as number[];
  // appearanceCrop with pad 0.15 in the mirrored frame: x 0.055..0.445 → px 70..570 (mirrored: 710..1210),
  // y 0.14..0.66 → px 100..476.
  assert.deepEqual([x, y, width, height], [1280 - 570, 100, 500, 376]);
  const job = env.workers[0].posted[0];
  assert.equal(env.workers[0].name, 'guide');
  assert.equal(job.kind, 'appearance');
  assert.equal(job.from, 'bitmap');
  assert.equal(job.mirror, true, 'the worker mirrors the crop while reducing it');
});

test('an appearance sample is skipped while the tracking lane is behind, and when superseded', async () => {
  const v2 = await fresh();
  const video = withStream(false);
  // One quick tracking capture measures the lane's cost (~0 ms here).
  await v2.captureFrameV2(video, { lane: 'track', mirror: false, base64: false });
  // The next one never answers: once it is later than its own cost (≥ one camera frame), the lane is behind.
  env.reply = (message) => (message.kind === 'frame' ? null : defaultReply(message));
  void v2.captureFrameV2(video, { lane: 'track', mirror: false, base64: false }).catch(() => {});
  assert.ok(await v2.sampleAppearanceV2(video, BOX, false), 'not behind yet');
  await new Promise((resolve) => setTimeout(resolve, 50));
  await assert.rejects(v2.sampleAppearanceV2(video, BOX, false), v2.CaptureSkippedError);

  const again = await fresh();
  env.reply = (message) => (message.kind === 'appearance' ? { id: message.id, superseded: true } : defaultReply(message));
  await assert.rejects(again.sampleAppearanceV2(withStream(false), BOX, false), again.CaptureSkippedError);
});

test('a worker failure rejects the request and retires the worker; an oversized reply is refused', async () => {
  const v2 = await fresh();
  const video = withStream(false);
  env.reply = (message) => ({ id: message.id, error: 'encode failed' });
  await assert.rejects(v2.captureFrameV2(video, { lane: 'guide', mirror: false, base64: true }), /encode failed/);
  assert.equal(env.workers[0].terminated, true);
  env.reply = (message) => ({ ...defaultReply(message), byteLength: 1_600_000 });
  await assert.rejects(v2.captureFrameV2(video, { lane: 'guide', mirror: false, base64: true }), /1\.5MB/);
  assert.equal(env.workers.length, 2, 'a fresh worker after the failed one');
});

test('a camera with no frame yet is the transient error before anything is posted', async () => {
  const v2 = await fresh();
  const video = { readyState: 1, videoWidth: 0, videoHeight: 0, srcObject: null } as unknown as HTMLVideoElement;
  await assert.rejects(v2.captureFrameV2(video, { lane: 'track', mirror: false, base64: false }), v2.V2FrameUnavailableError);
  assert.equal(await v2.sampleAppearanceV2(video, BOX, false), null);
  assert.equal(env.workers.length, 0);
});

test('the tracking lane reports reading the stream only for the attached live track', async () => {
  const v2 = await fresh();
  const video = withStream(true);
  assert.equal(v2.trackLaneReadsStream(video), false, 'nothing attached before the first capture');
  await v2.captureFrameV2(video, { lane: 'track', mirror: false, base64: false });
  assert.equal(v2.trackLaneReadsStream(video), true);
  (video as unknown as { srcObject: FakeMediaStream }).srcObject = new FakeMediaStream([{ id: 'cam-9', readyState: 'live' }]);
  assert.equal(v2.trackLaneReadsStream(video), false, 'another camera: not until it is attached');
  const noStream = withStream(false);
  assert.equal(v2.trackLaneReadsStream(noStream), false);
});
