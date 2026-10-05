/**
 * Which capture path runs, and that every path keeps the same contract (`image-processor.ts`):
 *
 * - the worker path freezes with `createImageBitmap(video)` and never draws the video on the main thread;
 *   it asks the worker for the frame size, quality 0.88, the mirror, and (on request) the appearance box;
 * - `capturedAt` is the freeze time, read before the off-thread work, not when the reply arrives;
 * - an engine that refuses `createImageBitmap(video)` falls back to the canvas path, and a browser without
 *   the worker APIs to the synchronous path — each still returning a luma grid of the frame;
 * - an oversized reply is refused; a camera with no frame yet is the transient error.
 *
 * The DOM, the worker and the bitmaps are fakes: what is checked is the routing and the contract. Pixel
 * equivalence of the worker's output with the main-thread path needs a real browser (measured separately).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// The app's modules import each other without extensions (Vite resolves them). For this test only, let
// Node resolve a relative extensionless import to its `.ts` file, so the real capture modules load.
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

type Listener = (event: unknown) => void;

interface Posted {
  id: string;
  kind: string;
  priority: string;
  mirror?: boolean;
  targetWidth?: number;
  targetHeight?: number;
  quality?: number;
  appearance?: { box: unknown; config: { grid: number } } | null;
}

const env = {
  posted: [] as Posted[],
  transfers: [] as unknown[][],
  /** Main-thread draws whose source was the video element. */
  videoDraws: 0,
  toDataUrlCalls: 0,
  bitmapCalls: [] as Array<{ source: unknown; at: number }>,
  videoBitmapFails: false,
  replyDelayMs: 0,
  reply: (message: Posted): Record<string, unknown> => defaultReply(message),
  events: [] as Array<{ path: string; priority: string; capturedAt: number }>,
};

function defaultReply(message: Posted): Record<string, unknown> {
  if (message.kind === 'capture') {
    return {
      id: message.id,
      rawBase64: 'QUJD',
      width: message.targetWidth,
      height: message.targetHeight,
      byteLength: 3,
      lumaGrid: new Float32Array(48 * 36).fill(0.25),
      appearanceGrid: message.appearance ? new Float32Array(24 * 24).fill(0.75) : null,
    };
  }
  if (message.kind === 'appearance') return { id: message.id, appearanceGrid: new Float32Array(24 * 24).fill(0.5) };
  return { id: message.id, arrayBuffer: new Uint8Array([65, 66, 67]).buffer, width: 1280, height: 720 };
}

class FakeWorker {
  listeners = new Map<string, Set<Listener>>();
  addEventListener(type: string, listener: Listener): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(listener);
  }
  removeEventListener(type: string, listener: Listener): void {
    this.listeners.get(type)?.delete(listener);
  }
  postMessage(message: Posted, transfer: unknown[]): void {
    env.posted.push(message);
    env.transfers.push(transfer);
    setTimeout(() => {
      for (const listener of [...(this.listeners.get('message') ?? [])]) listener({ data: env.reply(message) });
    }, env.replyDelayMs);
  }
  terminate(): void {}
}

const video = { readyState: 4, videoWidth: 1280, videoHeight: 720, tagName: 'VIDEO' };

function fakeContext(canvas: FakeCanvas) {
  return {
    canvas,
    fillStyle: '',
    imageSmoothingEnabled: true,
    imageSmoothingQuality: 'low',
    setTransform() {},
    fillRect() {},
    drawImage(source: unknown) {
      if (source === video) env.videoDraws += 1;
    },
    getImageData(_x: number, _y: number, w: number, h: number) {
      return { data: new Uint8ClampedArray(w * h * 4).fill(128) };
    },
  };
}

class FakeCanvas {
  width = 300;
  height = 150;
  private context = fakeContext(this);
  getContext() {
    return this.context;
  }
  toDataURL() {
    env.toDataUrlCalls += 1;
    return 'data:image/jpeg;base64,QUJD';
  }
}

const globals = globalThis as unknown as Record<string, unknown>;
const saved = new Map<string, unknown>();
function setGlobal(key: string, value: unknown): void {
  if (!saved.has(key)) saved.set(key, globals[key]);
  globals[key] = value;
}

function OffscreenCanvasStub() {}
OffscreenCanvasStub.prototype.convertToBlob = () => {};

setGlobal('window', new EventTarget());
setGlobal('document', { createElement: () => new FakeCanvas() });
setGlobal('Worker', FakeWorker);
setGlobal('OffscreenCanvas', OffscreenCanvasStub);
setGlobal('createImageBitmap', async (source: unknown) => {
  env.bitmapCalls.push({ source, at: Date.now() });
  if (source === video && env.videoBitmapFails) throw new DOMException('not decodable', 'InvalidStateError');
  return { width: 1280, height: 720, close() {} };
});
(globals.window as EventTarget).addEventListener('synoptics:capture', (event) => {
  env.events.push((event as CustomEvent).detail);
});

test.after(() => {
  for (const [key, value] of saved) {
    if (value === undefined) delete globals[key];
    else globals[key] = value;
  }
});

function reset(): void {
  env.posted = [];
  env.transfers = [];
  env.videoDraws = 0;
  env.toDataUrlCalls = 0;
  env.bitmapCalls = [];
  env.videoBitmapFails = false;
  env.replyDelayMs = 0;
  env.reply = defaultReply;
  env.events = [];
  setGlobal('OffscreenCanvas', OffscreenCanvasStub);
}

// These cases describe the worker1 path (`?capture=worker1`); v2 has its own (`captureV2.test.ts`).
const mode = await import('./captureMode.ts');
mode.setCaptureSettingsForTest({ mode: 'worker1', trackScale: 'off', trackUpload: 'binary', v2Source: 'auto' });
const processor = await import('./image-processor.ts');
const asVideo = video as unknown as HTMLVideoElement;

test('the worker path freezes with createImageBitmap(video) and does nothing else with the frame on the main thread', async () => {
  reset();
  env.replyDelayMs = 30;
  const frame = await processor.captureVideoFrameAsync(asVideo, { mirror: true, priority: 'track' });
  assert.equal(env.videoDraws, 0, 'no drawImage(video) on the main thread');
  assert.equal(env.toDataUrlCalls, 0);
  assert.equal(env.bitmapCalls.length, 1);
  assert.equal(env.bitmapCalls[0].source, video, 'the bitmap is the raw video frame');

  const sent = env.posted[0];
  assert.equal(sent.kind, 'capture');
  assert.equal(sent.priority, 'track');
  assert.equal(sent.mirror, true);
  assert.equal(sent.targetWidth, 1280, 'same resolution: the frame is not shrunk');
  assert.equal(sent.targetHeight, 720);
  assert.equal(sent.quality, 0.88);
  assert.equal(sent.appearance, null);
  assert.equal(env.transfers[0].length, 1, 'the bitmap is transferred, not copied');

  assert.equal(frame.rawBase64, 'QUJD');
  assert.equal(frame.dataUrl, 'data:image/jpeg;base64,QUJD');
  assert.equal(frame.width, 1280);
  assert.equal(frame.byteLength, 3);
  assert.equal(frame.anchorGrid?.length, 48 * 36);
  assert.equal('appearanceGrid' in frame, false, 'no appearance grid unless asked for');
  // The freeze time, read beside createImageBitmap — not the reply's arrival 30 ms later.
  assert.ok(frame.capturedAt <= env.bitmapCalls[0].at);
  assert.ok(Date.now() - frame.capturedAt >= 25);

  assert.equal(env.events.length, 1);
  assert.equal(env.events[0].path, 'worker');
  assert.equal(env.events[0].priority, 'track');
  assert.equal(env.events[0].capturedAt, frame.capturedAt);
});

test('a follow capture gets the appearance grid of the same frozen frame from the worker', async () => {
  reset();
  const box = { x: 0.2, y: 0.3, width: 0.2, height: 0.2 };
  const frame = await processor.captureVideoFrameAsync(asVideo, { mirror: false, appearanceBox: box });
  assert.equal(env.posted.length, 1, 'one job: the JPEG and the appearance grid come from one snapshot');
  assert.deepEqual(env.posted[0].appearance?.box, box);
  assert.equal(env.posted[0].appearance?.config.grid, 24);
  assert.equal(env.posted[0].priority, 'guide', 'guide captures queue behind tracking frames');
  assert.equal(frame.appearanceGrid?.length, 24 * 24);
});

test('an engine that refuses createImageBitmap(video) falls back to the canvas path', async () => {
  reset();
  env.videoBitmapFails = true;
  const frame = await processor.captureVideoFrameAsync(asVideo, { mirror: true, priority: 'track' });
  assert.equal(env.videoDraws, 1, 'one main-thread read of the video');
  assert.equal(env.posted.length, 1);
  assert.equal(env.posted[0].kind, 'encode');
  assert.equal(env.posted[0].priority, 'track');
  assert.equal(frame.rawBase64, 'QUJD');
  assert.equal(frame.anchorGrid?.length, 48 * 36);
  assert.equal(env.events[0].path, 'canvas');
});

test('a browser without the worker APIs uses the synchronous main-thread path', async () => {
  reset();
  setGlobal('OffscreenCanvas', undefined);
  const box = { x: 0.2, y: 0.3, width: 0.2, height: 0.2 };
  const frame = await processor.captureVideoFrameAsync(asVideo, { mirror: false, appearanceBox: box });
  assert.equal(env.posted.length, 0, 'no worker');
  assert.equal(env.videoDraws, 1);
  assert.equal(env.toDataUrlCalls, 1);
  assert.equal(frame.rawBase64, 'QUJD');
  assert.equal(frame.anchorGrid?.length, 48 * 36);
  assert.equal(frame.appearanceGrid?.length, 24 * 24, 'the appearance grid still comes with the frame');
  assert.equal(env.events[0].path, 'main');
});

test('an oversized reply is refused, and a camera without a frame yet is the transient error', async () => {
  reset();
  env.reply = (message) => ({ ...defaultReply(message), byteLength: 1_500_001 });
  await assert.rejects(processor.captureVideoFrameAsync(asVideo), /크기 제한/);

  reset();
  const notReady = { readyState: 1, videoWidth: 0, videoHeight: 0 } as unknown as HTMLVideoElement;
  await assert.rejects(
    processor.captureVideoFrameAsync(notReady),
    (error: unknown) => error instanceof processor.CameraFrameUnavailableError
  );
  assert.equal(env.bitmapCalls.length, 0);
});

test('the appearance sample runs in the worker from a raw video bitmap', async () => {
  reset();
  const box = { x: 0.1, y: 0.1, width: 0.3, height: 0.3 };
  const grid = await processor.sampleVideoAppearanceAsync(asVideo, box, true);
  assert.equal(env.videoDraws, 0);
  assert.equal(env.posted[0].kind, 'appearance');
  assert.equal(env.posted[0].mirror, true);
  assert.deepEqual(env.posted[0].appearance?.box, box);
  assert.equal(grid?.length, 24 * 24);
});
