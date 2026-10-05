/**
 * The off-thread encoder's failure paths.
 *
 * The real hazard is a HANG: the tracking loop awaits this promise, so a worker that fails to load, an
 * uncaught error inside it, a message that cannot be deserialized, or a reply that carries the request
 * id but none of the expected fields must all REJECT. A promise that never settles would stall the run
 * forever with no error shown to the user, which no test of the happy path would catch.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

type Listener = (event: unknown) => void;

class FakeWorker {
  listeners = new Map<string, Set<Listener>>();
  posted: unknown[] = [];
  terminated = false;
  postError: Error | null = null;

  addEventListener(type: string, listener: Listener): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(listener);
  }
  removeEventListener(type: string, listener: Listener): void {
    this.listeners.get(type)?.delete(listener);
  }
  postMessage(message: unknown): void {
    if (this.postError) throw this.postError;
    this.posted.push(message);
  }
  terminate(): void {
    this.terminated = true;
  }
  emit(type: string, event: unknown): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener(event);
  }
}

const OPTIONS = { quality: 0.88, maxBytes: 100, maxBase64Length: 200 };

/** A frame stand-in that records whether it was released. */
function makeBitmap(): { bitmap: ImageBitmap; closed: () => boolean } {
  let closed = false;
  const bitmap = {
    width: 8,
    height: 8,
    close: () => {
      closed = true;
    },
  } as unknown as ImageBitmap;
  return { bitmap, closed: () => closed };
}

/**
 * Install the globals `worker-jpeg.ts` feature-detects and use, then return the created workers plus a
 * restore function. The replaced globals are put back afterwards so this file does not leave process-wide
 * state behind for whatever runs next.
 */
function installEnvironment(): { workers: FakeWorker[]; restore: () => void } {
  const workers: FakeWorker[] = [];
  const globals = globalThis as unknown as Record<string, unknown>;
  const saved: Array<[string, unknown]> = [];
  const set = (key: string, value: unknown) => {
    saved.push([key, globals[key]]);
    globals[key] = value;
  };
  set('OffscreenCanvas', function OffscreenCanvas() {});
  (globals.OffscreenCanvas as { prototype: Record<string, unknown> }).prototype.convertToBlob = () => {};
  set('createImageBitmap', async () => ({ width: 8, height: 8, close: () => {} }));
  set('Worker', function Worker(this: unknown) {
    const worker = new FakeWorker();
    workers.push(worker);
    return worker;
  });
  return {
    workers,
    restore: () => {
      for (const [key, value] of saved) {
        if (value === undefined) delete globals[key];
        else globals[key] = value;
      }
    },
  };
}

// Deterministic cache-busting: the shared worker and pending map are module-scoped, so each test needs a
// fresh module. A counter keeps the specifier stable across runs, unlike a random one.
let moduleCase = 0;
async function loadModule() {
  moduleCase += 1;
  return import(`./worker-jpeg.ts?case=${moduleCase}`);
}

test('a worker error rejects instead of hanging, and the worker is retired', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { encodeJpegOffThread } = await loadModule();
    const { bitmap } = makeBitmap();
    const pending = encodeJpegOffThread(bitmap, OPTIONS);
    workers[0].emit('error', { message: 'boom' });
    await assert.rejects(pending, /boom/);
    assert.equal(workers[0].terminated, true, 'a failed worker is terminated, not reused');
  } finally {
    restore();
  }
});

test('an undeserializable message rejects instead of hanging', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { encodeJpegOffThread } = await loadModule();
    const { bitmap } = makeBitmap();
    const pending = encodeJpegOffThread(bitmap, OPTIONS);
    workers[0].emit('messageerror', {});
    await assert.rejects(pending, (error: unknown) => error instanceof Error);
    assert.equal(workers[0].terminated, true);
  } finally {
    restore();
  }
});

test('a reply carrying our id but no usable fields rejects instead of hanging', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { encodeJpegOffThread } = await loadModule();
    const { bitmap } = makeBitmap();
    const pending = encodeJpegOffThread(bitmap, OPTIONS);
    const sent = workers[0].posted[0] as { id: string };
    workers[0].emit('message', { data: { id: sent.id } });
    await assert.rejects(pending, (error: unknown) => error instanceof Error);
    assert.equal(workers[0].terminated, true);
  } finally {
    restore();
  }
});

test('a worker-reported encode error rejects and the worker is retired', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { encodeJpegOffThread } = await loadModule();
    const { bitmap } = makeBitmap();
    const pending = encodeJpegOffThread(bitmap, OPTIONS);
    const sent = workers[0].posted[0] as { id: string };
    workers[0].emit('message', { data: { id: sent.id, error: 'encode failed' } });
    await assert.rejects(pending, /encode failed/);
    assert.equal(workers[0].terminated, true);
  } finally {
    restore();
  }
});

test('a reply for an unknown id is ignored, and a later real reply still settles', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { encodeJpegOffThread } = await loadModule();
    const { bitmap } = makeBitmap();
    const pending = encodeJpegOffThread(bitmap, OPTIONS);
    const sent = workers[0].posted[0] as { id: string };
    workers[0].emit('message', { data: { id: 'someone-else', arrayBuffer: new ArrayBuffer(1), width: 1, height: 1 } });
    workers[0].emit('message', { data: { id: sent.id, arrayBuffer: new ArrayBuffer(2), width: 8, height: 8 } });
    const encoded = await pending;
    assert.equal(encoded.arrayBuffer.byteLength, 2);
  } finally {
    restore();
  }
});

test('a postMessage failure rejects and releases the frame it was handed', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { encodeJpegOffThread } = await loadModule();
    // The worker is created lazily by the first call, so bring it into existence before failing a send.
    const warmup = makeBitmap();
    const first = encodeJpegOffThread(warmup.bitmap, OPTIONS);
    const firstSent = workers[0].posted[0] as { id: string };
    workers[0].emit('message', { data: { id: firstSent.id, arrayBuffer: new ArrayBuffer(2), width: 8, height: 8 } });
    await first;

    workers[0].postError = new Error('DataCloneError');
    const { bitmap, closed } = makeBitmap();
    const pending = encodeJpegOffThread(bitmap, OPTIONS);
    await assert.rejects(pending, /DataCloneError/);
    assert.equal(closed(), true, 'the frame is released when the hand-off to the worker fails');
  } finally {
    restore();
  }
});

test('jobs in flight together are matched by id: replies out of order each reach their own caller', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { captureFrameOffThread, encodeJpegOffThread } = await loadModule();
    const capture = captureFrameOffThread(makeBitmap().bitmap, {
      ...OPTIONS,
      priority: 'track',
      mirror: true,
      targetWidth: 8,
      targetHeight: 8,
      appearance: null,
    });
    const encode = encodeJpegOffThread(makeBitmap().bitmap, OPTIONS);
    const [captureSent, encodeSent] = workers[0].posted as Array<{ id: string; kind: string; priority: string }>;
    assert.equal(captureSent.kind, 'capture');
    assert.equal(captureSent.priority, 'track');
    assert.equal(encodeSent.kind, 'encode');
    assert.equal(encodeSent.priority, 'guide', 'an encode is a guide job unless asked otherwise');
    workers[0].emit('message', { data: { id: encodeSent.id, arrayBuffer: new ArrayBuffer(3), width: 8, height: 8 } });
    const lumaGrid = new Float32Array(4);
    workers[0].emit('message', {
      data: { id: captureSent.id, rawBase64: 'QUJD', width: 8, height: 8, byteLength: 3, lumaGrid, appearanceGrid: null },
    });
    assert.equal((await encode).arrayBuffer.byteLength, 3);
    const captured = await capture;
    assert.equal(captured.rawBase64, 'QUJD');
    assert.equal(captured.lumaGrid, lumaGrid);
    assert.equal(captured.appearanceGrid, null);
  } finally {
    restore();
  }
});

test('a capture reply missing its fields rejects instead of hanging; an appearance reply may carry a null grid', async () => {
  const { workers, restore } = installEnvironment();
  try {
    const { captureFrameOffThread, sampleAppearanceOffThread } = await loadModule();
    const appearance = sampleAppearanceOffThread(makeBitmap().bitmap, {
      mirror: false,
      appearance: { box: { x: 0, y: 0, width: 1, height: 1 }, config: { grid: 24, pad: 0.15, sampleMs: 250, threshold: 0.06, sustain: 2, patchCells: 6 } },
    });
    const appearanceSent = workers[0].posted[0] as { id: string; kind: string };
    assert.equal(appearanceSent.kind, 'appearance');
    workers[0].emit('message', { data: { id: appearanceSent.id, appearanceGrid: null } });
    assert.equal((await appearance).grid, null, 'no sample is an answer, not a malformed reply');

    const capture = captureFrameOffThread(makeBitmap().bitmap, {
      ...OPTIONS,
      priority: 'guide',
      mirror: false,
      targetWidth: 8,
      targetHeight: 8,
      appearance: null,
    });
    const captureSent = workers[0].posted[1] as { id: string };
    workers[0].emit('message', { data: { id: captureSent.id, rawBase64: 'QUJD', width: 8 } });
    await assert.rejects(capture, (error: unknown) => error instanceof Error);
    assert.equal(workers[0].terminated, true);
  } finally {
    restore();
  }
});
