/**
 * Main-thread side of the off-thread capture worker (`jpeg-encoder.worker.ts`).
 *
 * The worker itself is a typed Vite module, so Vite emits it as a normal asset — no `blob:` CSP
 * allowance and no untyped source string to drift out of sync. One worker serves every job (`capture`,
 * `appearance`, `encode`); replies are matched to requests by a monotonic id, so jobs from the tracking
 * loop and the guide can be in flight together without ever reading each other's reply.
 *
 * Every failure path settles the request: a worker that fails to load, an uncaught error inside it, a
 * message that cannot be deserialized, or a reply that does not match the request all REJECT instead
 * of leaving the caller awaiting forever. Listeners are removed on settle, and a worker that has
 * failed is terminated so it is never reused in a broken state. Browsers without the required APIs are
 * reported by `isOffscreenJpegSupported()` so the caller can use the main-thread encoder — that is an
 * established path, not an error to swallow.
 */
import type { CapturePriority } from './captureJobQueue';
import type {
  AppearanceRequest,
  CaptureWorkerRequest,
  CaptureWorkerResponse,
  JpegBounds,
  WorkerStages,
} from './jpeg-encoder.worker';

export type { CapturePriority } from './captureJobQueue';
export type { WorkerStages } from './jpeg-encoder.worker';

export interface JpegEncoded {
  arrayBuffer: ArrayBuffer;
  width: number;
  height: number;
  stages?: WorkerStages;
}

export interface FrameCaptured {
  rawBase64: string;
  width: number;
  height: number;
  byteLength: number;
  lumaGrid: Float32Array | null;
  appearanceGrid: Float32Array | null;
  stages?: WorkerStages;
}

interface PendingRequest {
  /** The reply's payload when it has this job's shape, otherwise null (a malformed reply). */
  parse: (data: Record<string, unknown>) => unknown;
  resolve: (value: never) => void;
  reject: (reason: Error) => void;
}

let sharedWorker: Worker | null = null;
const pending = new Map<string, PendingRequest>();

/**
 * Whether this browser can encode off the main thread at all. Every capability is required: without
 * `OffscreenCanvas.convertToBlob` the worker cannot produce JPEG bytes, and without
 * `createImageBitmap` the frame cannot cross the thread boundary without a copy.
 */
export function isOffscreenJpegSupported(): boolean {
  return (
    typeof Worker !== 'undefined' &&
    typeof createImageBitmap === 'function' &&
    typeof OffscreenCanvas !== 'undefined' &&
    typeof OffscreenCanvas.prototype.convertToBlob === 'function'
  );
}

/** Reject everything still waiting, then drop the worker so the next call builds a fresh one. */
function teardownWithError(reason: Error): void {
  const worker = sharedWorker;
  sharedWorker = null;
  if (worker) {
    worker.removeEventListener('message', onMessage);
    worker.removeEventListener('error', onError);
    worker.removeEventListener('messageerror', onMessageError);
    worker.terminate();
  }
  const waiting = [...pending.values()];
  pending.clear();
  for (const request of waiting) request.reject(reason);
}

function onMessage(event: MessageEvent<CaptureWorkerResponse>): void {
  const data = event.data as unknown as Record<string, unknown> | null;
  // A reply for a request nobody is waiting on (a superseded id) is ignored, not a failure.
  if (!data || typeof data.id !== 'string') return;
  const request = pending.get(data.id);
  if (!request) return;
  pending.delete(data.id);

  if (typeof data.error === 'string') {
    // The worker reported a real failure. It is alive but could not do the job; retire it so a broken
    // context or encode state is never reused.
    const failure = new Error(data.error);
    request.reject(failure);
    teardownWithError(failure);
    return;
  }
  const value = request.parse(data);
  if (value !== null) {
    request.resolve(value as never);
    return;
  }
  // A reply carrying this request's id but none of the expected fields: reject rather than hang.
  const malformed = new Error('JPEG 인코더가 알 수 없는 응답을 보냈습니다.');
  request.reject(malformed);
  teardownWithError(malformed);
}

function onError(event: ErrorEvent): void {
  teardownWithError(new Error(`JPEG 인코더를 불러오지 못했습니다${event.message ? `: ${event.message}` : ''}.`));
}

function onMessageError(): void {
  teardownWithError(new Error('JPEG 인코더 메시지를 해석할 수 없습니다.'));
}

function ensureWorker(): Worker {
  if (!sharedWorker) {
    sharedWorker = new Worker(new URL('./jpeg-encoder.worker.ts', import.meta.url), { type: 'module' });
    sharedWorker.addEventListener('message', onMessage);
    sharedWorker.addEventListener('error', onError);
    sharedWorker.addEventListener('messageerror', onMessageError);
  }
  return sharedWorker;
}

/** Monotonic, so two jobs in flight can never read each other's reply. */
let encodeSeq = 0;

type Unsent<R> = R extends CaptureWorkerRequest ? Omit<R, 'id' | 'postedAbs'> : never;

/**
 * Send one job. The `ImageBitmap` is transferred, so the caller must not use it again; if the hand-off
 * itself fails the bitmap is closed here so it cannot leak.
 */
function send<T>(job: Unsent<CaptureWorkerRequest>, parse: (data: Record<string, unknown>) => T | null): Promise<T> {
  const bitmap = job.imageBitmap;
  let worker: Worker;
  try {
    worker = ensureWorker();
  } catch (error) {
    bitmap.close();
    return Promise.reject(error instanceof Error ? error : new Error(String(error)));
  }

  const id = `enc-${(encodeSeq += 1)}`;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { parse, resolve: resolve as (value: never) => void, reject });
    try {
      worker.postMessage({ ...job, id, postedAbs: performance.timeOrigin + performance.now() }, [bitmap]);
    } catch (error) {
      pending.delete(id);
      bitmap.close();
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}

const isGrid = (value: unknown): value is Float32Array | null => value === null || value instanceof Float32Array;

/** Stage times ride along when the worker sends them (measurement only; never required). */
const stagesOf = (data: Record<string, unknown>): { stages?: WorkerStages } =>
  data.stages && typeof data.stages === 'object' ? { stages: data.stages as WorkerStages } : {};

/** Encode one already-drawn, already-sized frame off the main thread (the canvas fallback path). */
export function encodeJpegOffThread(
  bitmap: ImageBitmap,
  options: JpegBounds & { priority?: CapturePriority }
): Promise<JpegEncoded> {
  const { priority = 'guide', ...bounds } = options;
  return send({ kind: 'encode', priority, imageBitmap: bitmap, ...bounds }, (data) =>
    data.arrayBuffer instanceof ArrayBuffer && typeof data.width === 'number' && typeof data.height === 'number'
      ? { arrayBuffer: data.arrayBuffer, width: data.width, height: data.height, ...stagesOf(data) }
      : null
  );
}

/**
 * Freeze, mirror, reduce and encode one RAW video frame (`createImageBitmap(video)`) entirely off the main
 * thread. See the worker's `capture` job.
 */
export function captureFrameOffThread(
  bitmap: ImageBitmap,
  options: JpegBounds & {
    priority: CapturePriority;
    mirror: boolean;
    targetWidth: number;
    targetHeight: number;
    appearance: AppearanceRequest | null;
  }
): Promise<FrameCaptured> {
  return send({ kind: 'capture', imageBitmap: bitmap, ...options }, (data) =>
    typeof data.rawBase64 === 'string' &&
    typeof data.width === 'number' &&
    typeof data.height === 'number' &&
    typeof data.byteLength === 'number' &&
    isGrid(data.lumaGrid) &&
    isGrid(data.appearanceGrid)
      ? {
          rawBase64: data.rawBase64,
          width: data.width,
          height: data.height,
          byteLength: data.byteLength,
          lumaGrid: data.lumaGrid,
          appearanceGrid: data.appearanceGrid,
          ...stagesOf(data),
        }
      : null
  );
}

/** The appearance grid of one RAW video frame inside a box, mirrored first when asked. */
export function sampleAppearanceOffThread(
  bitmap: ImageBitmap,
  options: { mirror: boolean; appearance: AppearanceRequest }
): Promise<{ grid: Float32Array | null; stages?: WorkerStages }> {
  return send({ kind: 'appearance', priority: 'guide', imageBitmap: bitmap, ...options }, (data) =>
    // Wrapped, because a null grid ("no sample") is a valid answer and null here means "malformed".
    'appearanceGrid' in data && isGrid(data.appearanceGrid) ? { grid: data.appearanceGrid, ...stagesOf(data) } : null
  );
}
