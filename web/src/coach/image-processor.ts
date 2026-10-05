/**
 * Client-side frame preparation: every live camera capture (tracking, guide plan/follow/confirm) goes
 * through one encoder and one set of transport bounds.
 *
 * Requirements:
 * - JPEG format only.
 * - Decoded raw JPEG bytes strictly <= 1,500,000 bytes (1.5MB server cap).
 *   Base64 character length strictly <= 2,000,000 characters.
 * - Decoded pixel resolution: min 320x240, max 2048x2048, total pixels <= 2,073,600 (~2.1MP).
 * - Never stretch or distort aspect ratio. Scale uniformly.
 * - Extreme aspect ratios (> 4:1 or < 1:4) or tiny source (< 320x240) rejected with clear actionable message.
 */

export interface ProcessedImage {
  rawBase64: string;
  dataUrl: string;
  width: number;
  height: number;
  byteLength: number;
}

export interface CapturedImage extends ProcessedImage {
  /**
   * Luma grid of the SAME frozen snapshot the JPEG was encoded from, taken through the same canvas path
   * every later live sample uses (and with the same mirror setting). Nothing reads it since the
   * scene-change lane was removed (2026-10-01): the legacy and worker1 paths still compute it (they are kept
   * as they were, for A/B measurement), v2 does not and returns null. Null also when the video could not be
   * read.
   */
  anchorGrid: Float32Array | null;
  /**
   * When the frame was FROZEN — read beside the call that freezes it (`createImageBitmap(video)` or
   * `drawVideoFrame`), not after encoding. Encoding can
   * be asynchronous (off the main thread), so a timestamp taken afterwards would describe the end of the
   * encode rather than the pixels: the age a caller derives from it would understate how stale the image
   * is, and a held or slow encode would present an old frame as fresh.
   */
  capturedAt: number;
  /**
   * The appearance grid inside `CaptureOptions.appearanceBox`, from the SAME frozen snapshot as the JPEG
   * (the guide follow's `target_changed` reference). Present only when a box was asked for; null when the
   * crop could not be sampled.
   */
  appearanceGrid?: Float32Array | null;
}

import type { Box } from '@visual-coach/visual-tools';

import { drawVideoFrame, sampleLumaGrid } from '../camera/luma';
import {
  JPEG_START_QUALITY,
  MAX_BASE64_LENGTH,
  MAX_DECODED_BYTES,
  computeTransportSize,
  type SceneSubject,
} from './transportSize';
import { DEFAULT_APPEARANCE_CONFIG } from '../intent/anchorAppearance';
import { sampleAnchorGrid, sampleVideoAnchorGrid } from '../intent/anchorSampler';
import { captureSettings, type CaptureMode } from './captureMode';
import {
  V2FrameUnavailableError,
  captureFrameV2,
  isV2Supported,
  sampleAppearanceV2,
  trackLaneReadsStream,
  type V2Captured,
} from './captureV2';
import {
  captureFrameOffThread,
  encodeJpegOffThread,
  isOffscreenJpegSupported,
  sampleAppearanceOffThread,
  type CapturePriority,
  type WorkerStages,
} from './worker-jpeg';

export { CaptureSkippedError } from './captureV2';

export interface CaptureOptions {
  /** Flip the frame horizontally while freezing it, so the upload looks like the mirrored preview. */
  mirror?: boolean;
  /** The capture worker's queue priority: the tracking loop's frames are never queued behind guide frames. */
  priority?: CapturePriority;
  /** Also reduce the frozen frame inside this box to the appearance grid (`CapturedImage.appearanceGrid`). */
  appearanceBox?: Box | null;
}

/**
 * The window event every successful camera capture dispatches (measurement hook: the in-page latency panel
 * listens for it now that the tracking capture no longer calls `drawImage(video)` on the main thread).
 * Dispatching with no listener costs nothing measurable, and nothing accumulates (unlike a performance
 * timeline entry).
 */
export const CAPTURE_EVENT = 'synoptics:capture';

/**
 * Per-stage times of one capture, in ms (absent = the stage does not exist on that path). Main thread:
 * `drawMs`/`lumaMs`/`appearanceMs` on the legacy path, `bitmapMs` (`createImageBitmap`), `b64Ms` when the
 * main thread base64s. Worker: `postMs` (post → received), `queueMs` (waiting behind this worker's other
 * jobs), `waitMs` (v2 stream: waiting for a newer camera frame), `drawMs`, `lumaMs`, `appearanceMs`,
 * `encodeMs`, `b64Ms`, `replyMs` (reply posted → received on the main thread). `workMs` is the CPU work:
 * draw + luma + appearance + encode + base64.
 */
export interface CaptureStages {
  bitmapMs?: number;
  postMs?: number;
  queueMs?: number;
  waitMs?: number;
  drawMs?: number;
  lumaMs?: number;
  appearanceMs?: number;
  encodeMs?: number;
  b64Ms?: number;
  replyMs?: number;
  workMs?: number;
}

export interface CaptureEventDetail {
  /**
   * `main`: everything on the main thread (no worker APIs); `canvas`: main-thread draw, worker encode (the
   * legacy path, or worker1's fallback); `worker`: worker1; `v2-stream`/`v2-bitmap`: v2 with the worker
   * reading the camera itself / with a main-thread `createImageBitmap`.
   */
  path: 'worker' | 'canvas' | 'main' | 'v2-stream' | 'v2-bitmap';
  /** The page's `?capture=` mode. */
  mode: CaptureMode;
  /** `frame`: an encoded capture; `appearance`: a guide appearance sample (no JPEG, `byteLength` 0). */
  kind: 'frame' | 'appearance';
  /** Which worker did the work: worker1's one shared worker, a v2 lane's, or none. */
  worker: 'shared' | 'track' | 'guide' | null;
  priority: CapturePriority;
  /** `performance.now()` when the frame was frozen (the same instant as `capturedAt`). */
  frozenAt: number;
  /** `performance.now()` when the encoded frame was ready on the main thread. */
  readyAt: number;
  /** Wall-clock freeze time, the value `CapturedImage.capturedAt` carries. */
  capturedAt: number;
  byteLength: number;
  width: number;
  height: number;
  stages: CaptureStages;
}

function reportCapture(detail: CaptureEventDetail): void {
  if (typeof window === 'undefined' || typeof CustomEvent !== 'function') return;
  window.dispatchEvent(new CustomEvent<CaptureEventDetail>(CAPTURE_EVENT, { detail }));
}

/**
 * The camera has not produced a readable frame yet. This is **transient** — the caller should wait for
 * the next presented frame and try again — so it is its own type rather than a message to pattern-match.
 * Every other failure from a capture (bounds rejection, a canvas that cannot be created, a worker error,
 * output still over the transport bound) is a REAL error and must surface instead of being retried.
 */
export class CameraFrameUnavailableError extends Error {}


/**
 * Capture the video frame displayed right now, under the server's transport bounds.
 *
 * Reads the element's own intrinsic size, so the encoded frame matches the geometry the guidance
 * overlay is drawn on. With `mirror` set, the pixels are flipped horizontally while they are drawn:
 * the preview flips the `<video>` element with CSS, but `drawImage` reads raw pixels and ignores CSS
 * transforms, so this is the only place that can make the uploaded JPEG look exactly like the preview.
 * The returned anchor grid describes the very pixels that were encoded — never a second, later read of
 * the live video, which would describe a different moment of the scene. Throws when no decodable frame
 * exists yet instead of uploading a black canvas.
 */
export function captureVideoFrame(video: HTMLVideoElement, options: CaptureOptions = {}): CapturedImage {
  assertFrameReady(video);
  // ONE read of the live element. The JPEG and the anchor grid both come from this same frozen
  // snapshot, so the frame that is uploaded is exactly the frame that was measured — a second read of
  // the video would pair one frame's pixels with another frame's grid, and reading the video into the
  // small grid instead of through a canvas would carry a resampling difference that reads as a
  // permanent scene change (see camera/luma.ts). The snapshot is mirrored at draw time, which is also
  // the only place the orientation is applied.
  // Read the freeze time BESIDE the draw, not after the encode: encoding is the slow part, and a
  // timestamp taken at the end would describe the encode rather than the pixels.
  const capturedAt = Date.now();
  const frozenAt = performance.now();
  const snapshot = drawVideoFrame(video, { mirror: Boolean(options.mirror) });
  if (!snapshot) throw new CameraFrameUnavailableError('카메라 화면을 읽지 못했습니다. 잠시 후 다시 시도해주세요.');
  const drawn = performance.now();
  const anchorGrid = sampleLumaGrid(snapshot);
  const lumaDone = performance.now();
  const appearance = appearanceOfSnapshot(snapshot, options);
  const appearanceDone = performance.now();
  const encoded = encodeScaledFrame(snapshot, snapshot.width, snapshot.height, '카메라 화면');
  const readyAt = performance.now();
  reportCapture({
    path: 'main',
    mode: captureSettings().mode,
    kind: 'frame',
    worker: null,
    priority: options.priority ?? 'guide',
    frozenAt,
    readyAt,
    capturedAt,
    byteLength: encoded.byteLength,
    width: encoded.width,
    height: encoded.height,
    stages: {
      drawMs: drawn - frozenAt,
      lumaMs: lumaDone - drawn,
      appearanceMs: appearanceDone - lumaDone,
      encodeMs: readyAt - appearanceDone,
      workMs: readyAt - frozenAt,
    },
  });
  return { ...encoded, anchorGrid, capturedAt, ...appearance };
}

/** HAVE_CURRENT_DATA, spelled out: Safari has not always exposed the engine constant off the interface. */
const HAVE_CURRENT_DATA = 2;

function assertFrameReady(video: HTMLVideoElement): void {
  if (video.readyState < HAVE_CURRENT_DATA || video.videoWidth === 0 || video.videoHeight === 0) {
    throw new CameraFrameUnavailableError('카메라 화면이 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.');
  }
}

/** The requested appearance grid of a main-thread snapshot, in the shape `CapturedImage` carries it. */
function appearanceOfSnapshot(
  snapshot: HTMLCanvasElement,
  options: CaptureOptions
): Pick<CapturedImage, 'appearanceGrid'> {
  if (!options.appearanceBox) return {};
  return {
    appearanceGrid: sampleAnchorGrid(
      snapshot,
      { width: snapshot.width, height: snapshot.height },
      options.appearanceBox,
      DEFAULT_APPEARANCE_CONFIG
    ),
  };
}

/** Worker stage times (on the worker's clock) as `CaptureStages`, with the hop back measured here. */
function workerStages(stages: WorkerStages | undefined, bitmapMs?: number): CaptureStages {
  if (!stages) return bitmapMs === undefined ? {} : { bitmapMs };
  const { doneAbs, ...rest } = stages;
  return { ...rest, ...(bitmapMs === undefined ? {} : { bitmapMs }), replyMs: performance.timeOrigin + performance.now() - doneAbs };
}

/**
 * Capture the video frame displayed right now **off the main thread**, by the page's `?capture=` path
 * (`captureMode.ts`): v2 (default, `captureV2.ts`), worker1 or legacy. Every path keeps the same contract:
 * the anchor/appearance grids describe the very pixels encoded, `capturedAt` is the freeze time, and the
 * transport bounds are shared.
 *
 * worker1: the main thread only calls `createImageBitmap(video)`, which is asynchronous: it snapshots the
 * frame at the current playback position when it is CALLED (that is the frame the result describes,
 * whenever the promise settles) without the synchronous GPU→CPU read a `drawImage(video)` into a readable
 * canvas costs — measured p50 17 ms / max 91 ms on the main thread of a Windows Edge laptop, per capture.
 * The raw bitmap is transferred to the shared worker, which applies exactly what `captureVideoFrame` does:
 * the mirror once while freezing a full-size snapshot, the luma grid through the same ladder
 * (`reduceToLumaGrid`), the same transport size (`computeTransportSize`, computed here), the same starting
 * quality and the same quality search.
 *
 * Fallbacks, both established working paths rather than failures to hide: a browser without
 * `Worker`/`OffscreenCanvas`/`createImageBitmap` uses the synchronous main-thread encoder, and an engine
 * that refuses to snapshot this video element into a bitmap uses `captureVideoFrameViaCanvas` (draw on the
 * main thread, encode in the worker — the legacy path). A failure *inside* a worker (a real encode error,
 * or output that still exceeds the transport bound) is propagated as an error.
 */
export async function captureVideoFrameAsync(
  video: HTMLVideoElement,
  options: CaptureOptions = {}
): Promise<CapturedImage> {
  if (!isOffscreenJpegSupported()) return captureVideoFrame(video, options);
  const { mode } = captureSettings();
  if (mode === 'legacy') return captureVideoFrameViaCanvas(video, options);
  if (mode === 'v2' && isV2Supported()) {
    const captured = await captureV2Frame(video, options, true);
    const rawBase64 = captured.rawBase64 ?? '';
    return {
      rawBase64,
      dataUrl: `data:image/jpeg;base64,${rawBase64}`,
      width: captured.width,
      height: captured.height,
      byteLength: captured.byteLength,
      anchorGrid: null,
      capturedAt: captured.capturedAt,
      ...(options.appearanceBox ? { appearanceGrid: captured.appearanceGrid } : {}),
    };
  }
  assertFrameReady(video);
  const priority = options.priority ?? 'guide';
  // The freeze time is read here, beside the call that snapshots the frame — not after the off-thread
  // work. Stamping after it would age the image as young as the encode was slow, so a held worker reply
  // would render an old frame as fresh.
  const capturedAt = Date.now();
  const frozenAt = performance.now();
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(video);
  } catch {
    return captureVideoFrameViaCanvas(video, options);
  }
  const bitmapMs = performance.now() - frozenAt;
  let size: { targetWidth: number; targetHeight: number };
  try {
    size = computeTransportSize(bitmap.width, bitmap.height, '카메라 화면');
  } catch (error) {
    bitmap.close();
    throw error;
  }
  const captured = await captureFrameOffThread(bitmap, {
    priority,
    mirror: Boolean(options.mirror),
    ...size,
    appearance: options.appearanceBox ? { box: options.appearanceBox, config: DEFAULT_APPEARANCE_CONFIG } : null,
    quality: JPEG_START_QUALITY,
    maxBytes: MAX_DECODED_BYTES,
    maxBase64Length: MAX_BASE64_LENGTH,
  });
  // Re-check the bound here too, so a worker that ignored its arguments could not smuggle an
  // oversized frame onto the wire.
  if (captured.byteLength > MAX_DECODED_BYTES || captured.rawBase64.length > MAX_BASE64_LENGTH) {
    throw new Error('카메라 화면 압축 후에도 크기 제한(1.5MB)을 초과했습니다. 다른 입력을 사용해주세요.');
  }
  reportCapture({
    path: 'worker',
    mode,
    kind: 'frame',
    worker: 'shared',
    priority,
    frozenAt,
    readyAt: performance.now(),
    capturedAt,
    byteLength: captured.byteLength,
    width: captured.width,
    height: captured.height,
    stages: workerStages(captured.stages, bitmapMs),
  });
  return {
    rawBase64: captured.rawBase64,
    dataUrl: `data:image/jpeg;base64,${captured.rawBase64}`,
    width: captured.width,
    height: captured.height,
    byteLength: captured.byteLength,
    anchorGrid: captured.lumaGrid,
    capturedAt,
    ...(options.appearanceBox ? { appearanceGrid: captured.appearanceGrid } : {}),
  };
}

/** One v2 capture, reported; a camera with no frame yet surfaces as the usual transient error. */
async function captureV2Frame(video: HTMLVideoElement, options: CaptureOptions, base64: boolean): Promise<V2Captured> {
  const priority = options.priority ?? 'guide';
  let captured: V2Captured;
  try {
    captured = await captureFrameV2(video, {
      lane: priority === 'track' ? 'track' : 'guide',
      mirror: Boolean(options.mirror),
      base64,
      appearanceBox: options.appearanceBox ?? null,
    });
  } catch (error) {
    if (error instanceof V2FrameUnavailableError) throw new CameraFrameUnavailableError(error.message);
    throw error;
  }
  const { doneAbs: _doneAbs, ...stages } = captured.stages;
  reportCapture({
    path: captured.source === 'stream' ? 'v2-stream' : 'v2-bitmap',
    mode: 'v2',
    kind: 'frame',
    worker: priority === 'track' ? 'track' : 'guide',
    priority,
    frozenAt: captured.frozenAt,
    readyAt: performance.now(),
    capturedAt: captured.capturedAt,
    byteLength: captured.byteLength,
    width: captured.width,
    height: captured.height,
    stages: { ...stages, replyMs: captured.replyMs, ...(captured.source === 'bitmap' ? { bitmapMs: captured.bitmapMs } : {}) },
  });
  return captured;
}

/**
 * Tracking frames come straight from the camera track in the v2 tracking worker, which itself never answers
 * from the frame it answered from last: the tracking loop then need not wait for the <video> to present a
 * newer frame first (the worker has it earlier).
 */
export function trackFramesFromStream(video: HTMLVideoElement): boolean {
  return captureSettings().mode === 'v2' && isV2Supported() && trackLaneReadsStream(video);
}

/** A tracking frame as the tracking loop sends it: raw JPEG bytes, or base64 for the JSON form. */
export interface TrackFrameCapture {
  /** The JPEG, when the frame is to be uploaded as raw bytes (v2); null otherwise. */
  jpeg: Blob | null;
  /** The JPEG as base64, when the frame is to be uploaded as JSON; null otherwise. */
  rawBase64: string | null;
  width: number;
  height: number;
  byteLength: number;
  capturedAt: number;
}

/**
 * Capture a tracking frame. `binary` asks for the raw-JPEG form, which only v2 produces (no base64 at all);
 * every other case is `captureVideoFrameAsync` with the tracking priority.
 */
export async function captureTrackFrameAsync(
  video: HTMLVideoElement,
  options: { mirror: boolean; binary: boolean }
): Promise<TrackFrameCapture> {
  if (options.binary && isOffscreenJpegSupported() && captureSettings().mode === 'v2' && isV2Supported()) {
    const captured = await captureV2Frame(video, { mirror: options.mirror, priority: 'track' }, false);
    return {
      jpeg: captured.blob,
      rawBase64: null,
      width: captured.width,
      height: captured.height,
      byteLength: captured.byteLength,
      capturedAt: captured.capturedAt,
    };
  }
  const frame = await captureVideoFrameAsync(video, { mirror: options.mirror, priority: 'track' });
  return {
    jpeg: null,
    rawBase64: frame.rawBase64,
    width: frame.width,
    height: frame.height,
    byteLength: frame.byteLength,
    capturedAt: frame.capturedAt,
  };
}

/**
 * The legacy off-thread path (before fa9ec32), and worker1's fallback for an engine that cannot snapshot the
 * video element into a bitmap: one read of the live element ON THE MAIN THREAD (`drawVideoFrame`), the
 * anchor grid from that same snapshot, then the encode in the shared worker.
 */
async function captureVideoFrameViaCanvas(video: HTMLVideoElement, options: CaptureOptions): Promise<CapturedImage> {
  assertFrameReady(video);
  const priority = options.priority ?? 'guide';
  const capturedAt = Date.now();
  const frozenAt = performance.now();
  const snapshot = drawVideoFrame(video, { mirror: Boolean(options.mirror) });
  if (!snapshot) throw new CameraFrameUnavailableError('카메라 화면을 읽지 못했습니다. 잠시 후 다시 시도해주세요.');
  const drawn = performance.now();
  const anchorGrid = sampleLumaGrid(snapshot);
  const lumaDone = performance.now();
  const appearance = appearanceOfSnapshot(snapshot, options);
  const appearanceDone = performance.now();

  const { targetWidth, targetHeight } = computeTransportSize(snapshot.width, snapshot.height, '카메라 화면');

  let scaled: HTMLCanvasElement = snapshot;
  if (targetWidth !== snapshot.width || targetHeight !== snapshot.height) {
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D 컨텍스트를 생성할 수 없습니다.');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, targetWidth, targetHeight);
    ctx.drawImage(snapshot, 0, 0, targetWidth, targetHeight);
    scaled = canvas;
  }

  const bitmapStart = performance.now();
  const bitmap = await createImageBitmap(scaled);
  const bitmapMs = performance.now() - bitmapStart;
  const encoded = await encodeJpegOffThread(bitmap, {
    priority,
    quality: JPEG_START_QUALITY,
    maxBytes: MAX_DECODED_BYTES,
    maxBase64Length: MAX_BASE64_LENGTH,
  });
  const b64Start = performance.now();
  const bytes = new Uint8Array(encoded.arrayBuffer);
  // Base64 in chunks: a byte-at-a-time concat would allocate a new string per byte on this thread.
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  const rawBase64 = btoa(binary);
  const readyAt = performance.now();
  // Re-check the bound here too, so a worker that ignored its arguments could not smuggle an
  // oversized frame onto the wire.
  if (bytes.byteLength > MAX_DECODED_BYTES || rawBase64.length > MAX_BASE64_LENGTH) {
    throw new Error('카메라 화면 압축 후에도 크기 제한(1.5MB)을 초과했습니다. 다른 입력을 사용해주세요.');
  }
  const fromWorker = workerStages(encoded.stages);
  reportCapture({
    path: 'canvas',
    mode: captureSettings().mode,
    kind: 'frame',
    worker: 'shared',
    priority,
    frozenAt,
    readyAt,
    capturedAt,
    byteLength: bytes.byteLength,
    width: encoded.width,
    height: encoded.height,
    stages: {
      ...fromWorker,
      // The main thread's draw, luma and appearance; the worker's own draw + encode are in encodeMs.
      drawMs: drawn - frozenAt,
      lumaMs: lumaDone - drawn,
      appearanceMs: appearanceDone - lumaDone,
      bitmapMs,
      encodeMs: (fromWorker.drawMs ?? 0) + (fromWorker.encodeMs ?? 0),
      b64Ms: readyAt - b64Start,
      workMs: appearanceDone - frozenAt + (fromWorker.workMs ?? 0) + (readyAt - b64Start),
    },
  });
  return {
    rawBase64,
    dataUrl: `data:image/jpeg;base64,${rawBase64}`,
    width: encoded.width,
    height: encoded.height,
    byteLength: bytes.byteLength,
    anchorGrid,
    capturedAt,
    ...appearance,
  };
}

/**
 * The appearance grid of the live video inside `box` (the guide's 250 ms `target_changed` sample), by the
 * page's `?capture=` path: v2 reduces only the crop in the guide worker (and throws `CaptureSkippedError`
 * while the tracking lane is behind — the caller tries again later); worker1 freezes the whole frame in the
 * shared worker; legacy draws synchronously on the main thread (`sampleVideoAnchorGrid`, as before
 * fa9ec32). Null when no sample could be taken.
 */
export async function sampleVideoAppearanceAsync(
  video: HTMLVideoElement,
  box: Box,
  mirror: boolean
): Promise<Float32Array | null> {
  if (video.readyState < HAVE_CURRENT_DATA || video.videoWidth === 0 || video.videoHeight === 0) return null;
  const { mode } = captureSettings();
  const report = (path: CaptureEventDetail['path'], worker: CaptureEventDetail['worker'], frozenAt: number, stages: CaptureStages) =>
    reportCapture({
      path,
      mode,
      kind: 'appearance',
      worker,
      priority: 'guide',
      frozenAt,
      readyAt: performance.now(),
      capturedAt: Date.now() - (performance.now() - frozenAt),
      byteLength: 0,
      width: DEFAULT_APPEARANCE_CONFIG.grid,
      height: DEFAULT_APPEARANCE_CONFIG.grid,
      stages,
    });
  if (mode === 'legacy' || !isOffscreenJpegSupported()) {
    const frozenAt = performance.now();
    const grid = sampleVideoAnchorGrid(video, box, mirror);
    const ms = performance.now() - frozenAt;
    report('main', null, frozenAt, { drawMs: ms, workMs: ms });
    return grid;
  }
  if (mode === 'v2' && isV2Supported()) {
    const sampled = await sampleAppearanceV2(video, box, mirror);
    if (!sampled) return null;
    const { doneAbs: _doneAbs, ...stages } = sampled.stages;
    report(sampled.source === 'stream' ? 'v2-stream' : 'v2-bitmap', 'guide', sampled.frozenAt, {
      ...stages,
      replyMs: sampled.replyMs,
      ...(sampled.source === 'bitmap' ? { bitmapMs: sampled.bitmapMs } : {}),
    });
    return sampled.grid;
  }
  const frozenAt = performance.now();
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(video);
  } catch {
    return sampleVideoAnchorGrid(video, box, mirror);
  }
  const bitmapMs = performance.now() - frozenAt;
  const sampled = await sampleAppearanceOffThread(bitmap, { mirror, appearance: { box, config: DEFAULT_APPEARANCE_CONFIG } });
  report('worker', 'shared', frozenAt, workerStages(sampled.stages, bitmapMs));
  return sampled.grid;
}

/**
 * Uniform-scale one already-decoded, already-oriented source to the transport pixel/byte bounds and
 * encode it as JPEG. Never flips: orientation is decided where the pixels are first read.
 */
function encodeScaledFrame(
  source: CanvasImageSource,
  width: number,
  height: number,
  subject: SceneSubject
): ProcessedImage {
  const { targetWidth, targetHeight } = computeTransportSize(width, height, subject);
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas 2D 컨텍스트를 생성할 수 없습니다.');
  }

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, targetWidth, targetHeight);
  ctx.drawImage(source, 0, 0, targetWidth, targetHeight);

  // Progressive quality search to satisfy both byte length and base64 length limits
  let quality = JPEG_START_QUALITY;
  let dataUrl = canvas.toDataURL('image/jpeg', quality);
  let rawBase64 = dataUrl.replace(/^data:image\/jpeg;base64,/, '');
  let approxBytes = Math.ceil((rawBase64.length * 3) / 4);

  while ((approxBytes > MAX_DECODED_BYTES || rawBase64.length > MAX_BASE64_LENGTH) && quality > 0.3) {
    quality -= 0.1;
    dataUrl = canvas.toDataURL('image/jpeg', quality);
    rawBase64 = dataUrl.replace(/^data:image\/jpeg;base64,/, '');
    approxBytes = Math.ceil((rawBase64.length * 3) / 4);
  }

  if (approxBytes > MAX_DECODED_BYTES || rawBase64.length > MAX_BASE64_LENGTH) {
    throw new Error(`${subject} 압축 후에도 크기 제한(1.5MB)을 초과했습니다. 다른 입력을 사용해주세요.`);
  }

  return {
    rawBase64,
    dataUrl,
    width: targetWidth,
    height: targetHeight,
    byteLength: approxBytes,
  };
}
