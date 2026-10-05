/**
 * The off-main-thread capture worker.
 *
 * A real Vite worker module (`new Worker(new URL(..., import.meta.url), { type: 'module' })`) rather
 * than a Blob of source text: Vite emits it as a normal asset, it is type-checked with the rest of the
 * app, and it needs no `blob:` CSP allowance. The main thread sends an `ImageBitmap` (transferred, not
 * copied) and receives the result. Three jobs:
 *
 * - `capture`: a RAW video frame (`createImageBitmap(video)`, unmirrored). The worker freezes it into a
 *   full-size snapshot with the mirror applied, reduces that snapshot to the luma grid (and, on request,
 *   to the appearance grid inside a box), scales it to the transport size, and encodes the JPEG and its
 *   base64. Everything the main thread used to do with `drawImage(video)` + `getImageData` + the encode
 *   happens here; the main thread only asks for the bitmap and reads the reply.
 * - `appearance`: a raw video frame reduced to the appearance grid inside a box (the 250 ms
 *   `target_changed` sample) — the same freeze + crop the capture job uses.
 * - `encode`: an already drawn, already sized frame (the fallback path, `captureVideoFrameViaCanvas`).
 *
 * Jobs run one at a time, tracking frames first (`CaptureJobQueue`).
 */
import type { Box } from '@visual-coach/visual-tools';

import {
  LUMA_GRID_HEIGHT,
  LUMA_GRID_WIDTH,
  configureLadderContexts,
  reduceToLumaGrid,
  type Context2D,
} from '../camera/luma';
import type { AppearanceConfig } from '../intent/anchorAppearance';
import { reduceAnchorGrid } from '../intent/anchorSampler';
import { CaptureJobQueue, type CapturePriority } from './captureJobQueue';

interface JobBase {
  id: string;
  priority: CapturePriority;
  imageBitmap: ImageBitmap;
  /** `performance.timeOrigin + performance.now()` on the page when posted (stage timing). */
  postedAbs: number;
  /** Set on arrival here. */
  receivedAbs?: number;
}

/** Stage times of one job (ms, this worker's clock); see `CaptureStages` in image-processor.ts. */
export interface WorkerStages {
  postMs: number;
  queueMs: number;
  drawMs: number;
  lumaMs: number;
  appearanceMs: number;
  encodeMs: number;
  b64Ms: number;
  workMs: number;
  /** `performance.timeOrigin + performance.now()` when the reply was posted. */
  doneAbs: number;
}

const absNow = () => performance.timeOrigin + performance.now();

function startStages(job: JobBase): WorkerStages {
  const receivedAbs = job.receivedAbs ?? absNow();
  return {
    postMs: receivedAbs - job.postedAbs,
    queueMs: absNow() - receivedAbs,
    drawMs: 0,
    lumaMs: 0,
    appearanceMs: 0,
    encodeMs: 0,
    b64Ms: 0,
    workMs: 0,
    doneAbs: 0,
  };
}

function finishStages(stages: WorkerStages): WorkerStages {
  stages.workMs = stages.drawMs + stages.lumaMs + stages.appearanceMs + stages.encodeMs + stages.b64Ms;
  stages.doneAbs = absNow();
  return stages;
}

export interface JpegBounds {
  quality: number;
  /** Decoded JPEG byte bound (the server's 1.5MB cap). */
  maxBytes: number;
  /** Base64 character bound; base64 length for n bytes is ceil(n/3)*4. */
  maxBase64Length: number;
}

export interface AppearanceRequest {
  box: Box;
  config: AppearanceConfig;
}

export interface JpegEncodeRequest extends JobBase, JpegBounds {
  kind: 'encode';
}

export interface FrameCaptureRequest extends JobBase, JpegBounds {
  kind: 'capture';
  /** Flip horizontally while freezing, exactly as `drawVideoFrame` does on the main thread. */
  mirror: boolean;
  /** Transport size (`computeTransportSize` on the main thread); the frame size unless that is over the bounds. */
  targetWidth: number;
  targetHeight: number;
  /** Also reduce the mirrored snapshot inside this box to the appearance grid. */
  appearance: AppearanceRequest | null;
}

export interface AppearanceSampleRequest extends JobBase {
  kind: 'appearance';
  mirror: boolean;
  appearance: AppearanceRequest;
}

export type CaptureWorkerRequest = JpegEncodeRequest | FrameCaptureRequest | AppearanceSampleRequest;

export interface JpegEncodeReply {
  id: string;
  arrayBuffer: ArrayBuffer;
  width: number;
  height: number;
  stages: WorkerStages;
}

export interface FrameCaptureReply {
  id: string;
  rawBase64: string;
  width: number;
  height: number;
  byteLength: number;
  lumaGrid: Float32Array | null;
  appearanceGrid: Float32Array | null;
  stages: WorkerStages;
}

export interface AppearanceSampleReply {
  id: string;
  appearanceGrid: Float32Array | null;
  stages: WorkerStages;
}

export type CaptureWorkerResponse =
  | JpegEncodeReply
  | FrameCaptureReply
  | AppearanceSampleReply
  | { id: string; error: string };

/** Scratch canvases, reused across jobs. Jobs run one at a time, so no two jobs ever share one mid-use. */
let snapshot: OffscreenCanvasRenderingContext2D | null = null;
let scaled: OffscreenCanvasRenderingContext2D | null = null;
let lumaLadder: [Context2D, Context2D] | null = null;
let lumaSampler: Context2D | null = null;
let appearanceLadder: [Context2D, Context2D] | null = null;
let appearanceGridContext: Context2D | null = null;

function context2d(width: number, height: number, willReadFrequently = false): OffscreenCanvasRenderingContext2D {
  const context = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently });
  if (!context) throw new Error('OffscreenCanvas 2D 컨텍스트를 생성할 수 없습니다.');
  return context;
}

function sized(
  context: OffscreenCanvasRenderingContext2D | null,
  width: number,
  height: number
): OffscreenCanvasRenderingContext2D {
  if (!context) return context2d(width, height);
  if (context.canvas.width !== width || context.canvas.height !== height) {
    context.canvas.width = width;
    context.canvas.height = height;
  }
  return context;
}

/**
 * Freeze the raw frame into the full-size snapshot, mirrored when asked, and release the bitmap. The
 * mirror is applied here and only here — every reduction below reads the snapshot unflipped — the same
 * rule `drawVideoFrame` keeps on the main thread.
 */
function freeze(bitmap: ImageBitmap, mirror: boolean): OffscreenCanvas {
  const width = bitmap.width;
  const height = bitmap.height;
  try {
    const context = sized(snapshot, width, height);
    snapshot = context;
    try {
      context.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? width : 0, 0);
      context.drawImage(bitmap, 0, 0, width, height);
    } finally {
      context.setTransform(1, 0, 0, 1, 0, 0);
    }
    return context.canvas;
  } finally {
    bitmap.close();
  }
}

function lumaGridOf(source: OffscreenCanvas): Float32Array | null {
  if (!lumaLadder) {
    lumaLadder = [context2d(1, 1), context2d(1, 1)];
    configureLadderContexts(lumaLadder);
  }
  lumaSampler ??= context2d(LUMA_GRID_WIDTH, LUMA_GRID_HEIGHT, true);
  return reduceToLumaGrid(source, { width: source.width, height: source.height }, lumaLadder, lumaSampler);
}

function appearanceGridOf(source: OffscreenCanvas, request: AppearanceRequest): Float32Array | null {
  if (!appearanceLadder) {
    appearanceLadder = [context2d(1, 1), context2d(1, 1)];
    configureLadderContexts(appearanceLadder);
  }
  if (!appearanceGridContext) {
    appearanceGridContext = context2d(request.config.grid, request.config.grid, true);
    appearanceGridContext.imageSmoothingEnabled = true;
    appearanceGridContext.imageSmoothingQuality = 'low';
  }
  return reduceAnchorGrid(
    source,
    { width: source.width, height: source.height },
    request.box,
    request.config,
    appearanceLadder,
    appearanceGridContext
  );
}

/**
 * The progressive quality search, the same schedule as the main-thread encoder: same starting quality,
 * same step, same floor, same byte/base64 bounds. This is a shared SCHEDULE, not a claim that the two
 * encoders accept and reject exactly the same frames — they are different implementations and are not
 * bit-identical.
 */
async function encodeJpeg(canvas: OffscreenCanvas, bounds: JpegBounds): Promise<ArrayBuffer> {
  let quality = bounds.quality;
  let buffer = await (await canvas.convertToBlob({ type: 'image/jpeg', quality })).arrayBuffer();
  while (
    (buffer.byteLength > bounds.maxBytes || Math.ceil(buffer.byteLength / 3) * 4 > bounds.maxBase64Length) &&
    quality > 0.3
  ) {
    quality -= 0.1;
    buffer = await (await canvas.convertToBlob({ type: 'image/jpeg', quality })).arrayBuffer();
  }
  return buffer;
}

/** Base64 in chunks: a byte-at-a-time concat would allocate a new string per byte. */
function toBase64(bytes: Uint8Array): string {
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function post(message: CaptureWorkerResponse, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(message, transfer);
}

function gridBuffers(...grids: Array<Float32Array | null>): Transferable[] {
  return grids.filter((grid): grid is Float32Array => grid !== null).map((grid) => grid.buffer);
}

async function runEncode(job: JpegEncodeRequest): Promise<void> {
  const stages = startStages(job);
  let t = absNow();
  const { id, imageBitmap } = job;
  const width = imageBitmap.width;
  const height = imageBitmap.height;
  let canvas: OffscreenCanvas;
  // The transferred bitmap is ours on arrival and is only needed until it has been drawn. Release it on
  // EVERY path — a null 2D context or a failed draw included — so an encode that cannot produce a reply
  // still does not leak the transfer.
  try {
    const context = sized(scaled, width, height);
    scaled = context;
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, width, height);
    context.drawImage(imageBitmap, 0, 0);
    canvas = context.canvas;
  } finally {
    imageBitmap.close();
  }
  stages.drawMs = absNow() - t;
  t = absNow();
  const buffer = await encodeJpeg(canvas, job);
  stages.encodeMs = absNow() - t;
  post({ id, arrayBuffer: buffer, width, height, stages: finishStages(stages) }, [buffer]);
}

async function runCapture(job: FrameCaptureRequest): Promise<void> {
  const stages = startStages(job);
  let t = absNow();
  const frozen = freeze(job.imageBitmap, job.mirror);
  stages.drawMs = absNow() - t;
  t = absNow();
  const lumaGrid = lumaGridOf(frozen);
  stages.lumaMs = absNow() - t;
  t = absNow();
  const appearanceGrid = job.appearance ? appearanceGridOf(frozen, job.appearance) : null;
  stages.appearanceMs = absNow() - t;

  let out = frozen;
  if (job.targetWidth !== frozen.width || job.targetHeight !== frozen.height) {
    t = absNow();
    const context = sized(scaled, job.targetWidth, job.targetHeight);
    scaled = context;
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, job.targetWidth, job.targetHeight);
    context.drawImage(frozen, 0, 0, job.targetWidth, job.targetHeight);
    out = context.canvas;
    stages.drawMs += absNow() - t;
  }
  t = absNow();
  const buffer = await encodeJpeg(out, job);
  stages.encodeMs = absNow() - t;
  t = absNow();
  const rawBase64 = toBase64(new Uint8Array(buffer));
  stages.b64Ms = absNow() - t;
  const reply: FrameCaptureReply = {
    id: job.id,
    rawBase64,
    width: out.width,
    height: out.height,
    byteLength: buffer.byteLength,
    lumaGrid,
    appearanceGrid,
    stages: finishStages(stages),
  };
  post(reply, gridBuffers(lumaGrid, appearanceGrid));
}

async function runAppearance(job: AppearanceSampleRequest): Promise<void> {
  const stages = startStages(job);
  let t = absNow();
  const frozen = freeze(job.imageBitmap, job.mirror);
  stages.drawMs = absNow() - t;
  t = absNow();
  const appearanceGrid = appearanceGridOf(frozen, job.appearance);
  stages.appearanceMs = absNow() - t;
  const reply: AppearanceSampleReply = { id: job.id, appearanceGrid, stages: finishStages(stages) };
  post(reply, gridBuffers(appearanceGrid));
}

const queue = new CaptureJobQueue<CaptureWorkerRequest>(async (job) => {
  try {
    if (job.kind === 'capture') await runCapture(job);
    else if (job.kind === 'appearance') await runAppearance(job);
    else await runEncode(job);
  } catch (error) {
    // `freeze`/`runEncode` close the bitmap themselves; closing an already closed bitmap is a no-op, and a
    // job that failed before reaching them still owns it.
    job.imageBitmap.close();
    post({ id: job.id, error: error instanceof Error ? error.message : String(error) });
  }
});

self.onmessage = (event: MessageEvent<CaptureWorkerRequest>) => {
  event.data.receivedAbs = absNow();
  queue.push(event.data);
};
