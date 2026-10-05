/**
 * The v2 capture worker. The page runs one instance per lane (`name` 'track' or 'guide', see
 * `captureV2.ts`), so a tracking frame never waits behind a guide capture and the two use two cores.
 *
 * Where it gets pixels from:
 * - `stream`: Chrome/Edge hand this worker the camera track's frames directly (a `MediaStreamTrackProcessor`
 *   readable, transferred by the page). The worker keeps only the newest `VideoFrame` and answers a job
 *   from it — the main thread does nothing per capture but post the request.
 * - `bitmap`: everywhere else the page snapshots the video with `createImageBitmap(video)` (whole frame for
 *   a capture, or only the appearance crop for a sample) and transfers it here.
 *
 * Only what each job needs:
 * - `frame`: one draw of the source, mirrored when asked, straight to the transport size; the JPEG (same
 *   start quality and quality search as every other path); base64 only when the request travels as JSON;
 *   the appearance grid only when asked (a follow's reference). No 48×36 luma grid — it has had no consumer
 *   since the scene-change lane was removed (2026-10-01).
 * - `appearance`: the crop of the tracked box only. From a stream frame: `VideoFrame.copyTo` of the crop and
 *   an area average of its Y plane (`yPlaneToGrid`) — no RGB conversion, no canvas; engines/frames without a
 *   readable I420/NV12 layout fall back to the canvas ladder (`reduceCropToGrid`). From a bitmap (already the
 *   crop): the canvas ladder. The full frame is never drawn for a sample.
 *
 * Every reply carries stage times (`V2Stages`) for the latency panel. Times are measured on this worker's
 * clock; `*Abs` values are `performance.timeOrigin + performance.now()`, comparable across threads.
 */
import type { Box } from '@visual-coach/visual-tools';

import { configureLadderContexts, type Context2D } from '../camera/luma';
import type { AppearanceConfig } from '../intent/anchorAppearance';
import { rawAppearanceCrop, reduceCropToGrid } from '../intent/anchorSampler';
import { LaneQueue } from './laneQueue';
import { computeTransportSize, type SceneSubject } from './transportSize';
import { frameYGrid } from './yPlaneGrid';

export type CaptureLane = 'track' | 'guide';

export interface V2Bounds {
  quality: number;
  maxBytes: number;
  maxBase64Length: number;
}

export interface V2SourceMessage {
  kind: 'source';
  sourceId: string;
  readable: ReadableStream<VideoFrame>;
}

interface JobBase {
  id: string;
  /** `performance.timeOrigin + performance.now()` on the page when posted. */
  postedAbs: number;
  from: 'stream' | 'bitmap';
  /** Transferred; only for `from: 'bitmap'`. */
  imageBitmap?: ImageBitmap;
  mirror: boolean;
}

export interface V2FrameJob extends JobBase, V2Bounds {
  kind: 'frame';
  /** Long-side cap of the upload (the adaptive tracking size), or null for the frame's own size. */
  maxLongSide: number | null;
  /** Never answer from the same camera frame as the previous `distinct` job (tracking frames). */
  distinct: boolean;
  /** Also produce base64 (the request travels as JSON). */
  base64: boolean;
  appearance: { box: Box; config: AppearanceConfig } | null;
}

export interface V2AppearanceJob extends JobBase {
  kind: 'appearance';
  box: Box;
  config: AppearanceConfig;
}

export type V2Job = V2FrameJob | V2AppearanceJob;
export type V2Message = V2SourceMessage | V2Job;

export interface V2Stages {
  /** Page post → this worker received the job. */
  postMs: number;
  /** Received → started (waiting behind other jobs of this lane). */
  queueMs: number;
  /** Started → a camera frame to use (stream: waiting for a newer frame). */
  waitMs: number;
  drawMs: number;
  appearanceMs: number;
  encodeMs: number;
  b64Ms: number;
  /** draw + appearance + encode + base64: this job's CPU work. */
  workMs: number;
  /** `performance.timeOrigin + performance.now()` when the reply was posted. */
  doneAbs: number;
}

export interface V2FrameReply {
  id: string;
  ok: 'frame';
  blob: Blob;
  rawBase64: string | null;
  width: number;
  height: number;
  byteLength: number;
  appearanceGrid: Float32Array | null;
  /** Stream frames: when the frame reached this worker (wall clock, and the cross-thread clock). */
  frameWall: number | null;
  frameAbs: number | null;
  stages: V2Stages;
}

export interface V2AppearanceReply {
  id: string;
  ok: 'appearance';
  grid: Float32Array | null;
  stages: V2Stages;
}

/**
 * `unavailable`: the stream source has no usable frame (none yet, ended, or no newer one in time) — the
 * page answers this job from a bitmap instead. `superseded`: a newer sample replaced it in the queue.
 */
export type V2Reply =
  | V2FrameReply
  | V2AppearanceReply
  | { id: string; unavailable: 'no_source' | 'no_frame' | 'ended' }
  | { id: string; superseded: true }
  | { id: string; error: string };

const lane: CaptureLane = (self as unknown as { name?: string }).name === 'guide' ? 'guide' : 'track';
const SUBJECT: SceneSubject = '카메라 화면';
/** How long a stream job waits for a (newer) camera frame before the page falls back to a bitmap. */
const FRAME_WAIT_MS = 400;

const absNow = () => performance.timeOrigin + performance.now();

// ------------------------------------------------------------------------------------------ stream source

interface StreamSource {
  id: string;
  reader: ReadableStreamDefaultReader<VideoFrame>;
  latest: VideoFrame | null;
  latestWall: number;
  latestAbs: number;
  ended: boolean;
  waiters: Set<() => void>;
}

let source: StreamSource | null = null;
/** The camera timestamp of the last `distinct` job's frame. */
let lastDistinctTimestamp: number | null = null;

function wake(src: StreamSource): void {
  const waiters = [...src.waiters];
  src.waiters.clear();
  for (const waiter of waiters) waiter();
}

async function pump(src: StreamSource): Promise<void> {
  try {
    for (;;) {
      const { value, done } = await src.reader.read();
      if (done) break;
      if (source !== src) {
        value.close();
        break;
      }
      // Only the newest frame is kept: the camera's buffer pool is small and an old frame is never used.
      src.latest?.close();
      src.latest = value;
      src.latestWall = Date.now();
      src.latestAbs = absNow();
      wake(src);
    }
  } catch {
    // A reader that errors (track ended under us, stream torn down) is the same as one that ended.
  }
  src.ended = true;
  src.latest?.close();
  src.latest = null;
  wake(src);
}

function attach(message: V2SourceMessage): void {
  const previous = source;
  source = null;
  if (previous) {
    previous.ended = true;
    previous.latest?.close();
    previous.latest = null;
    wake(previous);
    // Releases this worker's read of the old track; the track itself is the page's and stays as it is.
    void previous.reader.cancel().catch(() => {});
  }
  const src: StreamSource = {
    id: message.sourceId,
    reader: message.readable.getReader(),
    latest: null,
    latestWall: 0,
    latestAbs: 0,
    ended: false,
    waiters: new Set(),
  };
  source = src;
  void pump(src);
}

type Taken =
  | { frame: VideoFrame; wall: number; abs: number }
  | { unavailable: 'no_source' | 'no_frame' | 'ended' };

/**
 * A clone of the newest stream frame (the caller closes it), or why there is none. Cloned synchronously
 * at the moment it is picked: the pump closes the frame it replaces, possibly before the caller resumes.
 */
async function takeStreamFrame(distinct: boolean): Promise<Taken> {
  const src = source;
  if (!src) return { unavailable: 'no_source' };
  const deadline = performance.now() + FRAME_WAIT_MS;
  for (;;) {
    if (src.ended || source !== src) return { unavailable: 'ended' };
    const latest = src.latest;
    if (latest && !(distinct && latest.timestamp === lastDistinctTimestamp)) {
      if (distinct) lastDistinctTimestamp = latest.timestamp;
      return { frame: latest.clone(), wall: src.latestWall, abs: src.latestAbs };
    }
    const remaining = deadline - performance.now();
    if (remaining <= 0) return { unavailable: 'no_frame' };
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        src.waiters.delete(done);
        resolve();
      }, remaining);
      const done = () => {
        clearTimeout(timer);
        resolve();
      };
      src.waiters.add(done);
    });
  }
}

// ------------------------------------------------------------------------------------------ canvases

let out: OffscreenCanvasRenderingContext2D | null = null;
let ladder: [Context2D, Context2D] | null = null;
let gridContext: Context2D | null = null;

function context2d(width: number, height: number, willReadFrequently = false): OffscreenCanvasRenderingContext2D {
  const context = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently });
  if (!context) throw new Error('OffscreenCanvas 2D 컨텍스트를 생성할 수 없습니다.');
  return context;
}

function appearanceContexts(config: AppearanceConfig): { ladder: [Context2D, Context2D]; grid: Context2D } {
  if (!ladder) {
    ladder = [context2d(1, 1), context2d(1, 1)];
    configureLadderContexts(ladder);
  }
  if (!gridContext) {
    gridContext = context2d(config.grid, config.grid, true);
    gridContext.imageSmoothingEnabled = true;
    gridContext.imageSmoothingQuality = 'low';
  }
  return { ladder, grid: gridContext };
}

type Drawable = VideoFrame | ImageBitmap;

/** The appearance grid of a raw crop: the Y plane of a stream frame where possible, else the canvas ladder. */
async function appearanceOf(drawable: Drawable, crop: { x: number; y: number; width: number; height: number }, config: AppearanceConfig, mirror: boolean): Promise<Float32Array | null> {
  if (!(drawable instanceof ImageBitmap)) {
    const grid = await frameYGrid(drawable, crop, config.grid, mirror);
    if (grid) return grid;
  }
  const { ladder: steps, grid: gridOut } = appearanceContexts(config);
  return reduceCropToGrid(drawable, crop, config, steps, gridOut, mirror);
}

function sizeOf(drawable: Drawable): { width: number; height: number } {
  return drawable instanceof ImageBitmap
    ? { width: drawable.width, height: drawable.height }
    : { width: drawable.displayWidth, height: drawable.displayHeight };
}

/** The source drawn once, mirrored when asked, at the transport size. */
function drawFrame(drawable: Drawable, width: number, height: number, mirror: boolean): OffscreenCanvas {
  if (!out) out = context2d(width, height);
  if (out.canvas.width !== width || out.canvas.height !== height) {
    out.canvas.width = width;
    out.canvas.height = height;
  }
  try {
    out.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? width : 0, 0);
    out.drawImage(drawable, 0, 0, width, height);
  } finally {
    out.setTransform(1, 0, 0, 1, 0, 0);
  }
  return out.canvas;
}

/** The same quality search as every other path: start quality, 0.1 steps, floor 0.3, byte/base64 bounds. */
async function encodeJpeg(canvas: OffscreenCanvas, bounds: V2Bounds): Promise<Blob> {
  const over = (blob: Blob) => blob.size > bounds.maxBytes || Math.ceil(blob.size / 3) * 4 > bounds.maxBase64Length;
  let quality = bounds.quality;
  let blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  while (over(blob) && quality > 0.3) {
    quality -= 0.1;
    blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  }
  return blob;
}

type Base64Bytes = Uint8Array & { toBase64?: () => string };

/** Native `Uint8Array.prototype.toBase64` where the engine has it, chunked `btoa` otherwise. */
function toBase64(bytes: Base64Bytes): string {
  if (typeof bytes.toBase64 === 'function') return bytes.toBase64();
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  return btoa(binary);
}

// ------------------------------------------------------------------------------------------ jobs

interface Received {
  job: V2Job;
  receivedAbs: number;
}

function post(message: V2Reply, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(message, transfer);
}

interface Pixels {
  drawable: Drawable;
  release: () => void;
  wall: number | null;
  abs: number | null;
}

async function pixelsFor(job: V2Job, distinct: boolean): Promise<Pixels | { unavailable: 'no_source' | 'no_frame' | 'ended' }> {
  if (job.from === 'bitmap') {
    const bitmap = job.imageBitmap;
    if (!bitmap) throw new Error('캡처 비트맵이 없습니다.');
    return { drawable: bitmap, release: () => bitmap.close(), wall: null, abs: null };
  }
  const taken = await takeStreamFrame(distinct);
  if ('unavailable' in taken) return taken;
  return { drawable: taken.frame, release: () => taken.frame.close(), wall: taken.wall, abs: taken.abs };
}

async function runJob({ job, receivedAbs }: Received): Promise<void> {
  const startedAbs = absNow();
  const stages: V2Stages = {
    postMs: receivedAbs - job.postedAbs,
    queueMs: startedAbs - receivedAbs,
    waitMs: 0,
    drawMs: 0,
    appearanceMs: 0,
    encodeMs: 0,
    b64Ms: 0,
    workMs: 0,
    doneAbs: 0,
  };
  const pixels = await pixelsFor(job, job.kind === 'frame' && job.distinct);
  if ('unavailable' in pixels) {
    post({ id: job.id, unavailable: pixels.unavailable });
    return;
  }
  stages.waitMs = absNow() - startedAbs;
  const { drawable } = pixels;
  let t = absNow();
  if (job.kind === 'appearance') {
    let grid: Float32Array | null;
    try {
      const size = sizeOf(drawable);
      // A bitmap sample is already the crop (the page cut it out); a stream frame is the whole frame.
      const crop =
        job.from === 'bitmap'
          ? { x: 0, y: 0, width: size.width, height: size.height }
          : rawAppearanceCrop(job.box, size, job.config, job.mirror);
      grid = crop ? await appearanceOf(drawable, crop, job.config, job.mirror) : null;
    } finally {
      pixels.release();
    }
    stages.appearanceMs = absNow() - t;
    stages.workMs = stages.appearanceMs;
    stages.doneAbs = absNow();
    post({ id: job.id, ok: 'appearance', grid, stages }, grid ? [grid.buffer] : []);
    return;
  }

  let canvas: OffscreenCanvas;
  let appearanceGrid: Float32Array | null = null;
  try {
    const size = sizeOf(drawable);
    const { targetWidth, targetHeight } = computeTransportSize(size.width, size.height, SUBJECT, job.maxLongSide ?? undefined);
    canvas = drawFrame(drawable, targetWidth, targetHeight, job.mirror);
    stages.drawMs = absNow() - t;
    if (job.appearance) {
      t = absNow();
      const crop = rawAppearanceCrop(job.appearance.box, size, job.appearance.config, job.mirror);
      // The same method as the samples it will be compared with (the Y plane for a stream frame).
      appearanceGrid = crop ? await appearanceOf(drawable, crop, job.appearance.config, job.mirror) : null;
      stages.appearanceMs = absNow() - t;
    }
  } finally {
    // Everything that reads the source is above; the encode reads the canvas.
    pixels.release();
  }
  t = absNow();
  const blob = await encodeJpeg(canvas, job);
  stages.encodeMs = absNow() - t;
  let rawBase64: string | null = null;
  if (job.base64) {
    t = absNow();
    rawBase64 = toBase64(new Uint8Array(await blob.arrayBuffer()));
    stages.b64Ms = absNow() - t;
  }
  stages.workMs = stages.drawMs + stages.appearanceMs + stages.encodeMs + stages.b64Ms;
  stages.doneAbs = absNow();
  const reply: V2FrameReply = {
    id: job.id,
    ok: 'frame',
    blob,
    rawBase64,
    width: canvas.width,
    height: canvas.height,
    byteLength: blob.size,
    appearanceGrid,
    frameWall: pixels.wall,
    frameAbs: pixels.abs,
    stages,
  };
  post(reply, appearanceGrid ? [appearanceGrid.buffer] : []);
}

const queue = new LaneQueue<V2Job['kind'], Received & { kind: V2Job['kind'] }>(
  // Guide lane: a follow/confirm frame (a request waits on it) before an appearance sample, and only the
  // newest sample waits. The tracking lane only ever gets frames; they are never superseded, because the
  // tracking loop awaits every one it asks for (and asks for one at a time).
  { order: ['frame', 'appearance'], latestWins: lane === 'guide' ? ['appearance'] : [] },
  async (received) => {
    try {
      await runJob(received);
    } catch (error) {
      received.job.imageBitmap?.close();
      post({ id: received.job.id, error: error instanceof Error ? error.message : String(error) });
    }
  },
  (superseded) => {
    superseded.job.imageBitmap?.close();
    post({ id: superseded.job.id, superseded: true });
  }
);

self.onmessage = (event: MessageEvent<V2Message>) => {
  const message = event.data;
  if (message.kind === 'source') {
    attach(message);
    return;
  }
  queue.push({ job: message, receivedAbs: absNow(), kind: message.kind });
};
