/**
 * Main-thread side of the v2 capture path (`?capture=v2`, the default; see `captureMode.ts`).
 *
 * What changed against worker1 (one shared worker doing everything for every capture):
 * - Two workers, one per lane (`capture-v2.worker.ts`): tracking frames and guide work (plan/follow/
 *   confirm/talk frames, the 250 ms appearance samples) never queue behind each other and run on two cores.
 * - Where the engine has `MediaStreamTrackProcessor` (Chrome/Edge), each worker reads the camera track's
 *   frames itself: the main thread does nothing per capture but post a request — no `createImageBitmap`,
 *   no draw. Elsewhere (Safari, Firefox, a file source) the main thread snapshots with `createImageBitmap`
 *   as before, and an appearance sample snapshots only the crop it needs.
 * - Only the work each job needs: no luma grid (no consumer since 2026-10-01), one draw straight to the
 *   upload size, base64 only for requests that travel as JSON (tracking frames go as raw JPEG,
 *   `trackClient.ts`), the appearance grid from the crop only.
 * - Back-pressure: an appearance sample is skipped (`CaptureSkippedError`) while the tracking lane is behind
 *   — a tracking capture is in flight for longer than its own recent cost — so the guide never adds work
 *   when the live box is already late. Inside the guide worker only the newest sample waits.
 *
 * Contracts kept from every other path: `capturedAt` is when the pixels were frozen (for a stream frame,
 * when it reached the worker — never later than the moment it was asked for); the mirror is applied once,
 * while drawing; a follow's appearance reference comes from the very frame that is encoded; the transport
 * bounds and quality search are shared (`transportSize.ts`); every failure settles the request.
 */
import type { Box } from '@visual-coach/visual-tools';

import { DEFAULT_APPEARANCE_CONFIG } from '../intent/anchorAppearance';
import { rawAppearanceCrop } from '../intent/anchorSampler';
import { DurationEstimate } from '../tracking/framePrefetch';
import type { CaptureLane, V2Job, V2Reply, V2Stages } from './capture-v2.worker';
import { captureSettings } from './captureMode';
import { TrackScaleController } from './trackScale';
import { JPEG_START_QUALITY, MAX_BASE64_LENGTH, MAX_DECODED_BYTES } from './transportSize';

export type { CaptureLane, V2Stages } from './capture-v2.worker';

/** The camera has no readable frame yet (transient: try again on the next frame). */
export class V2FrameUnavailableError extends Error {}

/** An appearance sample was not taken because the tracking lane is behind. Not an error: try later. */
export class CaptureSkippedError extends Error {}

export interface V2Captured {
  blob: Blob;
  rawBase64: string | null;
  width: number;
  height: number;
  byteLength: number;
  appearanceGrid: Float32Array | null;
  capturedAt: number;
  /** Main-thread `performance.now()` of the freeze. */
  frozenAt: number;
  source: 'stream' | 'bitmap';
  /** Main-thread `createImageBitmap` time (bitmap source only). */
  bitmapMs: number;
  /** Worker reply posted → received here. */
  replyMs: number;
  stages: V2Stages;
  /** The upload was reduced by the adaptive tracking size. */
  reduced: boolean;
}

export interface V2Sampled {
  grid: Float32Array | null;
  source: 'stream' | 'bitmap';
  bitmapMs: number;
  replyMs: number;
  stages: V2Stages;
  frozenAt: number;
}

/** Whether this engine can run the v2 workers at all (the same APIs worker1 needs). */
export function isV2Supported(): boolean {
  return (
    typeof Worker !== 'undefined' &&
    typeof createImageBitmap === 'function' &&
    typeof OffscreenCanvas !== 'undefined' &&
    typeof OffscreenCanvas.prototype.convertToBlob === 'function'
  );
}

type TrackProcessorCtor = new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> };

function trackProcessorCtor(): TrackProcessorCtor | null {
  const ctor = (globalThis as { MediaStreamTrackProcessor?: TrackProcessorCtor }).MediaStreamTrackProcessor;
  return typeof ctor === 'function' && typeof VideoFrame === 'function' ? ctor : null;
}

/** The live camera track behind a video element, or null (a file source, an ended track). */
function liveTrackOf(video: HTMLVideoElement): MediaStreamTrack | null {
  const stream = video.srcObject;
  if (typeof MediaStream === 'undefined' || !(stream instanceof MediaStream)) return null;
  const track = stream.getVideoTracks()[0];
  return track && track.readyState === 'live' ? track : null;
}

const absNow = () => performance.timeOrigin + performance.now();

type Unsent<J> = J extends V2Job ? Omit<J, 'id' | 'postedAbs'> : never;

interface Pending {
  resolve: (reply: V2Reply) => void;
  reject: (error: Error) => void;
  postedAt: number;
}

class Lane {
  readonly name: CaptureLane;
  private worker: Worker | null = null;
  private readonly pending = new Map<string, Pending>();
  private seq = 0;
  /** The track id this lane's worker reads, or null. */
  private sourceId: string | null = null;
  /** The stream hand-off failed once on this engine: bitmaps only from now on. */
  private streamBroken = false;
  /** This lane's capture cost (request → reply), for the back-pressure rule. */
  readonly cost = new DurationEstimate();

  constructor(name: CaptureLane) {
    this.name = name;
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('./capture-v2.worker.ts', import.meta.url), { type: 'module', name: this.name });
    worker.addEventListener('message', (event: MessageEvent<V2Reply>) => this.onMessage(event));
    worker.addEventListener('error', (event: ErrorEvent) =>
      this.teardown(new Error(`캡처 워커를 불러오지 못했습니다${event.message ? `: ${event.message}` : ''}.`))
    );
    worker.addEventListener('messageerror', () => this.teardown(new Error('캡처 워커 메시지를 해석할 수 없습니다.')));
    this.worker = worker;
    return worker;
  }

  private onMessage(event: MessageEvent<V2Reply>): void {
    const data = event.data as V2Reply | null;
    if (!data || typeof data.id !== 'string') return;
    const request = this.pending.get(data.id);
    if (!request) return;
    this.pending.delete(data.id);
    if ('error' in data) {
      // A real failure inside the worker: retire it so a broken context is never reused.
      const failure = new Error(data.error);
      request.reject(failure);
      this.teardown(failure);
      return;
    }
    if ('unavailable' in data && data.unavailable !== 'no_frame') this.sourceId = null;
    request.resolve(data);
  }

  private teardown(reason: Error): void {
    const worker = this.worker;
    this.worker = null;
    this.sourceId = null;
    worker?.terminate();
    const waiting = [...this.pending.values()];
    this.pending.clear();
    for (const request of waiting) request.reject(reason);
  }

  /**
   * Make this lane's worker read `video`'s camera track, if the engine can hand frames to a worker. True
   * when the worker has (or is about to have) the stream. A track change (a new camera) re-attaches.
   */
  attachStream(video: HTMLVideoElement): boolean {
    if (this.streamBroken || captureSettings().v2Source === 'bitmap') return false;
    const Processor = trackProcessorCtor();
    const track = liveTrackOf(video);
    if (!Processor || !track) return false;
    if (this.sourceId === track.id) return true;
    let worker: Worker;
    try {
      worker = this.ensureWorker();
    } catch {
      return false;
    }
    try {
      const { readable } = new Processor({ track });
      worker.postMessage({ kind: 'source', sourceId: track.id, readable }, [readable as unknown as Transferable]);
    } catch {
      // No transferable streams (or no processor on this track): this engine uses bitmaps.
      this.streamBroken = true;
      return false;
    }
    this.sourceId = track.id;
    return true;
  }

  /** Whether this lane's worker reads `video`'s live camera track right now. */
  readsStream(video: HTMLVideoElement): boolean {
    if (this.streamBroken || this.sourceId === null || captureSettings().v2Source === 'bitmap') return false;
    return liveTrackOf(video)?.id === this.sourceId;
  }

  request(job: Unsent<V2Job>): Promise<V2Reply> {
    const bitmap = job.imageBitmap;
    let worker: Worker;
    try {
      worker = this.ensureWorker();
    } catch (error) {
      bitmap?.close();
      return Promise.reject(error instanceof Error ? error : new Error(String(error)));
    }
    const id = `${this.name}-${(this.seq += 1)}`;
    return new Promise<V2Reply>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, postedAt: performance.now() });
      try {
        worker.postMessage({ ...job, id, postedAbs: absNow() }, bitmap ? [bitmap] : []);
      } catch (error) {
        this.pending.delete(id);
        bitmap?.close();
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  /**
   * Behind: a request has been in flight for longer than this lane's recent cost (×1.5, and at least one
   * camera frame), i.e. it is late now. Nothing in flight, or no cost measured yet, is not behind.
   */
  behind(now: number): boolean {
    const cost = this.cost.get();
    if (cost === null) return false;
    const limit = Math.max(1000 / 30, 1.5 * cost);
    for (const request of this.pending.values()) if (now - request.postedAt > limit) return true;
    return false;
  }
}

const lanes: Record<CaptureLane, Lane> = { track: new Lane('track'), guide: new Lane('guide') };

/** The tracking worker reads `video`'s camera track itself (and so picks only newer frames itself). */
export function trackLaneReadsStream(video: HTMLVideoElement): boolean {
  return lanes.track.readsStream(video);
}
let trackScale: TrackScaleController | null = null;

function scaleController(): TrackScaleController {
  trackScale ??= new TrackScaleController(captureSettings().trackScale);
  return trackScale;
}

const HAVE_CURRENT_DATA = 2;

function frameReady(video: HTMLVideoElement): boolean {
  return video.readyState >= HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0;
}

/** Reject an oversized reply here too, so a worker that ignored its bounds cannot put it on the wire. */
function assertBounded(byteLength: number, base64: string | null): void {
  if (byteLength > MAX_DECODED_BYTES || (base64 !== null && base64.length > MAX_BASE64_LENGTH)) {
    throw new Error('카메라 화면 압축 후에도 크기 제한(1.5MB)을 초과했습니다. 다른 입력을 사용해주세요.');
  }
}

export interface V2FrameOptions {
  lane: CaptureLane;
  mirror: boolean;
  /** Produce base64 (the request travels as JSON). */
  base64: boolean;
  appearanceBox?: Box | null;
}

/** Capture the current camera frame through the lane's worker. */
export async function captureFrameV2(video: HTMLVideoElement, options: V2FrameOptions): Promise<V2Captured> {
  if (!frameReady(video)) throw new V2FrameUnavailableError('카메라 화면이 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.');
  const lane = lanes[options.lane];
  const isTrack = options.lane === 'track';
  const maxLongSide = isTrack ? scaleController().longSide() : null;
  const base = {
    kind: 'frame' as const,
    mirror: options.mirror,
    maxLongSide,
    distinct: isTrack,
    base64: options.base64,
    appearance: options.appearanceBox ? { box: options.appearanceBox, config: DEFAULT_APPEARANCE_CONFIG } : null,
    quality: JPEG_START_QUALITY,
    maxBytes: MAX_DECODED_BYTES,
    maxBase64Length: MAX_BASE64_LENGTH,
  };
  const startedAt = performance.now();

  if (lane.attachStream(video)) {
    const reply = await lane.request({ ...base, from: 'stream' });
    if ('ok' in reply && reply.ok === 'frame') {
      lane.cost.add(performance.now() - startedAt);
      return finishFrame(reply, 'stream', 0, maxLongSide !== null, isTrack);
    }
    // A tracking frame with no NEWER camera frame in time is transient, never a repeat of the last one.
    if ('unavailable' in reply && reply.unavailable === 'no_frame' && isTrack) {
      throw new V2FrameUnavailableError('카메라 화면이 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.');
    }
    // No stream frame (none yet, or the track ended): answer this one from a bitmap below.
  }

  // Read beside the call that snapshots the frame, not after the off-thread work.
  const capturedAt = Date.now();
  const frozenAt = performance.now();
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(video);
  } catch {
    throw new V2FrameUnavailableError('카메라 화면을 읽지 못했습니다. 잠시 후 다시 시도해주세요.');
  }
  const bitmapMs = performance.now() - frozenAt;
  const reply = await lane.request({ ...base, from: 'bitmap', imageBitmap: bitmap });
  if (!('ok' in reply) || reply.ok !== 'frame') throw new Error('캡처 워커가 프레임을 돌려주지 않았습니다.');
  lane.cost.add(performance.now() - startedAt);
  const captured = finishFrame(reply, 'bitmap', bitmapMs, maxLongSide !== null, isTrack);
  return { ...captured, capturedAt, frozenAt };
}

function finishFrame(
  reply: Extract<V2Reply, { ok: 'frame' }>,
  source: 'stream' | 'bitmap',
  bitmapMs: number,
  reduced: boolean,
  isTrack: boolean
): V2Captured {
  assertBounded(reply.byteLength, reply.rawBase64);
  const replyMs = absNow() - reply.stages.doneAbs;
  if (isTrack) scaleController().observe(reply.stages.workMs, performance.now(), reduced);
  return {
    blob: reply.blob,
    rawBase64: reply.rawBase64,
    width: reply.width,
    height: reply.height,
    byteLength: reply.byteLength,
    appearanceGrid: reply.appearanceGrid,
    // Stream frames: when the frame reached the worker (it cannot be younger than that).
    capturedAt: reply.frameWall ?? Date.now(),
    frozenAt: reply.frameAbs !== null ? reply.frameAbs - performance.timeOrigin : performance.now(),
    source,
    bitmapMs,
    replyMs,
    stages: reply.stages,
    reduced,
  };
}

/**
 * The appearance grid inside `box` of the current frame (the guide's 250 ms `target_changed` sample).
 * Throws `CaptureSkippedError` while the tracking lane is behind; null when no sample could be taken.
 */
export async function sampleAppearanceV2(video: HTMLVideoElement, box: Box, mirror: boolean): Promise<V2Sampled | null> {
  if (!frameReady(video)) return null;
  if (lanes.track.behind(performance.now())) throw new CaptureSkippedError('추적 캡처가 밀려 외관 표본을 건너뜁니다.');
  const lane = lanes.guide;
  const config = DEFAULT_APPEARANCE_CONFIG;
  const base = { kind: 'appearance' as const, mirror, box, config };
  if (lane.attachStream(video)) {
    const frozenAt = performance.now();
    const reply = await lane.request({ ...base, from: 'stream' });
    if ('superseded' in reply) throw new CaptureSkippedError('더 새 외관 표본으로 대체되었습니다.');
    if ('ok' in reply && reply.ok === 'appearance') {
      return { grid: reply.grid, source: 'stream', bitmapMs: 0, replyMs: absNow() - reply.stages.doneAbs, stages: reply.stages, frozenAt };
    }
  }
  // Snapshot only the crop (in the raw, unmirrored frame); the worker mirrors it while reducing.
  const crop = rawAppearanceCrop(box, { width: video.videoWidth, height: video.videoHeight }, config, mirror);
  if (!crop) return null;
  const frozenAt = performance.now();
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(video, crop.x, crop.y, crop.width, crop.height);
  } catch {
    return null;
  }
  const bitmapMs = performance.now() - frozenAt;
  const reply = await lane.request({ ...base, from: 'bitmap', imageBitmap: bitmap });
  if ('superseded' in reply) throw new CaptureSkippedError('더 새 외관 표본으로 대체되었습니다.');
  if (!('ok' in reply) || reply.ok !== 'appearance') return null;
  return { grid: reply.grid, source: 'bitmap', bitmapMs, replyMs: absNow() - reply.stages.doneAbs, stages: reply.stages, frozenAt };
}
