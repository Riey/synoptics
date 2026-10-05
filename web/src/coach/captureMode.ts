/**
 * Which camera-capture path this page runs, chosen by URL query so one build can be A/B-measured on the
 * device that matters (the latency panel at `/__perf/` keeps the query):
 *
 * - `?capture=legacy`: the path before fa9ec32 — the main thread draws the video into a readable canvas
 *   (`drawVideoFrame`) and reduces the luma grid there; the shared worker only encodes the JPEG; the guide's
 *   250 ms appearance sample is a synchronous main-thread draw.
 * - `?capture=worker1`: fa9ec32/b91cc05 — `createImageBitmap(video)` on the main thread, then ONE worker
 *   freezes, mirrors, reduces the luma grid, encodes and base64s every capture of every kind.
 * - `?capture=v2` (default): `captureV2.ts` — a worker per lane (tracking, guide), each reading camera
 *   frames itself where the engine can (`MediaStreamTrackProcessor`), only the work each job needs (no luma
 *   grid, appearance samples reduced from the crop only), tracking frames uploaded as raw JPEG.
 *
 * `?trackscale=auto|960` (v2 only; default off): `auto` lowers the tracking upload to 960 px on the long side
 * while this device's capture cost is high (`trackScale.ts`); `960` forces it. Guide frames keep full size.
 * `?trackupload=json` (v2 only) keeps the JSON + base64 tracking upload, to separate the two v2 changes.
 * `?v2source=bitmap` (v2 only) makes the workers take main-thread `createImageBitmap` snapshots even where
 * the engine could hand them the camera track, to measure v2's two frame sources against each other.
 *
 * Pure (reads only the string it is given), so `node --test` runs it.
 */
export type CaptureMode = 'legacy' | 'worker1' | 'v2';
export type TrackScaleMode = 'off' | 'auto' | '960';
export type TrackUpload = 'binary' | 'json';
export type V2Source = 'auto' | 'bitmap';

export interface CaptureSettings {
  mode: CaptureMode;
  trackScale: TrackScaleMode;
  trackUpload: TrackUpload;
  v2Source: V2Source;
}

export const DEFAULT_CAPTURE_SETTINGS: CaptureSettings = {
  mode: 'v2',
  trackScale: 'off',
  trackUpload: 'binary',
  v2Source: 'auto',
};

function pick<T extends string>(params: URLSearchParams, name: string, allowed: readonly T[], fallback: T): T {
  const raw = params.get(name)?.trim().toLowerCase();
  return raw && (allowed as readonly string[]).includes(raw) ? (raw as T) : fallback;
}

/** Unknown or missing values fall back to the defaults, so a typo never selects a half-configured path. */
export function parseCaptureSettings(search: string): CaptureSettings {
  const params = new URLSearchParams(search);
  return {
    mode: pick(params, 'capture', ['legacy', 'worker1', 'v2'] as const, DEFAULT_CAPTURE_SETTINGS.mode),
    trackScale: pick(params, 'trackscale', ['off', 'auto', '960'] as const, DEFAULT_CAPTURE_SETTINGS.trackScale),
    trackUpload: pick(params, 'trackupload', ['binary', 'json'] as const, DEFAULT_CAPTURE_SETTINGS.trackUpload),
    v2Source: pick(params, 'v2source', ['auto', 'bitmap'] as const, DEFAULT_CAPTURE_SETTINGS.v2Source),
  };
}

let cached: CaptureSettings | null = null;

/** This page's settings, read once (the query does not change without a reload). */
export function captureSettings(): CaptureSettings {
  if (!cached) {
    cached =
      typeof location === 'undefined' ? DEFAULT_CAPTURE_SETTINGS : parseCaptureSettings(location.search ?? '');
  }
  return cached;
}

/** Tests only: replace the page's settings. */
export function setCaptureSettingsForTest(settings: CaptureSettings | null): void {
  cached = settings;
}
