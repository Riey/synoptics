/**
 * Thin transport for the two tracker routes (contract v4 §1).
 *
 * Mirrors `coach/api.ts`'s conventions: JSON (tracking frames may also go as raw JPEG), `credentials: 'same-origin'`, the server's own
 * error token preserved on the thrown `ApiError`. The 409 bodies' extra fields are not needed to
 * handle a response correctly, so they are not parsed into custom types.
 */
import type { Box } from '@visual-coach/visual-tools';

import { ApiError } from '../coach/api';
import { captureSettings } from '../coach/captureMode';
import type {
  TrackControlRequest,
  TrackControlResponse,
  TrackFrameQuery,
  TrackFrameRequest,
  TrackFrameResponse,
  TrackSelectRequest,
  TrackSelectResponse,
} from '../generated/api.generated';

async function postTrack<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    credentials: 'same-origin',
    signal,
  });
  return readTrackResponse<T>(response);
}

async function readTrackResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let code = 'request_failed';
    try {
      const body: unknown = await response.json();
      if (body && typeof body === 'object' && typeof (body as Record<string, unknown>).error === 'string') {
        code = (body as Record<string, string>).error;
      }
    } catch {
      // keep the fallback code
    }
    throw new ApiError(response.status, code);
  }

  return (await response.json()) as T;
}

export function trackControl(payload: TrackControlRequest, signal?: AbortSignal): Promise<TrackControlResponse> {
  return postTrack<TrackControlResponse>('/api/track/control', payload, signal);
}

/** The server refused the raw-JPEG form once (an older server: `415 json_required`); JSON from then on. */
let binaryRefused = false;

/** Whether tracking frames go as raw JPEG: v2 capture, not opted out (`?trackupload=json`), not refused. */
export function trackFrameBinaryAvailable(): boolean {
  const settings = captureSettings();
  return settings.mode === 'v2' && settings.trackUpload === 'binary' && !binaryRefused;
}

/** Base64 of a JPEG on the main thread: only for the one frame a refusing server sends back as JSON. */
async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  return btoa(binary);
}

/**
 * One tracking frame. A raw JPEG goes as the request body (`Content-Type: image/jpeg`, the fields in the
 * query string — no base64, a third less upload); base64 goes as JSON (`TrackFrameRequest`). A seeded first
 * frame is always JSON (the binary form carries no seed). If the server refuses the binary form with 415,
 * this same frame is resent as JSON and every later one goes as JSON; the refusal consumed no frame
 * sequence (the server rejects it before admission).
 */
export async function trackFrame(
  query: TrackFrameQuery,
  image: { jpeg: Blob } | { base64: string },
  seedBox: Box | undefined,
  signal?: AbortSignal
): Promise<TrackFrameResponse> {
  if ('jpeg' in image && !seedBox && !binaryRefused) {
    try {
      return await postTrackJpeg(query, image.jpeg, signal);
    } catch (error) {
      if (!(error instanceof ApiError && error.status === 415)) throw error;
      binaryRefused = true;
    }
  }
  const imageBase64 = 'jpeg' in image ? await blobToBase64(image.jpeg) : image.base64;
  const payload: TrackFrameRequest = { ...query, image_base64: imageBase64, ...(seedBox ? { seed_box: seedBox } : {}) };
  return postTrack<TrackFrameResponse>('/api/track/frame', payload, signal);
}

async function postTrackJpeg(query: TrackFrameQuery, jpeg: Blob, signal?: AbortSignal): Promise<TrackFrameResponse> {
  const search = new URLSearchParams({
    session_id: query.session_id,
    run_id: query.run_id,
    frame_id: query.frame_id,
    frame_seq: String(query.frame_seq),
  });
  const response = await fetch(`/api/track/frame?${search}`, {
    method: 'POST',
    headers: { 'Content-Type': 'image/jpeg' },
    body: jpeg,
    credentials: 'same-origin',
    signal,
  });
  return readTrackResponse<TrackFrameResponse>(response);
}

/**
 * The one paid startup call: analyze the captured frame plus the current task and choose the object to
 * track. It touches no tracker run — a successful selection is followed by an ordinary
 * `/api/track/control` start from the caller.
 */
export function selectTarget(payload: TrackSelectRequest, signal?: AbortSignal): Promise<TrackSelectResponse> {
  return postTrack<TrackSelectResponse>('/api/track/select', payload, signal);
}
