/**
 * Transport for the guide lane: `/api/guide/plan` (streamed), `/api/guide/follow`, `/api/guide/confirm`,
 * `/api/guide/talk` and the guide fields of `/api/health`. Same-origin cookie session, JSON bodies, the
 * guidance body cap. Errors are `IntentHttpError` (an `ApiError` that also keeps the JSON body, so 409
 * `stale_plan` can name the plan to continue from).
 */
import type {
  GuideConfirmRequest,
  GuideConfirmResponse,
  GuideFollowRequest,
  GuideFollowResponse,
  GuideHealthResponse,
  GuidePlanRequest,
  GuidePlanResponse,
  GuideTalkRequest,
  GuideTalkResponse,
} from '../generated/api.generated';
import { GUIDANCE_MAX_BODY_BYTES } from '../coach/api';
import { tapGuideCall } from '../debug/guideTap';
import { IntentHttpError, readJsonResponse, readPlanResponse, type PlanPartial } from './planStream';

function serialize(payload: unknown): string {
  const text = JSON.stringify(payload);
  if (new TextEncoder().encode(text).length > GUIDANCE_MAX_BODY_BYTES) throw new IntentHttpError(413, 'body_too_large');
  return text;
}

async function post(path: string, payload: unknown, signal: AbortSignal | undefined, accept?: string): Promise<Response> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (accept) headers.Accept = accept;
  return fetch(path, { method: 'POST', headers, body: serialize(payload), credentials: 'same-origin', signal });
}

/**
 * Ask for a plan. Requests the event stream; `onPartial` receives each display hint the moment it is
 * parsed. A server that answers plain JSON is read exactly like the non-stream route.
 */
export async function requestGuidePlan(
  payload: GuidePlanRequest,
  options: { signal?: AbortSignal; onPartial?: (partial: PlanPartial) => void } = {}
): Promise<GuidePlanResponse> {
  return tapGuideCall('/api/guide/plan', payload, async () => {
    const response = await post('/api/guide/plan', payload, options.signal, 'text/event-stream');
    return readPlanResponse(response, options.onPartial);
  });
}

export async function requestGuideFollow(payload: GuideFollowRequest, signal?: AbortSignal): Promise<GuideFollowResponse> {
  return tapGuideCall('/api/guide/follow', payload, async () =>
    readJsonResponse<GuideFollowResponse>(await post('/api/guide/follow', payload, signal))
  );
}

export async function requestGuideConfirm(
  payload: GuideConfirmRequest,
  signal?: AbortSignal
): Promise<GuideConfirmResponse> {
  return tapGuideCall('/api/guide/confirm', payload, async () =>
    readJsonResponse<GuideConfirmResponse>(await post('/api/guide/confirm', payload, signal))
  );
}

export async function requestGuideTalk(payload: GuideTalkRequest, signal?: AbortSignal): Promise<GuideTalkResponse> {
  return tapGuideCall('/api/guide/talk', payload, async () =>
    readJsonResponse<GuideTalkResponse>(await post('/api/guide/talk', payload, signal))
  );
}

/** `/api/health` read for the guide fields (the server answers GuideHealthResponse). */
export async function getGuideHealth(signal?: AbortSignal): Promise<GuideHealthResponse> {
  const response = await fetch('/api/health', { method: 'GET', credentials: 'same-origin', signal });
  return readJsonResponse<GuideHealthResponse>(response);
}
