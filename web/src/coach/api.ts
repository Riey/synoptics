/**
 * Thin transport for the session and health routes, plus the shared error type and body cap.
 *
 * This module knows only the application HTTP contract (generated into
 * `web/src/generated/api.generated.ts`); the visual contract lives in `@visual-coach/visual-tools`.
 */

import type {
  EndSessionResponse,
  HealthResponse,
  SessionRequest,
  SessionResponse,
} from '../generated/api.generated';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  /**
   * The server's closed-set stage token for a provider refusal, or null when it sent none.
   *
   * It is a member of a set the server owns (`invalid_provider_output` stages such as `truncated`,
   * `schema`, `provenance`, `context`; `provider_unavailable` stages `transport` and
   * `upstream_status`), never prose: it says which stage refused the answer, which the status alone
   * cannot (every provider refusal is a 502).
   */
  readonly reason: string | null;
  readonly retryAfterMs: number | null;

  constructor(status: number, code: string, retryAfterMs: number | null = null, reason: string | null = null) {
    super(`HTTP ${status}: ${code}`);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.retryAfterMs = retryAfterMs;
    this.reason = reason;
  }
}

// Server-enforced request body ceiling (mirrors backend MAX_GUIDANCE_BODY).
export const GUIDANCE_MAX_BODY_BYTES = 10 * 1024 * 1024;

function parseRetryAfter(header: string | null): number | null {
  if (!header) return null;
  const seconds = Number(header);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds * 1000 : null;
}

async function performRequest<T>(
  path: string,
  method: 'GET' | 'POST',
  serializedBody: string | undefined,
  signal?: AbortSignal
): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: serializedBody === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: serializedBody,
    credentials: 'same-origin',
    signal,
  });

  if (!response.ok) {
    let code = 'request_failed';
    let reason: string | null = null;
    try {
      const payload: unknown = await response.json();
      if (payload && typeof payload === 'object' && 'error' in payload && typeof payload.error === 'string') {
        code = payload.error;
      }
      // Additive: only a provider refusal carries a stage, and an older server simply omits it.
      if (payload && typeof payload === 'object' && 'reason' in payload && typeof payload.reason === 'string') {
        reason = payload.reason;
      }
    } catch {
      // keep the fallback code
    }
    throw new ApiError(response.status, code, parseRetryAfter(response.headers.get('retry-after')), reason);
  }

  return (await response.json()) as T;
}

function serializeBody(payload: unknown, maxBytes: number): string {
  const serialized = JSON.stringify(payload);
  const byteLength = new TextEncoder().encode(serialized).length;
  if (byteLength > maxBytes) {
    throw new ApiError(413, 'body_too_large');
  }
  return serialized;
}

export async function getHealth(): Promise<HealthResponse> {
  return performRequest<HealthResponse>('/api/health', 'GET', undefined);
}

/**
 * A session round trip is a small local POST. Bounding it keeps one stalled request from holding the
 * app's serialized session chain — and a queued revocation end — open indefinitely.
 */
const SESSION_TIMEOUT_MS = 15_000;

async function boundedSessionRequest<T>(run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const timeout = AbortSignal.timeout(SESSION_TIMEOUT_MS);
  try {
    return await run(timeout);
  } catch (error) {
    // A timeout is a server verdict, not an opaque transport rejection: name it as such.
    if (timeout.aborted) throw new ApiError(504, 'session_timeout');
    throw error;
  }
}

/**
 * Opens a session. `accessCode` is `null` for a deployment whose health reported
 * `access_code_required: false` (explicit local-only or open-access mode), and the field is then omitted from the
 * body entirely; a code-protected deployment answers 401 for that.
 */
export async function createSession(accessCode: string | null, signal?: AbortSignal): Promise<SessionResponse> {
  const body: SessionRequest = accessCode ? { access_code: accessCode } : {};
  const serialized = serializeBody(body, 64 * 1024);
  return boundedSessionRequest((timeout) =>
    performRequest<SessionResponse>(
      '/api/session',
      'POST',
      serialized,
      signal ? AbortSignal.any([signal, timeout]) : timeout
    )
  );
}

export async function endSession(signal?: AbortSignal): Promise<EndSessionResponse> {
  return boundedSessionRequest((timeout) =>
    performRequest<EndSessionResponse>(
      '/api/session/end',
      'POST',
      undefined,
      signal ? AbortSignal.any([signal, timeout]) : timeout
    )
  );
}
