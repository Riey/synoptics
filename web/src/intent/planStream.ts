/**
 * Reads the answer of `POST /api/guide/plan`, streamed or not.
 *
 * SSE contract (shared with the backend): with `Accept: text/event-stream` the server answers `200
 * text/event-stream` and emits, in order, at most one `event: partial` `{"target"}`, at most one
 * `event: partial` `{"first_say"}`, then exactly one terminal event — `event: final` whose data is the full
 * validated `GuidePlanResponse` (identical to the JSON body), or `event: error` whose data is the JSON error
 * body the plain route would return plus `"status"`. `: ping` comment lines may arrive at any time.
 *
 * Partials are display-only hints: the caller shows them at once and replaces them with the final. A
 * response that is not `text/event-stream` is read as the plain JSON route (fallback). A stream that ends
 * without a terminal event, or whose final cannot be parsed, is an error — never a partial plan.
 */
import type { GuidePlanResponse } from '../generated/api.generated';
import { ApiError } from '../coach/api';

export interface SseEvent {
  event: string;
  data: string;
}

/** An HTTP or stream failure carrying the server's JSON error body (e.g. `stale_plan`'s plan identity). */
export class IntentHttpError extends ApiError {
  readonly body: Record<string, unknown> | null;

  constructor(
    status: number,
    code: string,
    retryAfterMs: number | null = null,
    reason: string | null = null,
    body: Record<string, unknown> | null = null
  ) {
    super(status, code, retryAfterMs, reason);
    this.name = 'IntentHttpError';
    this.body = body;
  }
}

/**
 * Incremental parser for the `text/event-stream` wire format. `push` takes any slice of decoded text (an
 * event may be split across chunks at any byte) and returns the events completed by it.
 */
export function createSseParser(): { push(text: string): SseEvent[]; flush(): SseEvent[] } {
  let buffer = '';
  let eventName = '';
  let dataLines: string[] = [];

  const takeLine = (line: string, out: SseEvent[]) => {
    if (line === '') {
      if (dataLines.length > 0) out.push({ event: eventName || 'message', data: dataLines.join('\n') });
      eventName = '';
      dataLines = [];
      return;
    }
    if (line.startsWith(':')) return; // comment / heartbeat
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') eventName = value;
    else if (field === 'data') dataLines.push(value);
    // `id`, `retry` and unknown fields are ignored.
  };

  return {
    push(text: string): SseEvent[] {
      buffer += text;
      const out: SseEvent[] = [];
      for (;;) {
        const match = /\r\n|\r|\n/.exec(buffer);
        if (!match) break;
        // A lone '\r' at the very end may be the first half of '\r\n': wait for the next chunk.
        if (match[0] === '\r' && match.index === buffer.length - 1) break;
        takeLine(buffer.slice(0, match.index), out);
        buffer = buffer.slice(match.index + match[0].length);
      }
      return out;
    },
    flush(): SseEvent[] {
      const out: SseEvent[] = [];
      if (buffer !== '') {
        takeLine(buffer, out);
        buffer = '';
      }
      takeLine('', out);
      return out;
    },
  };
}

export interface PlanPartial {
  target?: string;
  firstSay?: string;
}

function parseRetryAfter(header: string | null): number | null {
  if (!header) return null;
  const seconds = Number(header);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds * 1000 : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

/** The error a JSON error body (plain route) or an `event: error` payload (stream) stands for. */
export function errorFromBody(status: number, body: unknown, retryAfterMs: number | null = null): IntentHttpError {
  const record = asRecord(body);
  const code = typeof record?.error === 'string' ? record.error : 'request_failed';
  const reason = typeof record?.reason === 'string' ? record.reason : null;
  return new IntentHttpError(status, code, retryAfterMs, reason, record);
}

/** Minimal shape check of a final plan: the fields the client acts on must exist with the right types. */
export function isPlanResponse(value: unknown): value is GuidePlanResponse {
  const record = asRecord(value);
  return (
    record !== null &&
    typeof record.plan_id === 'string' &&
    typeof record.plan_revision === 'number' &&
    Array.isArray(record.steps) &&
    asRecord(record.selection) !== null &&
    typeof record.needs_clarification === 'boolean'
  );
}

/** Read a plain JSON answer: the body on 2xx, otherwise the `IntentHttpError` its error body stands for. */
export async function readJsonResponse<T>(response: Response): Promise<T> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) throw errorFromBody(response.status, body, parseRetryAfter(response.headers.get('retry-after')));
  if (body === null) throw new IntentHttpError(502, 'invalid_response');
  return body as T;
}

async function readJsonPlan(response: Response): Promise<GuidePlanResponse> {
  const body = await readJsonResponse<unknown>(response);
  if (!isPlanResponse(body)) throw new IntentHttpError(502, 'invalid_plan_response');
  return body;
}

export function isEventStream(response: Response): boolean {
  const type = response.headers.get('content-type') ?? '';
  return type.split(';', 1)[0].trim().toLowerCase() === 'text/event-stream';
}

/**
 * Read a plan response. Partials are reported through `onPartial` (each kind at most once, never after the
 * terminal event); the promise resolves with the final plan or rejects with an `IntentHttpError`.
 */
export async function readPlanResponse(
  response: Response,
  onPartial?: (partial: PlanPartial) => void
): Promise<GuidePlanResponse> {
  if (!isEventStream(response) || !response.body) return readJsonPlan(response);

  const parser = createSseParser();
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let sawTarget = false;
  let sawSay = false;

  /** Returns the terminal outcome when `event` is terminal, otherwise null. */
  const handle = (event: SseEvent): { plan: GuidePlanResponse } | { error: IntentHttpError } | null => {
    if (event.event === 'partial') {
      let data: Record<string, unknown> | null = null;
      try {
        data = asRecord(JSON.parse(event.data));
      } catch {
        return null; // a malformed hint is only a missing hint
      }
      if (!data) return null;
      if (!sawTarget && typeof data.target === 'string' && data.target.trim()) {
        sawTarget = true;
        onPartial?.({ target: data.target });
      }
      if (!sawSay && typeof data.first_say === 'string' && data.first_say.trim()) {
        sawSay = true;
        onPartial?.({ firstSay: data.first_say });
      }
      return null;
    }
    if (event.event === 'final') {
      let data: unknown = null;
      try {
        data = JSON.parse(event.data);
      } catch {
        return { error: new IntentHttpError(502, 'stream_malformed') };
      }
      return isPlanResponse(data) ? { plan: data } : { error: new IntentHttpError(502, 'invalid_plan_response') };
    }
    if (event.event === 'error') {
      let data: unknown = null;
      try {
        data = JSON.parse(event.data);
      } catch {
        return { error: new IntentHttpError(502, 'stream_malformed') };
      }
      const record = asRecord(data);
      const status = typeof record?.status === 'number' && record.status >= 400 ? record.status : 502;
      return { error: errorFromBody(status, record) };
    }
    return null; // unknown event names are ignored
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      const events = done ? [...parser.push(decoder.decode()), ...parser.flush()] : parser.push(decoder.decode(value, { stream: true }));
      for (const event of events) {
        const outcome = handle(event);
        if (outcome) {
          void reader.cancel().catch(() => {});
          if ('plan' in outcome) return outcome.plan;
          throw outcome.error;
        }
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock?.();
  }
  throw new IntentHttpError(502, 'stream_incomplete');
}
