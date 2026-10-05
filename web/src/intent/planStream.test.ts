/**
 * Reading `/api/guide/plan` as SSE (partials, final, error, framing) and the plain-JSON fallback.
 *
 * Loaded through Vite like the policy test, because the module imports `ApiError` with the app's
 * extensionless specifiers.
 */
import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const WEB_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const server = await createServer({
  configFile: false,
  root: WEB_ROOT,
  logLevel: 'silent',
  server: { middlewareMode: true, hmr: false, watch: null },
});
after(async () => {
  await server.close();
});

const { createSseParser, readPlanResponse, readJsonResponse, IntentHttpError } = await server.ssrLoadModule(
  '/src/intent/planStream.ts'
);
const { ApiError } = await server.ssrLoadModule('/src/coach/api.ts');

const PLAN = {
  selection: { status: 'selected', target: 'glasses', rationale: '안경이 보입니다' },
  steps: [{ id: 's1', say: '안경을 잡으세요', commands: [{ kind: 'focus', anchor: 'target' }], done_when: '손이 안경을 잡음' }],
  goal_when: '안경이 얼굴에서 벗겨짐',
  evidence_kind: 'observed_scene',
  needs_clarification: false,
  plan_id: '00000000-0000-4000-8000-000000000001',
  plan_revision: 1,
  frame_id: 'guide-1',
  provider: 'deepseek',
  model: 'deepseek-flash',
  usage: null,
};

/** A Response whose body arrives as the given chunks (strings split anywhere). */
function streamed(chunks: string[], init: { status?: number; type?: string } = {}) {
  const encoder = new TextEncoder();
  let index = 0;
  const body = new ReadableStream({
    pull(controller) {
      if (index < chunks.length) controller.enqueue(encoder.encode(chunks[index++]));
      else controller.close();
    },
  });
  return new Response(body, {
    status: init.status ?? 200,
    headers: { 'content-type': init.type ?? 'text/event-stream; charset=utf-8' },
  });
}

const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

test('parser handles chunk splits anywhere, CRLF, comments and multi-line data', () => {
  const parser = createSseParser();
  const text = ': ping\r\nevent: partial\r\ndata: {"target":\r\ndata: "glasses"}\r\n\r\nevent: final\ndata: {}\n\n';
  const events = [];
  for (const ch of text) events.push(...parser.push(ch));
  assert.deepEqual(events, [
    { event: 'partial', data: '{"target":\n"glasses"}' },
    { event: 'final', data: '{}' },
  ]);
});

test('partials are reported in order before the final, and the final is the plan', async () => {
  const seen: unknown[] = [];
  const text =
    ': ping\n\n' +
    sse('partial', { target: 'glasses' }) +
    sse('partial', { first_say: '안경을 잡으세요' }) +
    sse('final', PLAN);
  const response = streamed([text.slice(0, 17), text.slice(17, 60), text.slice(60)]);
  const plan = await readPlanResponse(response, (partial: unknown) => seen.push(partial));
  assert.deepEqual(seen, [{ target: 'glasses' }, { firstSay: '안경을 잡으세요' }]);
  assert.deepEqual(plan, PLAN);
});

test('a malformed partial is ignored, not fatal', async () => {
  const seen: unknown[] = [];
  const response = streamed(['event: partial\ndata: {not json\n\n', sse('final', PLAN)]);
  const plan = await readPlanResponse(response, (partial: unknown) => seen.push(partial));
  assert.deepEqual(seen, []);
  assert.equal(plan.plan_id, PLAN.plan_id);
});

test('event: error rejects with the body, its status and code', async () => {
  const response = streamed([sse('partial', { target: 'glasses' }), sse('error', { error: 'invalid_provider_output', reason: 'truncated', status: 502 })]);
  await assert.rejects(readPlanResponse(response), (error: InstanceType<typeof ApiError>) => {
    assert.ok(error instanceof ApiError);
    assert.ok(error instanceof IntentHttpError);
    assert.equal(error.status, 502);
    assert.equal(error.code, 'invalid_provider_output');
    assert.equal(error.reason, 'truncated');
    return true;
  });
});

test('a stream without a terminal event, or with a malformed final, fails closed', async () => {
  await assert.rejects(readPlanResponse(streamed([sse('partial', { target: 'glasses' })])), (error: { status: number }) => {
    assert.equal(error.status, 502);
    return true;
  });
  await assert.rejects(readPlanResponse(streamed(['event: final\ndata: {"plan_id": 3}\n\n'])), (error: { status: number }) => {
    assert.equal(error.status, 502);
    return true;
  });
});

test('a non-stream JSON answer is read like the plain route (fallback)', async () => {
  const response = new Response(JSON.stringify(PLAN), { status: 200, headers: { 'content-type': 'application/json' } });
  const seen: unknown[] = [];
  assert.deepEqual(await readPlanResponse(response, (partial: unknown) => seen.push(partial)), PLAN);
  assert.deepEqual(seen, []);
});

test('a non-stream error keeps status, code, Retry-After and the body (stale_plan)', async () => {
  const busy = new Response(JSON.stringify({ error: 'provider_busy' }), {
    status: 503,
    headers: { 'content-type': 'application/json', 'retry-after': '1' },
  });
  await assert.rejects(readPlanResponse(busy), (error: { status: number; code: string; retryAfterMs: number }) => {
    assert.equal(error.status, 503);
    assert.equal(error.code, 'provider_busy');
    assert.equal(error.retryAfterMs, 1000);
    return true;
  });
  const stale = new Response(JSON.stringify({ error: 'stale_plan', plan_id: 'p', plan_revision: 4 }), {
    status: 409,
    headers: { 'content-type': 'application/json' },
  });
  await assert.rejects(readJsonResponse(stale), (error: { code: string; body: Record<string, unknown> }) => {
    assert.equal(error.code, 'stale_plan');
    assert.equal(error.body.plan_revision, 4);
    return true;
  });
});
