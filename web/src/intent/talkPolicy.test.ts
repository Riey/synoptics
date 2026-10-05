/**
 * Talk gate (one in flight, no queue; shared paid-call budget, spacing and confirm floor), and
 * which 409 stale_plan answers a talk or replan has superseded. Loaded through Vite (runtime sibling import).
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
const { decideTalk, isSupersededStale } =
  await server.ssrLoadModule('/src/intent/talkPolicy.ts');
const { INITIAL_GATE_STATE, markDispatched, markTalk } = await server.ssrLoadModule('/src/intent/triggers.ts');

const CONFIG = {
  followLocalFloorMs: 1000,
  followRemoteFloorMs: 2200,
  confirmFloorMs: 2200,
  remoteSpacingMs: 1200,
  remoteBudget: 30,
  remoteWindowMs: 120_000,
  remotePerMinute: 20,
};
const input = (overrides: Record<string, unknown> = {}) => ({
  running: true,
  talkPending: false,
  gate: INITIAL_GATE_STATE,
  nowMs: 100_000,
  config: CONFIG,
  ...overrides,
});

test('a fresh run sends at once; not running or one already waiting refuses', () => {
  assert.deepEqual(decideTalk(input()), { block: null, waitMs: 0 });
  assert.deepEqual(decideTalk(input({ running: false })), { block: 'not_running', waitMs: 0 });
  assert.deepEqual(decideTalk(input({ talkPending: true })), { block: 'pending', waitMs: 0 });
});

test('the confirm floor and the paid-call spacing delay a talk; they never refuse it', () => {
  const afterConfirm = markDispatched(INITIAL_GATE_STATE, { kind: 'confirm', trigger: 'goal_check', lane: 'remote' }, 99_000, CONFIG);
  assert.deepEqual(decideTalk(input({ gate: afterConfirm })), { block: null, waitMs: 1_200 });
  const afterFollow = markDispatched(INITIAL_GATE_STATE, { kind: 'follow', trigger: 'heartbeat', lane: 'remote' }, 99_500, CONFIG);
  assert.deepEqual(decideTalk(input({ gate: afterFollow })), { block: null, waitMs: 700 });
  const afterTalk = markTalk(INITIAL_GATE_STATE, 99_000, CONFIG);
  assert.deepEqual(decideTalk(input({ gate: afterTalk })), { block: null, waitMs: 1_200 });
});

test('a spent run budget or per-minute ceiling refuses the talk (shown, not queued)', () => {
  let gate = INITIAL_GATE_STATE;
  for (let i = 0; i < 30; i += 1) gate = markTalk(gate, 1_000 + i * 3_000, CONFIG);
  assert.equal(decideTalk(input({ gate, nowMs: 92_000 })).block, 'budget');
  let minute = INITIAL_GATE_STATE;
  for (let i = 0; i < 20; i += 1) minute = markTalk(minute, 50_000 + i * 2_500, CONFIG);
  assert.equal(decideTalk(input({ gate: minute, nowMs: 100_000 })).block, 'per_minute');
});

test('a 409 for the plan the client is on, at its revision or one a talk is installing, is superseded', () => {
  const run = { planId: 'plan-1', planRevision: 5, talkPending: false };
  assert.equal(isSupersededStale({ plan_id: 'plan-1', plan_revision: 5 }, run), true);
  assert.equal(isSupersededStale({ plan_id: 'plan-1', plan_revision: 6 }, run), false);
  assert.equal(isSupersededStale({ plan_id: 'plan-1', plan_revision: 6 }, { ...run, talkPending: true }), true);
  assert.equal(isSupersededStale({ plan_id: 'plan-1', plan_revision: 4 }, { ...run, talkPending: true }), false);
  assert.equal(isSupersededStale({ plan_id: 'plan-2', plan_revision: 5 }, run), false);
  assert.equal(isSupersededStale({ plan_id: null, plan_revision: null }, run), false);
  assert.equal(isSupersededStale({}, { planId: null, planRevision: null, talkPending: false }), false);
});

test('review #1: an utterance is sent about the step on screen when the user pressed send', async () => {
  const { talkSendStep } = await server.ssrLoadModule('/src/intent/talkPolicy.ts');
  const steps = [
    { id: 's1', done_when: '안경다리를 잡음' },
    { id: 's2', done_when: '안경이 얼굴에서 떨어짐' },
  ];
  const accepted = { planId: 'plan-1', stepId: 's1', doneWhen: '안경다리를 잡음' };
  // A local follow moved the screen to s2 during the floor wait: the talk still names s1.
  assert.equal(talkSendStep(accepted, { planId: 'plan-1', steps }), 's1');
  // A step_say rewrite keeps the step (same id and done_when): still s1.
  assert.equal(talkSendStep(accepted, { planId: 'plan-1', steps: [{ ...steps[0] }, steps[1]] }), 's1');
  // A replan reused the id for another step, or a new plan started: nothing is sent.
  assert.equal(talkSendStep(accepted, { planId: 'plan-1', steps: [{ id: 's1', done_when: '컵이 손에 있음' }] }), null);
  assert.equal(talkSendStep(accepted, { planId: 'plan-2', steps }), null);
});
