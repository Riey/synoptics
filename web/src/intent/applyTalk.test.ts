/**
 * The talk reducer (pure): what an accepted talk answer does to the run, and what it asks the hook to do.
 *
 * Loaded through Vite like `planStream.test.ts`, because the module imports its siblings the app's way.
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
const { applyTalk, talkAction, talkChangesPlan, TALK_REPLAN_NOTICE } = await server.ssrLoadModule('/src/intent/applyTalk.ts');

const step = (id: string, say: string) => ({ id, say, commands: [], done_when: `${id} 결과가 보임` });
const STEPS = [step('s1', '안경다리를 잡으세요'), step('s2', '안경을 들어 올리세요'), step('s3', '안경을 내려놓으세요')];
const STATE = {
  steps: STEPS,
  stepIndex: 1,
  planRevision: 4,
  target: 'eyeglasses',
  skipped: [] as string[],
  userDone: ['s1'],
  unsureStreak: 1,
  stuckCount: 2,
  lastChecks: [{ step_id: 's2', visible: 'unsure' }],
  replanBudget: { used: 1, reserved: 1, lastPlanAt: 0 },
};
const answer = (fields: Record<string, unknown> = {}) => ({ reply: '네.', plan_revision: 4, ...fields });
const RESET = { unsureStreak: 0, stuckCount: 0, lastChecks: null };

test('a reply without an action changes nothing and is applied', () => {
  const outcome = applyTalk(STATE, 's2', answer(), 'bound', 1_000);
  assert.equal(outcome.applied, true);
  assert.equal(outcome.action, 'none');
  assert.equal(outcome.state, STATE);
  assert.deepEqual(outcome.effects, []);
});

test('a rejected answer is discarded whatever its action; text_only applies like bound', () => {
  for (const fields of [{ step_say: '더 쉽게' }, { step_mark: 'done' }, { go_to: 's1' }, { target: 'glasses' }]) {
    const rejected = applyTalk(STATE, 's2', answer(fields), 'rejected', 1_000);
    assert.equal(rejected.applied, false);
    assert.equal(rejected.state, STATE);
    assert.deepEqual(rejected.effects, []);
  }
  const textOnly = applyTalk(STATE, 's2', answer({ step_mark: 'done' }), 'text_only', 1_000);
  assert.equal(textOnly.applied, true);
  assert.equal(textOnly.state.stepIndex, 2);
});

test('step_say replaces the sentence of the step the user talked about; the step stays', () => {
  const outcome = applyTalk(STATE, 's2', answer({ step_say: '안경을 두 손으로 천천히 드세요', plan_revision: 5 }), 'bound', 1_000);
  assert.deepEqual(outcome.state.steps.map((s: { say: string }) => s.say), ['안경다리를 잡으세요', '안경을 두 손으로 천천히 드세요', '안경을 내려놓으세요']);
  assert.equal(outcome.state.stepIndex, 1);
  assert.equal(outcome.state.planRevision, 5);
  assert.equal(outcome.state.unsureStreak, 1, 'no step change, nothing reset');
  // The server rewrote the step named in the request even if the screen moved since.
  const moved = applyTalk({ ...STATE, stepIndex: 2 }, 's2', answer({ step_say: '새 문장', plan_revision: 5 }), 'bound', 1_000);
  assert.equal(moved.state.steps[1].say, '새 문장');
  assert.equal(moved.state.stepIndex, 2);
});

test('step_mark done: user-confirmed, next step, counters and the last checklist reset', () => {
  const outcome = applyTalk(STATE, 's2', answer({ step_mark: 'done' }), 'bound', 1_000);
  assert.deepEqual(outcome.state.userDone, ['s1', 's2']);
  assert.equal(outcome.state.stepIndex, 2);
  assert.deepEqual(
    { unsureStreak: outcome.state.unsureStreak, stuckCount: outcome.state.stuckCount, lastChecks: outcome.state.lastChecks },
    RESET
  );
  assert.deepEqual(outcome.effects, []);
});

test('step_mark done on the last step stays and asks confirm(goal_check)', () => {
  const last = { ...STATE, stepIndex: 2 };
  const outcome = applyTalk(last, 's3', answer({ step_mark: 'done' }), 'bound', 1_000);
  assert.equal(outcome.state.stepIndex, 2);
  assert.deepEqual(outcome.state.userDone, ['s1', 's3']);
  assert.deepEqual(outcome.effects, [{ kind: 'checkGoal' }]);
});

test('step_mark skipped: shown as skipped (never done), next step; on the last step it stays', () => {
  const outcome = applyTalk(STATE, 's2', answer({ step_mark: 'skipped' }), 'bound', 1_000);
  assert.deepEqual(outcome.state.skipped, ['s2']);
  assert.deepEqual(outcome.state.userDone, ['s1']);
  assert.equal(outcome.state.stepIndex, 2);
  const last = applyTalk({ ...STATE, stepIndex: 2 }, 's3', answer({ step_mark: 'skipped' }), 'bound', 1_000);
  assert.equal(last.state.stepIndex, 2);
  assert.deepEqual(last.effects, []);
});

test('step_mark after the screen moved (a follow advanced meanwhile): done is recorded, nothing moves', () => {
  const moved = { ...STATE, stepIndex: 2 };
  const done = applyTalk(moved, 's2', answer({ step_mark: 'done' }), 'bound', 1_000);
  assert.equal(done.applied, true);
  assert.equal(done.state.stepIndex, 2);
  assert.deepEqual(done.state.userDone, ['s1', 's2']);
  assert.equal(done.state.unsureStreak, moved.unsureStreak, 'no step change, nothing reset');
  // A skip of a step the screen already left changes nothing.
  assert.equal(applyTalk(moved, 's2', answer({ step_mark: 'skipped' }), 'bound', 1_000).state, moved);
});

test('review #3: a late user "done" for a passed step blocks the step_done revert', async () => {
  const { decideConfirm } = await server.ssrLoadModule('/src/intent/confirmDecision.ts');
  // Talk sent at s1; a follow advanced s1 -> s2 and asked confirm(step_done, s1); the talk answer lands first.
  const atS2 = { ...STATE, stepIndex: 1, userDone: [] };
  const outcome = applyTalk(atS2, 's1', answer({ step_mark: 'done' }), 'bound', 1_000);
  assert.deepEqual(outcome.state.userDone, ['s1']);
  assert.equal(outcome.state.stepIndex, 1);
  const verdict = decideConfirm(
    { trigger: 'step_done', goal_status: { status: 'in_progress' }, step_check: 'no', step_id: 's1' },
    { fromIndex: 0, toIndex: 1, stepId: 's1' },
    { stepIndex: 1, completion: 'none', trackLost: false, userDoneIds: outcome.state.userDone }
  );
  assert.deepEqual(verdict.step, { kind: 'none' });
});

test('go_to returns to an earlier step and forgets user marks and skips at or after it', () => {
  const from = { ...STATE, stepIndex: 2, skipped: ['s2'], userDone: ['s1'] };
  const toFirst = applyTalk(from, 's3', answer({ go_to: 's1' }), 'bound', 1_000);
  assert.equal(toFirst.state.stepIndex, 0);
  assert.deepEqual(toFirst.state.skipped, []);
  assert.deepEqual(toFirst.state.userDone, []);
  assert.equal(toFirst.state.lastChecks, null);
  const toSecond = applyTalk(from, 's3', answer({ go_to: 's2' }), 'bound', 1_000);
  assert.equal(toSecond.state.stepIndex, 1);
  assert.deepEqual(toSecond.state.userDone, ['s1']);
  assert.deepEqual(toSecond.state.skipped, []);
  // Not before the current step (the screen already went back further): nothing changes.
  const behind = { ...STATE, stepIndex: 0, userDone: [] };
  assert.equal(applyTalk(behind, 's3', answer({ go_to: 's2' }), 'bound', 1_000).state, behind);
});

test('target asks the hook to restart tracking with the new noun; the step stays', () => {
  const outcome = applyTalk(STATE, 's2', answer({ target: 'glasses frame' }), 'bound', 1_000);
  assert.equal(outcome.state.target, 'glasses frame');
  assert.equal(outcome.state.stepIndex, 1);
  assert.deepEqual(outcome.effects, [{ kind: 'retarget', target: 'glasses frame' }]);
});

test('replan installs the new steps from step 1, spends a replan, and restarts the replan gap', () => {
  const steps = [step('s1', '안경을 접으세요')];
  const outcome = applyTalk({ ...STATE, skipped: ['s1'] }, 's2', answer({ replan: { steps }, plan_revision: 5 }), 'bound', 10_000);
  assert.deepEqual(outcome.state.steps, steps);
  assert.equal(outcome.state.stepIndex, 0);
  assert.equal(outcome.state.planRevision, 5);
  assert.deepEqual(outcome.state.skipped, []);
  assert.deepEqual(outcome.state.userDone, []);
  // The talk's reservation is settled by the hook once the call ends; the reducer only counts the replan.
  assert.deepEqual(outcome.state.replanBudget, { used: 2, reserved: 1, lastPlanAt: 10_000 });
  assert.deepEqual(outcome.effects, [{ kind: 'notice', text: TALK_REPLAN_NOTICE }]);
});

test('only step_say and replan change the server plan', () => {
  assert.equal(talkAction(answer()), 'none');
  assert.equal(talkAction(answer({ go_to: 's1' })), 'go_to');
  assert.equal(talkChangesPlan(answer({ step_say: '새 문장' })), true);
  assert.equal(talkChangesPlan(answer({ replan: { steps: [] } })), true);
  for (const fields of [{}, { target: 'cup' }, { step_mark: 'done' }, { go_to: 's1' }]) {
    assert.equal(talkChangesPlan(answer(fields)), false);
  }
});

test('a server-normalised "already done" (user_says_done) advances like step_mark done', () => {
  const outcome = applyTalk(STATE, 's2', answer({ user_says_done: true, step_mark: 'done' }), 'bound', 1_000);
  assert.equal(outcome.action, 'step_mark');
  assert.equal(outcome.state.stepIndex, 2);
  assert.deepEqual(outcome.state.userDone, ['s1', 's2']);
});

test('glove case: step_say and target in one answer — the step is rewritten, then tracking moves', () => {
  const say = '장갑 낀 손으로 뚜껑을 반시계 방향으로 돌리세요';
  const outcome = applyTalk(
    STATE,
    's2',
    answer({ step_say: say, target: 'hand wearing rubber glove', plan_revision: 5 }),
    'bound',
    1_000
  );
  assert.deepEqual(outcome.actions, ['target', 'step_say']);
  assert.equal(outcome.state.steps[1].say, say);
  assert.equal(outcome.state.planRevision, 5);
  assert.equal(outcome.state.target, 'hand wearing rubber glove');
  assert.equal(outcome.state.stepIndex, 1);
  assert.deepEqual(outcome.effects, [{ kind: 'retarget', target: 'hand wearing rubber glove' }]);
});

test('replan with target: the new steps are installed, then tracking moves (notice first, retarget second)', () => {
  const steps = [step('s1', '장갑을 끼세요'), step('s2', '장갑 낀 손으로 뚜껑을 돌리세요')];
  const outcome = applyTalk(STATE, 's2', answer({ replan: { steps }, target: 'hand wearing rubber glove', plan_revision: 5 }), 'bound', 9_000);
  assert.deepEqual(outcome.actions, ['replan', 'target']);
  assert.deepEqual(outcome.state.steps, steps);
  assert.equal(outcome.state.stepIndex, 0);
  assert.deepEqual(outcome.effects, [
    { kind: 'notice', text: TALK_REPLAN_NOTICE },
    { kind: 'retarget', target: 'hand wearing rubber glove' },
  ]);
});

test('step_mark with step_say: the sentence is rewritten for the step the user talked about, then it is marked', () => {
  const outcome = applyTalk(STATE, 's2', answer({ step_mark: 'done', step_say: '안경을 들어 올리세요', plan_revision: 5 }), 'bound', 1_000);
  assert.deepEqual(outcome.actions, ['step_mark', 'step_say']);
  assert.equal(outcome.state.steps[1].say, '안경을 들어 올리세요');
  assert.deepEqual(outcome.state.userDone, ['s1', 's2']);
  assert.equal(outcome.state.stepIndex, 2);
  assert.equal(talkChangesPlan(answer({ step_mark: 'done', step_say: '새 문장' })), true);
});
