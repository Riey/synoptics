/**
 * Stuck detector (three accepted target_changed follows without a step change → replan) and the replan
 * budget (at most `REPLAN_MAX_PER_RUN` (10) per run, never within `REPLAN_MIN_GAP_MS` (8 s) of the last plan or replan).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  REPLAN_MAX_PER_RUN,
  REPLAN_MIN_GAP_MS,
  STUCK_TARGET_CHANGED_FOLLOWS,
  describeReplanBlock,
  freshReplanBudget,
  nextStuckCount,
  notePlanInstalled,
  replanBlock,
  spendReplan,
  releaseReplan,
  replanDispatch,
  reserveReplan,
} from './replanPolicy.ts';

test('three accepted target_changed follows with no step change → stuck, and the count starts over', () => {
  assert.equal(STUCK_TARGET_CHANGED_FOLLOWS, 3);
  let count = 0;
  const fired: boolean[] = [];
  for (let i = 0; i < 4; i += 1) {
    const next = nextStuckCount(count, { trigger: 'target_changed', stepChanged: false });
    count = next.count;
    fired.push(next.stuck);
  }
  assert.deepEqual(fired, [false, false, true, false]);
  assert.equal(count, 1);
});

test('a step change resets the count', () => {
  let state = nextStuckCount(0, { trigger: 'target_changed', stepChanged: false });
  state = nextStuckCount(state.count, { trigger: 'target_changed', stepChanged: false });
  assert.equal(state.count, 2);
  state = nextStuckCount(state.count, { trigger: 'target_changed', stepChanged: true });
  assert.deepEqual(state, { count: 0, stuck: false });
});

test('heartbeat and manual follows neither count nor reset (a still user is not stuck)', () => {
  let count = 2;
  for (const trigger of ['heartbeat', 'manual', 'heartbeat']) {
    const next = nextStuckCount(count, { trigger, stepChanged: false });
    assert.equal(next.stuck, false);
    count = next.count;
  }
  assert.equal(count, 2);
  assert.equal(nextStuckCount(count, { trigger: 'target_changed', stepChanged: false }).stuck, true);
});

test('replan budget: blocked for 8 s after the plan, then at most ten per run', () => {
  assert.equal(REPLAN_MAX_PER_RUN, 10);
  assert.equal(REPLAN_MIN_GAP_MS, 8_000);
  let budget = freshReplanBudget(1_000);
  assert.deepEqual(replanBlock(budget, 1_000 + 7_999), { block: 'cooldown', waitMs: 1 });
  assert.deepEqual(replanBlock(budget, 9_000), { block: null, waitMs: 0 });

  let now = 9_000;
  for (let i = 0; i < REPLAN_MAX_PER_RUN; i += 1) {
    assert.equal(replanBlock(budget, now).block, null);
    budget = spendReplan(budget, now);
    assert.equal(replanBlock(budget, now + 1).block, i < REPLAN_MAX_PER_RUN - 1 ? 'cooldown' : 'max');
    now += REPLAN_MIN_GAP_MS;
  }
  assert.equal(budget.used, REPLAN_MAX_PER_RUN);
  assert.equal(replanBlock(budget, now + 60_000).block, 'max');
});

test('an installed replan restarts the 8 s gap without spending the budget', () => {
  let budget = spendReplan(freshReplanBudget(0), 20_000);
  budget = notePlanInstalled(budget, 22_000);
  assert.equal(budget.used, 1);
  assert.deepEqual(replanBlock(budget, 29_000), { block: 'cooldown', waitMs: 1_000 });
  assert.equal(replanBlock(budget, 30_000).block, null);
});

test('the block reason is a sentence for the disabled button; no block, no reason', () => {
  assert.equal(describeReplanBlock(null), null);
  assert.match(describeReplanBlock('max') ?? '', /최대 10회/);
  assert.match(describeReplanBlock('cooldown') ?? '', /8초/);
});

test('review #4: a replan-capable talk reserves the last slot, so nothing else can take it meanwhile', () => {
  // All but one replan used; the talk reserves the last.
  const last = REPLAN_MAX_PER_RUN - 1;
  const two = { used: last, reserved: 0, lastPlanAt: 0 };
  const talk = reserveReplan(two);
  assert.equal(talk.reserved, true);
  assert.deepEqual(talk.budget, { used: last, reserved: 1, lastPlanAt: 0 });
  assert.equal(replanBlock(talk.budget, 60_000).block, 'max', 'manual/stuck replans are blocked while it waits');
  assert.equal(reserveReplan(talk.budget).reserved, false, 'a second talk gets replan_allowed=false');
  // The answer carried no replan: the slot comes back.
  assert.deepEqual(releaseReplan(talk.budget), two);
  assert.equal(replanBlock(releaseReplan(talk.budget), 60_000).block, null);
  // The answer replanned: used reaches the cap (the reducer counts it), the reservation is released, never over.
  const replanned = releaseReplan({ ...talk.budget, used: REPLAN_MAX_PER_RUN });
  assert.deepEqual(replanned, { used: REPLAN_MAX_PER_RUN, reserved: 0, lastPlanAt: 0 });
  assert.equal(reserveReplan(replanned).reserved, false);
  // Spending keeps a reservation in place.
  assert.deepEqual(spendReplan({ used: 0, reserved: 1, lastPlanAt: 0 }, 5), { used: 1, reserved: 1, lastPlanAt: 5 });
  assert.equal(freshReplanBudget(7).reserved, 0);
});

test('review #4: a queued replan confirm is re-checked when it leaves', () => {
  const budget = { used: 2, reserved: 0, lastPlanAt: 0 };
  assert.equal(replanDispatch(budget, 1, 1), 'send');
  // A talk replan was installed after this confirm was asked: its question was about the old plan.
  assert.equal(replanDispatch(budget, 1, 2), 'drop');
  // Over the cap (cannot happen through the hook, refused anyway).
  assert.equal(replanDispatch({ used: REPLAN_MAX_PER_RUN, reserved: 1, lastPlanAt: 0 }, 1, 1), 'drop');
  // A confirm asked without an epoch (older call) is sent as before.
  assert.equal(replanDispatch(budget, undefined, 3), 'send');
});
