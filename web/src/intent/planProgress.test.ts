/**
 * PlanCard's five step states (완료 ✓ / 사용자 확인 ✓ / 건너뜀 / 현재 / 대기) and the skipped-step bookkeeping
 * behind them.
 * `PlanCard.tsx` renders exactly `planStepStates` with `PLAN_STEP_STATE_TEXT`.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { PLAN_STEP_STATE_TEXT, addSkipped, planStepStates, skippedBefore } from './planProgress.ts';

const STEPS = [{ id: 's1' }, { id: 's2' }, { id: 's3' }, { id: 's4' }];

test('done / skipped before the current step, current, then pending', () => {
  assert.deepEqual(planStepStates(STEPS, 2, ['s1']), ['skipped', 'done', 'current', 'pending']);
  assert.deepEqual(planStepStates(STEPS, 0, []), ['current', 'pending', 'pending', 'pending']);
  assert.deepEqual(planStepStates(STEPS, 3, ['s2', 's3']), ['done', 'skipped', 'skipped', 'current']);
});

test('a skipped id at or after the current step is never shown as skipped', () => {
  assert.deepEqual(planStepStates(STEPS, 1, ['s2', 's3']), ['done', 'current', 'pending', 'pending']);
});

test('the labels are words, with the check mark only on done and user-confirmed', () => {
  assert.deepEqual(PLAN_STEP_STATE_TEXT, {
    done: '완료 ✓',
    user_done: '사용자 확인 ✓',
    skipped: '건너뜀',
    current: '현재',
    pending: '대기',
  });
});

test('skipped bookkeeping: add without duplicates; a revert forgets skips at or after the step it returns to', () => {
  assert.deepEqual(addSkipped(['s1'], ['s1', 's2']), ['s1', 's2']);
  assert.deepEqual(skippedBefore(STEPS, ['s1', 's2', 's3'], 1), ['s1']);
  assert.deepEqual(skippedBefore(STEPS, ['s9'], 3), []);
});

test('a step the user said was done is shown as user-confirmed, never as done or skipped', () => {
  assert.deepEqual(planStepStates(STEPS, 3, ['s2'], ['s1', 's3']), ['user_done', 'skipped', 'user_done', 'current']);
  // At or after the current step a user mark is not shown.
  assert.deepEqual(planStepStates(STEPS, 1, [], ['s2', 's3']), ['done', 'current', 'pending', 'pending']);
});
