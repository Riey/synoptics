/**
 * Follow checklist → step decision (pure): adjacent advance at once, a skip only when two consecutive
 * accepted follows show it, never on `unsure`, never after the screen moved, and the last step stays put.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { currentStepVisible, decideFollowStep, type StepCheck, type Visible } from './followStep.ts';

const STEPS = [{ id: 's1' }, { id: 's2' }, { id: 's3' }, { id: 's4' }];

/** Checks from `from` (index) to the last step, in plan order. */
function checks(from: number, ...visible: Visible[]): StepCheck[] {
  return visible.map((v, offset) => ({ step_id: STEPS[from + offset].id, visible: v }));
}

test('adjacent advance: the asked step yes moves to the next step at once (one follow is enough)', () => {
  assert.deepEqual(decideFollowStep(STEPS, 0, 0, checks(0, 'yes', 'no', 'no', 'no'), null), {
    kind: 'advance',
    fromIndex: 0,
    toIndex: 1,
    stepId: 's1',
  });
  assert.deepEqual(decideFollowStep(STEPS, 1, 1, checks(1, 'yes', 'unsure', 'no'), null), {
    kind: 'advance',
    fromIndex: 1,
    toIndex: 2,
    stepId: 's2',
  });
});

test('skip, first sighting: recorded only (pendingSkip), the screen does not move', () => {
  assert.deepEqual(decideFollowStep(STEPS, 0, 0, checks(0, 'no', 'no', 'yes', 'no'), null), {
    kind: 'pendingSkip',
    lastYesIndex: 2,
  });
  // The previous follow saw nothing that late: still a first sighting.
  assert.deepEqual(
    decideFollowStep(STEPS, 0, 0, checks(0, 'no', 'no', 'yes', 'no'), checks(0, 'no', 'yes', 'no', 'no')),
    { kind: 'pendingSkip', lastYesIndex: 2 }
  );
});

test('skip, second consecutive sighting: on to the step after the last yes; non-yes steps are skipped, not done', () => {
  const previous = checks(0, 'no', 'no', 'yes', 'no');
  assert.deepEqual(decideFollowStep(STEPS, 0, 0, checks(0, 'unsure', 'yes', 'yes', 'no'), previous), {
    kind: 'skip',
    fromIndex: 0,
    toIndex: 3,
    stepId: 's3',
    skippedIds: ['s1'],
  });
  // The previous follow saw an even later step: that also confirms this skip.
  assert.deepEqual(decideFollowStep(STEPS, 0, 0, checks(0, 'no', 'no', 'yes', 'no'), checks(0, 'no', 'no', 'no', 'yes')), {
    kind: 'skip',
    fromIndex: 0,
    toIndex: 3,
    stepId: 's3',
    skippedIds: ['s1', 's2'],
  });
});

test('a skip to the last step stays on the last step', () => {
  const now = checks(1, 'no', 'no', 'yes');
  assert.deepEqual(decideFollowStep(STEPS, 1, 1, now, now), {
    kind: 'skip',
    fromIndex: 1,
    toIndex: 3,
    stepId: 's4',
    skippedIds: ['s2', 's3'],
  });
});

test('the last step yes changes no step (completion goes through goal_seen → confirm(goal_check) only)', () => {
  assert.deepEqual(decideFollowStep(STEPS, 3, 3, checks(3, 'yes'), checks(3, 'yes')), { kind: 'none' });
});

test('stillThere: the screen moved while the call was in flight → nothing changes', () => {
  assert.deepEqual(decideFollowStep(STEPS, 0, 1, checks(0, 'yes', 'no', 'no', 'no'), null), { kind: 'none' });
  const skipping = checks(0, 'no', 'no', 'yes', 'no');
  assert.deepEqual(decideFollowStep(STEPS, 0, 1, skipping, skipping), { kind: 'none' });
});

test('unsure and no never move the screen, forward or back', () => {
  assert.deepEqual(decideFollowStep(STEPS, 0, 0, checks(0, 'unsure', 'unsure', 'unsure', 'unsure'), null), { kind: 'none' });
  assert.deepEqual(decideFollowStep(STEPS, 1, 1, checks(1, 'no', 'no', 'no'), checks(0, 'yes', 'no', 'no', 'no')), {
    kind: 'none',
  });
  assert.deepEqual(decideFollowStep(STEPS, 0, 0, [], null), { kind: 'none' });
});

test('ids outside the plan or before the asked step are ignored', () => {
  const odd: StepCheck[] = [
    { step_id: 's1', visible: 'yes' },
    { step_id: 's9', visible: 'yes' },
  ];
  assert.deepEqual(decideFollowStep(STEPS, 1, 1, odd, odd), { kind: 'none' });
});

test('currentStepVisible reads the asked step entry (null when absent)', () => {
  assert.equal(currentStepVisible(STEPS, 1, checks(1, 'unsure', 'yes', 'no')), 'unsure');
  assert.equal(currentStepVisible(STEPS, 0, checks(1, 'yes', 'yes', 'no')), null);
});
