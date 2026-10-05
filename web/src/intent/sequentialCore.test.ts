/**
 * The sequential core's step machine: a follower `yes` only nominates, and only the upper model's own
 * exact-step `step_done: yes` (on the candidate's frame, plan revision and step activation) moves the screen
 * — exactly one step. Everything else holds and releases the candidate for a later fresh-frame nomination.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  currentStepChecks,
  dropSequentialAwaiting,
  freshSequentialState,
  nominateSequential,
  noteStepActivation,
  resetSequentialForPlan,
  settleSequential,
  type SequentialOutcome,
  type SequentialSettlement,
  type SequentialState,
} from './sequentialCore.ts';

const STEPS = [{ id: 's1' }, { id: 's2' }, { id: 's3' }];
const REVISION = 7;

function nominate(state: SequentialState, frameId: string, stepIndex = 0, planRevision = REVISION): SequentialState {
  return nominateSequential(state, {
    planRevision,
    stepIndex,
    stepId: STEPS[stepIndex].id,
    frameId,
    askedAt: 0,
    activation: state.activation,
  });
}

function settle(
  state: SequentialState,
  overrides: Partial<SequentialSettlement> = {}
): { state: SequentialState; outcome: SequentialOutcome } {
  return settleSequential(state, {
    frameId: 'f1',
    planRevision: REVISION,
    stepIndex: 0,
    stepCheck: 'yes',
    stepId: 's1',
    stepCount: STEPS.length,
    ...overrides,
  });
}

test('a follower yes only nominates: the step does not move until the same-step upper yes arrives', () => {
  const nominated = nominate(freshSequentialState(), 'f1');
  assert.equal(nominated.awaiting.length, 1);
  assert.deepEqual(nominated.evidence, []);

  const held = settle(nominated, { stepCheck: 'no' });
  assert.equal(held.outcome.kind, 'hold');
  assert.deepEqual(held.state.awaiting, []);
  assert.deepEqual(held.state.evidence, []);

  // A hold does not latch the step shut: the next accepted follow may nominate on a fresh frame.
  const again = nominate(held.state, 'f2');
  const confirmed = settle(again, { frameId: 'f2' });
  assert.deepEqual(confirmed.outcome, { kind: 'advance', toIndex: 1 });
  assert.equal(confirmed.state.awaiting.length, 0);
  assert.deepEqual(confirmed.state.evidence, [
    { planRevision: REVISION, stepIndex: 0, stepId: 's1', activation: 1, frameId: 'f2' },
  ]);
  assert.deepEqual(confirmed.state.committedFrames, ['f2']);
});

test('an upper yes that names another step, or arrives for a frame never nominated, never moves the step', () => {
  const nominated = nominate(freshSequentialState(), 'f1');

  const wrongStep = settle(nominated, { stepId: 's2' });
  assert.equal(wrongStep.outcome.kind, 'hold');
  assert.equal(settle(nominated, { stepId: null }).outcome.kind, 'hold');

  // A frame this run never nominated is not a candidate of ours: it is neither applied nor consumed.
  const foreignFrame = settle(wrongStep.state, { frameId: 'f9', stepId: 's1' });
  assert.equal(foreignFrame.outcome.kind, 'none');
  assert.equal(foreignFrame.state.awaiting.length, 0);
  assert.deepEqual(foreignFrame.state.evidence, []);

  // `unsure` is a hold like `no` (the step stays; only a `yes` commits).
  const unsure = settle(nominate(freshSequentialState(), 'f1'), { stepCheck: 'unsure' });
  assert.equal(unsure.outcome.kind, 'hold');
  assert.deepEqual(unsure.state.evidence, []);
});

test('a replayed frame never advances a newly activated step', () => {
  const advanced = settle(nominate(freshSequentialState(), 'f1')).state;
  const nextActivation = noteStepActivation(advanced);

  // A used frame is rejected before a second upper confirmation can even be queued.
  const replayed = nominate(nextActivation, 'f1', 1);
  assert.deepEqual(replayed.awaiting, []);
  const outcome = settle(replayed, { frameId: 'f1', stepIndex: 1, stepId: 's2' });
  assert.equal(outcome.outcome.kind, 'none');
  assert.deepEqual(outcome.state.evidence.map((entry) => entry.stepId), ['s1']);
});

test('a new step activation ends the candidates of the previous one (a talk reposition back to the same index)', () => {
  const nominated = nominate(freshSequentialState(), 'f1');
  const reactivated = noteStepActivation(nominated);
  assert.deepEqual(reactivated.awaiting, []);

  const late = settle(reactivated, { frameId: 'f1' });
  assert.equal(late.outcome.kind, 'none');
  assert.deepEqual(late.state.evidence, []);
  const lateFollow = nominateSequential(reactivated, {
    planRevision: REVISION, stepIndex: 0, stepId: 's1', frameId: 'old-follow',
    askedAt: 0, activation: nominated.activation,
  });
  assert.equal(lateFollow, reactivated);
});

test('a candidate is bound to its plan revision and to the step index it was asked about', () => {
  const nominated = nominate(freshSequentialState(), 'f1');
  assert.equal(settle(nominated, { planRevision: REVISION + 1 }).outcome.kind, 'hold');
  // The screen moved on (a talk mark) while the confirm was out: the answer is about the step it left.
  assert.equal(settle(nominated, { stepIndex: 1, stepId: 's2' }).outcome.kind, 'hold');
});

test('the last step is confirmed without moving off the end, and it is not asked for twice', () => {
  const lastIndex = STEPS.length - 1;
  const nominated = nominate(freshSequentialState(), 'f1', lastIndex);
  const confirmed = settle(nominated, { stepIndex: lastIndex, stepId: 's3' });
  assert.equal(confirmed.outcome.kind, 'final');
  assert.deepEqual(confirmed.state.committedFrames, ['f1']);
  assert.equal(confirmed.state.evidence.length, 1);

  // This activation already carries its confirmation: a later `yes` nominates nothing (no repeat paid call).
  assert.equal(nominate(confirmed.state, 'f2', lastIndex), confirmed.state);
});

test('a new plan starts the step machine over and keeps the committed-frame guard', () => {
  const advanced = settle(nominate(freshSequentialState(), 'f1')).state;
  const replanned = resetSequentialForPlan(advanced);
  assert.equal(replanned.activation, advanced.activation + 1);
  assert.deepEqual(replanned.awaiting, []);
  assert.deepEqual(replanned.evidence, []);
  assert.deepEqual(replanned.committedFrames, ['f1']);
});

test('a slow upper confirmation keeps its candidate despite later follower results', () => {
  const first = nominate(freshSequentialState(), 'f1');
  let pending = first;
  for (let i = 2; i < 10; i++) pending = nominate(pending, `f${i}`);
  assert.equal(pending, first);
  assert.deepEqual(settle(pending).outcome, { kind: 'advance', toIndex: 1 });
  const released = dropSequentialAwaiting(pending, 'f1');
  assert.equal(nominate(released, 'fresh').awaiting[0].frameId, 'fresh');
});

test('only the current step observation is kept for the sequential core', () => {
  const checks = [
    { step_id: 's1', visible: 'yes' as const },
    { step_id: 's2', visible: 'no' as const },
  ];
  assert.deepEqual(currentStepChecks(STEPS, 0, checks), [{ step_id: 's1', visible: 'yes' }]);
  assert.deepEqual(currentStepChecks(STEPS, 1, checks), [{ step_id: 's2', visible: 'no' }]);
  assert.equal(currentStepChecks(STEPS, 9, checks), null);
  assert.equal(currentStepChecks(STEPS, 0, null), null);
});
