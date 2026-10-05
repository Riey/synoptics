/**
 * Confirm frame source (the follow's own frame for follow-raised confirms) and the bound-answer decision
 * (step and completion decided apart).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  confirmFrameSource,
  decideConfirm,
  decideConfirmPosition,
  exitEdgeOf,
  recheckCall,
  type FollowFrame,
} from './confirmDecision.ts';

const FRAME: FollowFrame = {
  frameId: 'guide-frame-1',
  imageBase64: 'AAAA',
  capturedAt: 1_000,
  anchor: {
    anchor_id: 'a1',
    role: 'target',
    label: 'glasses',
    run_id: 'run-1',
    track_id: 'trk-1',
    generation: 2,
    state: 'tracking',
    box: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 },
  },
  runId: 'run-1',
  planId: 'plan-1',
  planRevision: 1,
};
const CURRENT = { planId: 'plan-1', planRevision: 1, runId: 'run-1' };

test('follow-raised step_done and goal_check reuse the follow frame and anchor snapshot', () => {
  for (const trigger of ['step_done', 'goal_check']) {
    const source = confirmFrameSource(trigger, { followFrame: FRAME }, CURRENT);
    assert.equal(source.kind, 'follow');
    assert.equal(source.kind === 'follow' && source.frame, FRAME);
  }
});

test('anchor_lost goal_check, the completion re-check and the other confirms take a fresh capture', () => {
  assert.deepEqual(confirmFrameSource('goal_check', {}, CURRENT), { kind: 'fresh' });
  assert.deepEqual(confirmFrameSource('goal_check', { recheck: true, followFrame: FRAME }, CURRENT), { kind: 'fresh' });
  assert.deepEqual(confirmFrameSource('unsure_twice', { followFrame: FRAME }, CURRENT), { kind: 'fresh' });
  assert.deepEqual(confirmFrameSource('replan', { followFrame: FRAME }, CURRENT), { kind: 'fresh' });
});

test('a follow frame from another plan revision or tracking run is dropped, not sent', () => {
  assert.deepEqual(confirmFrameSource('step_done', { followFrame: FRAME }, { ...CURRENT, planRevision: 2 }), { kind: 'drop' });
  assert.deepEqual(confirmFrameSource('step_done', { followFrame: FRAME }, { ...CURRENT, planId: 'plan-2' }), { kind: 'drop' });
  assert.deepEqual(confirmFrameSource('goal_check', { followFrame: FRAME }, { ...CURRENT, runId: 'run-2' }), { kind: 'drop' });
  assert.deepEqual(confirmFrameSource('goal_check', { followFrame: FRAME }, { ...CURRENT, runId: null }), { kind: 'drop' });
});

const ADVANCE = { fromIndex: 0, toIndex: 1, stepId: 's1' };
const IDLE = { stepIndex: 1, completion: 'none' as const, trackLost: false };
const stepDone = (step_check: 'yes' | 'no' | 'unsure', status: string, step_id = 's1') => ({
  trigger: 'step_done',
  goal_status: { status },
  step_check,
  step_id,
});

test('step_check no + visually_satisfied: the step reverts AND completion starts its 1.5 s re-check', () => {
  assert.deepEqual(decideConfirm(stepDone('no', 'visually_satisfied'), ADVANCE, IDLE), {
    completion: 'start_check',
    step: { kind: 'revert', toIndex: 0 },
    lostNotice: false,
    inferred: false,
  });
});

test('step decision alone: no reverts, yes/unsure keep the advance; completion untouched unless satisfied', () => {
  assert.deepEqual(decideConfirm(stepDone('no', 'in_progress'), ADVANCE, IDLE), {
    completion: 'keep',
    step: { kind: 'revert', toIndex: 0 },
    lostNotice: false,
    inferred: false,
  });
  assert.deepEqual(decideConfirm(stepDone('yes', 'in_progress'), ADVANCE, IDLE).step, { kind: 'none' });
  assert.deepEqual(decideConfirm(stepDone('unsure', 'in_progress'), ADVANCE, IDLE).step, { kind: 'none' });
  assert.deepEqual(decideConfirm(stepDone('yes', 'visually_satisfied'), ADVANCE, IDLE), {
    completion: 'start_check',
    step: { kind: 'none' },
    lostNotice: false,
    inferred: false,
  });
});

test('no revert when the screen moved on, the echo names another step, or the advance was clamped', () => {
  assert.deepEqual(decideConfirm(stepDone('no', 'in_progress'), ADVANCE, { ...IDLE, stepIndex: 2 }).step, { kind: 'none' });
  assert.deepEqual(decideConfirm(stepDone('no', 'in_progress', 's2'), ADVANCE, IDLE).step, { kind: 'none' });
  const clamped = { fromIndex: 3, toIndex: 3, stepId: 's4' };
  assert.deepEqual(decideConfirm(stepDone('no', 'in_progress', 's4'), clamped, { ...IDLE, stepIndex: 3 }).step, { kind: 'none' });
});

test('a skip is re-checked like an advance: no for the skipped-to step reverts to the asked step', () => {
  // Skip from s1 (index 0) past s3 (yes twice) to s4 (index 3); the confirm checks s3.
  const skip = { fromIndex: 0, toIndex: 3, stepId: 's3' };
  const onTarget = { ...IDLE, stepIndex: 3 };
  assert.deepEqual(decideConfirm(stepDone('no', 'in_progress', 's3'), skip, onTarget), {
    completion: 'keep',
    step: { kind: 'revert', toIndex: 0 },
    lostNotice: false,
    inferred: false,
  });
  assert.deepEqual(decideConfirm(stepDone('yes', 'visually_satisfied', 's3'), skip, onTarget), {
    completion: 'start_check',
    step: { kind: 'none' },
    lostNotice: false,
    inferred: false,
  });
  assert.deepEqual(decideConfirm(stepDone('unsure', 'in_progress', 's3'), skip, onTarget).step, { kind: 'none' });
});

test('completion: re-check confirms only from checking; already checking does not restart', () => {
  const goal = (status: string) => ({ trigger: 'goal_check', goal_status: { status } });
  assert.equal(decideConfirm(goal('visually_satisfied'), { recheck: true }, { ...IDLE, completion: 'checking' }).completion, 'confirmed');
  assert.equal(decideConfirm(goal('in_progress'), { recheck: true }, { ...IDLE, completion: 'checking' }).completion, 'recheck_failed');
  assert.equal(decideConfirm(goal('visually_satisfied'), {}, { ...IDLE, completion: 'checking' }).completion, 'keep');
  assert.equal(decideConfirm(goal('uncertain'), {}, { ...IDLE, trackLost: true }).lostNotice, true);
  assert.equal(decideConfirm(goal('uncertain'), {}, IDLE).lostNotice, false);
});

test('a step the user said was done is never reverted by a late step_done no', () => {
  const userDone = { ...IDLE, userDoneIds: ['s1'] };
  assert.deepEqual(decideConfirm(stepDone('no', 'in_progress'), ADVANCE, userDone).step, { kind: 'none' });
  // Another step's user mark does not protect this one.
  assert.deepEqual(decideConfirm(stepDone('no', 'in_progress'), ADVANCE, { ...IDLE, userDoneIds: ['s2'] }).step, {
    kind: 'revert',
    toIndex: 0,
  });
});

test('review #2: a completion re-check dropped as stale is scheduled again while checking', async () => {
  const { retryRecheckAfterStale } = await import('./confirmDecision.ts');
  assert.equal(retryRecheckAfterStale({ recheck: true }, 'checking'), true);
  assert.equal(retryRecheckAfterStale({ recheck: true }, 'none'), false);
  assert.equal(retryRecheckAfterStale({}, 'checking'), false);
  assert.equal(retryRecheckAfterStale(undefined, 'checking'), false);
});

// ---------------------------------------------------------------------------------------------- target_left

const UNCERTAIN = { status: 'uncertain' };
const SATISFIED = { status: 'visually_satisfied' };
const LOST = { stepIndex: 2, completion: 'none' as const, trackLost: true };

test('target_left inferred_done yes starts the completion check without a visible goal', () => {
  const decision = decideConfirm({ trigger: 'target_left', goal_status: UNCERTAIN, inferred_done: 'yes' }, {}, LOST);
  assert.equal(decision.completion, 'start_check');
  assert.equal(decision.inferred, true);
  assert.equal(decision.lostNotice, false);
});

test('target_left no/unsure keeps the lost notice; a visible goal is not an inference', () => {
  for (const inferred of ['no', 'unsure'] as const) {
    const decision = decideConfirm({ trigger: 'target_left', goal_status: UNCERTAIN, inferred_done: inferred }, {}, LOST);
    assert.equal(decision.completion, 'keep');
    assert.equal(decision.lostNotice, true);
    assert.equal(decision.inferred, false);
  }
  const visible = decideConfirm({ trigger: 'target_left', goal_status: SATISFIED, inferred_done: 'yes' }, {}, LOST);
  assert.equal(visible.completion, 'start_check');
  assert.equal(visible.inferred, false);
});

test('inferred_done counts only on target_left', () => {
  const decision = decideConfirm({ trigger: 'goal_check', goal_status: UNCERTAIN, inferred_done: 'yes' }, {}, LOST);
  assert.equal(decision.completion, 'keep');
  assert.equal(decision.lostNotice, true);
});

test('a target_left re-check confirms on inference and is asked again as target_left with the same frame', () => {
  const recheck = recheckCall('target_left', { beforeFrame: FRAME, exitEdge: 'right' });
  assert.deepEqual(recheck, { trigger: 'target_left', meta: { recheck: true, beforeFrame: FRAME, exitEdge: 'right' } });
  assert.deepEqual(recheckCall('goal_check', { followFrame: FRAME }), { trigger: 'goal_check', meta: { recheck: true } });
  assert.deepEqual(recheckCall('target_left', {}), { trigger: 'goal_check', meta: { recheck: true } });

  const checking = { ...LOST, completion: 'checking' as const };
  const confirmed = decideConfirm({ trigger: 'target_left', goal_status: UNCERTAIN, inferred_done: 'yes' }, recheck.meta, checking);
  assert.equal(confirmed.completion, 'confirmed');
  assert.equal(confirmed.inferred, true);
  const failed = decideConfirm({ trigger: 'target_left', goal_status: UNCERTAIN, inferred_done: 'unsure' }, recheck.meta, checking);
  assert.equal(failed.completion, 'recheck_failed');
});

test('exit edge: the nearest frame edge within the margin, otherwise none', () => {
  assert.equal(exitEdgeOf({ x: 0.85, y: 0.4, width: 0.13, height: 0.2 }), 'right');
  assert.equal(exitEdgeOf({ x: 0.01, y: 0.4, width: 0.2, height: 0.2 }), 'left');
  assert.equal(exitEdgeOf({ x: 0.4, y: 0.75, width: 0.2, height: 0.22 }), 'bottom');
  assert.equal(exitEdgeOf({ x: 0.4, y: 0.0, width: 0.2, height: 0.2 }), 'top');
  assert.equal(exitEdgeOf({ x: 0.3, y: 0.3, width: 0.3, height: 0.3 }), 'none');
});

test('a completion re-check fired after the target was lost goes as target_left with the run start frame', () => {
  const lost = { beforeFrame: FRAME, exitEdge: 'bottom' as const };
  assert.deepEqual(recheckCall('goal_check', { followFrame: FRAME }, lost), {
    trigger: 'target_left',
    meta: { recheck: true, beforeFrame: FRAME, exitEdge: 'bottom' },
  });
  assert.deepEqual(recheckCall('goal_check', {}, null), { trigger: 'goal_check', meta: { recheck: true } });
  // A target_left completion keeps its own earlier frame.
  const own = { ...FRAME, frameId: 'guide-frame-0' };
  assert.equal(recheckCall('target_left', { beforeFrame: own }, lost).meta.beforeFrame, own);
});

// ------------------------------------------------- replan/unsure_twice step_checks place the screen

const STEPS = [{ id: 's1' }, { id: 's2' }, { id: 's3' }, { id: 's4' }];
const checks = (...visible: Array<'yes' | 'no' | 'unsure'>) =>
  visible.map((v, i) => ({ step_id: `s${i + 1}`, visible: v }));
/** Checks from step `from` (0-based) to the last, as the server asks for them. */
const checksFrom = (from: number, ...visible: Array<'yes' | 'no' | 'unsure'>) =>
  visible.map((v, i) => ({ step_id: `s${from + i + 1}`, visible: v }));

test('position: the current step yes advances to the next step, nothing skipped', () => {
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, checks('yes', 'no', 'unsure', 'no'), false), {
    kind: 'advance',
    toIndex: 1,
    skippedIds: [],
  });
});

test('position: a later yes skips straight past it; the non-yes steps between are skipped', () => {
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, checks('unsure', 'no', 'yes', 'no'), false), {
    kind: 'advance',
    toIndex: 3,
    skippedIds: ['s1', 's2'],
  });
  // Yes steps along the way are done, not skipped; the latest yes decides.
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, checks('yes', 'unsure', 'yes', 'no'), false), {
    kind: 'advance',
    toIndex: 3,
    skippedIds: ['s2'],
  });
});

test('position: from a later current step, the checks start there', () => {
  assert.deepEqual(decideConfirmPosition(STEPS, 1, 1, checksFrom(1, 'no', 'yes', 'unsure'), false), {
    kind: 'advance',
    toIndex: 3,
    skippedIds: ['s2'],
  });
});

test('position: no yes, or a yes only before the current step, changes nothing', () => {
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, checks('unsure', 'no', 'unsure', 'no'), false), { kind: 'none' });
  assert.deepEqual(decideConfirmPosition(STEPS, 2, 2, checks('yes', 'yes', 'no', 'unsure'), false), { kind: 'none' });
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, [], false), { kind: 'none' });
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, null, false), { kind: 'none' });
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, undefined, false), { kind: 'none' });
});

test('position: the last step yes clamps to it; already on it is completion business, not a move', () => {
  assert.deepEqual(decideConfirmPosition(STEPS, 1, 1, checksFrom(1, 'unsure', 'no', 'yes'), false), {
    kind: 'advance',
    toIndex: 3,
    skippedIds: ['s2', 's3'],
  });
  assert.deepEqual(decideConfirmPosition(STEPS, 3, 3, checksFrom(3, 'yes'), false), { kind: 'none' });
});

test('position: a screen that moved while the confirm was out is left alone', () => {
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 1, checks('yes', 'yes', 'no', 'no'), false), { kind: 'none' });
  assert.deepEqual(decideConfirmPosition(STEPS, 7, 7, checks('yes'), false), { kind: 'none' });
});

test('position: a replan in the same answer wins over its checks', () => {
  assert.deepEqual(decideConfirmPosition(STEPS, 0, 0, checks('yes', 'yes', 'no', 'no'), true), { kind: 'none' });
});

test('position: ids outside the plan are ignored', () => {
  assert.deepEqual(
    decideConfirmPosition(STEPS, 0, 0, [{ step_id: 's9', visible: 'yes' }, { step_id: 's1', visible: 'no' }], false),
    { kind: 'none' }
  );
});
