/**
 * The sequential core's step machine (pure; `useIntentLoop` applies what it returns).
 *
 * The follower's checklist only nominates; the upper model's `step_done` answer decides. A step moves
 * exactly ONE position, and only when all of these hold:
 *
 * - the answer is a bound confirm that carried the frame of the candidate (the follow's own frame), and
 * - that frame is one this run nominated for the step and has not already committed, and
 * - the candidate's plan revision, step index and step activation are still the run's current ones, and
 * - the answer's `step_id` names that same step, and
 * - `step_check` is `yes`.
 *
 * Everything else HOLDS: `no`, `unsure`, a missing check, a rejected/text-only answer, a stale revision, a
 * replayed frame, a step that was re-activated (a talk reposition back to the same index) or an answer that
 * arrived after the step moved. A hold only releases the candidate — the next accepted follow may nominate
 * again on a fresh frame, so a single `no` never latches the step shut.
 *
 * The last step can be confirmed the same way and stays where it is (`final`): there is no out-of-range
 * move, and completion is still decided only by the run's own goal confirm.
 *
 * Evidence committed this way is bound to (plan revision, step index and id, step activation, frame) and
 * kept for the run, so an answer can never be applied to a different step; a new plan (replan, or a talk
 * replan) starts the machine over, and every step activation invalidates the candidates raised before it.
 */
import type { StepCheck } from './followStep';

/** A step the follower saw done, waiting for the upper model's own `step_done: yes`. */
export interface SequentialCandidate {
  planRevision: number;
  /** The screen's step when the follow was asked (`step_done` checks this step). */
  stepIndex: number;
  stepId: string;
  /** The step activation the candidate was nominated under. */
  activation: number;
  /** `frame_id` of the follow frame its confirm re-checks. */
  frameId: string;
  /** `performance.now()` at nomination (the waiting notice's age). */
  askedAt: number;
}

/** One upper-confirmed step, bound to the revision, activation and frame it was confirmed on. */
export interface SequentialEvidence {
  planRevision: number;
  stepIndex: number;
  stepId: string;
  activation: number;
  frameId: string;
}

export interface SequentialState {
  /** Bumped on every step activation, so neither a candidate nor evidence applies to a later one. */
  activation: number;
  /** At most one candidate: later follows cannot displace an in-flight confirmation. */
  awaiting: SequentialCandidate[];
  /** Steps an upper `step_done: yes` committed in this plan. */
  evidence: SequentialEvidence[];
  /** Frames that already committed a step: a replay of one never moves another step. */
  committedFrames: string[];
}

export function freshSequentialState(): SequentialState {
  return { activation: 1, awaiting: [], evidence: [], committedFrames: [] };
}

/** A step activation begins (any step change): candidates raised for the previous one are moot. */
export function noteStepActivation(state: SequentialState): SequentialState {
  return { ...state, activation: state.activation + 1, awaiting: [] };
}

/** A new plan text (a replan): the step machine starts over on the new steps. Committed frames are kept. */
export function resetSequentialForPlan(state: SequentialState): SequentialState {
  return { activation: state.activation + 1, awaiting: [], evidence: [], committedFrames: state.committedFrames };
}

/**
 * Record a candidate for the step the follower just called done. A step whose confirmation for this
 * activation is already committed nominates nothing (there is nothing left for a confirm to decide).
 */
export function nominateSequential(
  state: SequentialState,
  candidate: SequentialCandidate
): SequentialState {
  if (candidate.activation !== state.activation || state.awaiting.length > 0 ||
      state.committedFrames.includes(candidate.frameId)) return state;
  const confirmed = state.evidence.some(
    (entry) => entry.activation === state.activation && entry.planRevision === candidate.planRevision && entry.stepId === candidate.stepId
  );
  if (confirmed) return state;
  return {
    ...state,
    awaiting: [candidate],
  };
}

/**
 * Drop the candidate that `frameId` was nominated with (a hold, a rejected answer, a failed call): either
 * by that frame (`null` drops nothing), so other candidates of the same step are untouched.
 */
export function dropSequentialAwaiting(state: SequentialState, frameId: string | null): SequentialState {
  if (frameId === null) return state;
  return { ...state, awaiting: state.awaiting.filter((entry) => entry.frameId !== frameId) };
}

export type SequentialOutcome =
  /** The answer was not about a candidate of this run. */
  | { kind: 'none' }
  /** The candidate resolved without moving the step (`no`, `unsure`, stale, replay). */
  | { kind: 'hold' }
  /** The step was committed: move exactly one step forward. */
  | { kind: 'advance'; toIndex: number }
  /** The LAST step was committed: stay on it (no out-of-range move; completion is the goal confirm's). */
  | { kind: 'final' };

/** The bound `step_done` answer a candidate is settled against, plus the run state it must still match. */
export interface SequentialSettlement {
  frameId: string | null;
  /** The run's current plan revision and step index. */
  planRevision: number;
  stepIndex: number;
  stepCheck: 'yes' | 'no' | 'unsure' | null;
  /** The step the server reported checking (null when it reported none). */
  stepId: string | null;
  stepCount: number;
}

/** Apply one bound `step_done` answer to the candidate it was asked with. */
export function settleSequential(
  state: SequentialState,
  answer: SequentialSettlement
): { state: SequentialState; outcome: SequentialOutcome } {
  if (answer.frameId === null) return { state, outcome: { kind: 'none' } };
  const candidate = state.awaiting.find((entry) => entry.frameId === answer.frameId);
  if (!candidate) return { state, outcome: { kind: 'none' } };

  const held = dropSequentialAwaiting(state, candidate.frameId);
  const stale =
    candidate.planRevision !== answer.planRevision ||
    candidate.stepIndex !== answer.stepIndex ||
    candidate.activation !== state.activation ||
    state.committedFrames.includes(candidate.frameId) ||
    answer.stepId !== candidate.stepId;
  if (stale || answer.stepCheck !== 'yes') return { state: held, outcome: { kind: 'hold' } };

  const evidence: SequentialEvidence = {
    planRevision: candidate.planRevision,
    stepIndex: candidate.stepIndex,
    stepId: candidate.stepId,
    activation: candidate.activation,
    frameId: candidate.frameId,
  };
  const committed: SequentialState = {
    ...held,
    evidence: [...held.evidence, evidence],
    committedFrames: [...held.committedFrames, candidate.frameId],
  };
  if (candidate.stepIndex >= answer.stepCount - 1) return { state: committed, outcome: { kind: 'final' } };
  return { state: committed, outcome: { kind: 'advance', toIndex: candidate.stepIndex + 1 } };
}

/**
 * The sequential core shows only the CURRENT step's observation: a future step's reading is not this
 * client's to display (the follower in this core judges the current step only).
 */
export function currentStepChecks(
  steps: ReadonlyArray<{ id: string }>,
  stepIndex: number,
  checks: ReadonlyArray<StepCheck> | null
): StepCheck[] | null {
  if (!checks) return null;
  const step = steps[stepIndex];
  if (!step) return null;
  return checks.filter((check) => check.step_id === step.id);
}
