/**
 * What a talk answer does to the guide run (design 2026-10-01 guide-talk §C). Pure; the hook applies it.
 *
 * - `rejected` (`acceptance.ts`) → nothing changes. `bound` and `text_only` both apply: the action comes from
 *   the user's words, not from geometry on the captured frame, so a newer anchor generation does not void it.
 * - One answer may combine actions (server rule: at most one of step_mark/go_to/replan, plus target and step_say,
 *   no step_say beside a replan). They apply in order: plan text, then position, then target — advice reaches the
 *   step sentence and the overlay together (operator, 2026-10-02: "장갑 끼고 왔어").
 * - `step_say` → the sentence of the step the user talked about is replaced; the screen's step stays.
 *   `replan` → the steps are replaced and the guide starts again at step 1 (as a confirm replan); it spends
 *   one of the run's replans but not the replan gap (`REPLAN_MIN_GAP_MS`). Both take the answer's (bumped) `plan_revision`.
 * - `step_mark` moves the screen only while it still shows the step the user talked about (a follow may have
 *   moved it while the call was out): `done` marks it user-confirmed, `skipped` marks it skipped, and either
 *   moves to the next step. A `done` for a step the screen already left is still recorded (so a late
 *   confirm(step_done) `no` cannot revert it) without moving anything. On the last step the screen stays and
 *   `done` asks confirm(goal_check) (`checkGoal`): completion is only ever decided by confirm.
 * - `go_to` → back to an earlier step; user marks and skips at or after it are forgotten.
 * - `target` → the tracker restarts with the new noun (`retarget`, no paid call); the step stays.
 * - Every step change resets the unsure streak, the stuck count and the last checklist (a skip's first
 *   sighting).
 */
import type { GuideStep, StepCheck } from '../generated/api.generated';
import type { Acceptance } from './acceptance';
import { addSkipped, skippedBefore } from './planProgress';
import { notePlanInstalled, type ReplanBudget } from './replanPolicy';

/** The run fields a talk answer may change (copied out of and back into the hook's run). */
export interface TalkRunState {
  steps: GuideStep[];
  stepIndex: number;
  planRevision: number;
  target: string;
  skipped: string[];
  /** Steps the user said were done (사용자 확인 ✓). */
  userDone: string[];
  unsureStreak: number;
  stuckCount: number;
  lastChecks: StepCheck[] | null;
  replanBudget: ReplanBudget;
}

/** The parts of a talk answer the reducer reads (the generated `GuideTalkResponse` fits it). */
export interface TalkAnswerFields {
  reply: string;
  step_say?: string | null;
  target?: string | null;
  step_mark?: 'done' | 'skipped' | null;
  go_to?: string | null;
  replan?: { steps: GuideStep[] } | null;
  plan_revision: number;
}

export type TalkAction = 'none' | 'step_say' | 'target' | 'step_mark' | 'go_to' | 'replan';

export type TalkEffect =
  | { kind: 'retarget'; target: string }
  /** The user said the LAST step is done: confirm(goal_check) on a fresh frame. */
  | { kind: 'checkGoal' }
  | { kind: 'notice'; text: string };

export interface TalkOutcome {
  state: TalkRunState;
  effects: TalkEffect[];
  /** The first of `actions` (`none` for a reply only). */
  action: TalkAction;
  /** Every action the answer took, in the server's precedence order (state action, target, step_say). */
  actions: TalkAction[];
  /** False when the answer was discarded (rejected); `state` is then the input, unchanged. */
  applied: boolean;
}

export const TALK_REPLAN_NOTICE = '말씀대로 계획을 다시 짰습니다. 새 1단계부터 안내합니다.';

const ACTION_ORDER = ['step_mark', 'go_to', 'replan', 'target', 'step_say'] as const;

/** The actions an answer takes, in the server's order (`guide_contracts.TALK_ACTION_PRECEDENCE`). */
export function talkActions(answer: TalkAnswerFields): TalkAction[] {
  return ACTION_ORDER.filter((name) => Boolean(answer[name]));
}

export function talkAction(answer: TalkAnswerFields): TalkAction {
  return talkActions(answer)[0] ?? 'none';
}

/** The server changed the stored plan for this answer, so its `plan_revision` is the new current one. */
export function talkChangesPlan(answer: TalkAnswerFields): boolean {
  return Boolean(answer.step_say || answer.replan);
}

function moveTo(state: TalkRunState, index: number): TalkRunState {
  return { ...state, stepIndex: index, unsureStreak: 0, stuckCount: 0, lastChecks: null };
}

/** The user's done/skipped mark for `sentStepId` (moves on only while the screen still shows that step). */
function applyMark(state: TalkRunState, sentStepId: string, mark: 'done' | 'skipped', effects: TalkEffect[]): TalkRunState {
  const index = state.stepIndex;
  if (state.steps[index]?.id !== sentStepId) {
    return mark === 'done' ? { ...state, userDone: addSkipped(state.userDone, [sentStepId]) } : state;
  }
  const last = index >= state.steps.length - 1;
  if (mark === 'done') {
    const marked = { ...state, userDone: addSkipped(state.userDone, [sentStepId]), skipped: state.skipped.filter((id) => id !== sentStepId) };
    if (last) effects.push({ kind: 'checkGoal' });
    return last ? marked : moveTo(marked, index + 1);
  }
  const marked = { ...state, skipped: addSkipped(state.skipped, [sentStepId]), userDone: state.userDone.filter((id) => id !== sentStepId) };
  return last ? marked : moveTo(marked, index + 1);
}

/** Back to the step `stepId` if it is before the current one; marks and skips at or after it are forgotten. */
function applyGoTo(state: TalkRunState, stepId: string): TalkRunState {
  const index = state.steps.findIndex((step) => step.id === stepId);
  if (index < 0 || index >= state.stepIndex) return state;
  return moveTo(
    { ...state, skipped: skippedBefore(state.steps, state.skipped, index), userDone: skippedBefore(state.steps, state.userDone, index) },
    index
  );
}

/**
 * Apply every action of one answer, in order: the plan text (a replan, or the step_say of the step the user
 * talked about), then the position (step_mark or go_to), then the target (tracking restarts on the new noun —
 * last, so the overlay moves onto the thing the rewritten step names).
 */
export function applyTalk(
  state: TalkRunState,
  sentStepId: string,
  answer: TalkAnswerFields,
  acceptance: Acceptance,
  nowMs: number
): TalkOutcome {
  const actions = talkActions(answer);
  const action = actions[0] ?? 'none';
  if (acceptance === 'rejected') return { state, effects: [], action, actions, applied: false };

  const effects: TalkEffect[] = [];
  let next = state;
  if (answer.replan) {
    const budget = notePlanInstalled({ ...next.replanBudget, used: next.replanBudget.used + 1 }, nowMs);
    next = { ...moveTo(next, 0), steps: answer.replan.steps, planRevision: answer.plan_revision, skipped: [], userDone: [], replanBudget: budget };
    effects.push({ kind: 'notice', text: TALK_REPLAN_NOTICE });
  } else if (answer.step_say) {
    const say = answer.step_say;
    next = { ...next, steps: next.steps.map((step) => (step.id === sentStepId ? { ...step, say } : step)), planRevision: answer.plan_revision };
  }
  if (answer.step_mark) next = applyMark(next, sentStepId, answer.step_mark, effects);
  else if (answer.go_to) next = applyGoTo(next, answer.go_to);
  if (answer.target) {
    next = { ...next, target: answer.target };
    effects.push({ kind: 'retarget', target: answer.target });
  }
  return { state: next, effects, action, actions, applied: true };
}
