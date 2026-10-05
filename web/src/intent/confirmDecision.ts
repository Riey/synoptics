/**
 * Confirm calls: which frame they are sent with, and what a bound answer does (pure; the hook applies it).
 *
 * Frame
 * - A confirm a FOLLOW answer asked for (step_done after an advance or a skip, goal_check after
 *   `goal_seen: yes`) re-checks what the follower saw, so it is sent with the follow's own captured frame
 *   (same `frame_id`, same JPEG bytes) and the same AnchorRef snapshot. A fresh capture 1.2-1.4 s later
 *   judged a later state and reverted correct advances (2/2 in the phase-4 E2E).
 * - Every other confirm (anchor_lost goal_check, unsure_twice, replan, the 1.5 s completion re-check) takes a
 *   fresh capture: judging the current frame is its purpose.
 * - The follow's frame is only usable for the plan and tracking run it was captured under; otherwise the
 *   question it raised is moot and the confirm is dropped instead of paid for.
 *
 * Answer
 * - The step decision and the completion decision are separate. `step_check: no` for the step the client
 *   left reverts it even when the same answer says `visually_satisfied`; completion then still goes
 *   through the normal 1.5 s re-check on a fresh frame.
 * - A step the user confirmed by talk (`user_done`) is never reverted: a late `step_check: no` for it is ignored.
 *
 * Target left the frame (`target_left`)
 * - When the tracker loses the target, the judge gets the FIRST tracked frame of the run (the starting state)
 *   beside a fresh one — not the last: a tracker can hold on to the face or a hand for seconds after the object
 *   is gone, so the last tracked frame may already show the end state (bug report 20261003-102137, 2.9 s) and may answer `inferred_done: yes` (taken off and put down out of view, for instance). That
 *   counts as satisfied for completion, through the same 1.5 s re-check (asked as `target_left` again, with
 *   the same earlier frame), and the completion notice says it was inferred.
 *
 * Position (`replan`/`unsure_twice`, `decideConfirmPosition`)
 * - Those confirms also answer `step_checks` (the follow checklist from the step the screen showed when it was
 *   asked, judged by the large model on the fresh frame). Unless the same answer installed a replan (a new plan
 *   starts at its own s1), the latest `yes` moves the screen to the step after it at once: no second sighting
 *   and no step_done re-check, the judge is the large model. Steps before it that are not `yes` are skipped.
 */
import type { AnchorRef, Box, ExitEdge, StepCheck } from '../generated/api.generated';

/** The frame a follow was sent with, kept only until the confirm it raised leaves (or is superseded). */
export interface FollowFrame {
  frameId: string;
  imageBase64: string;
  /** `Date.now()` at capture. */
  capturedAt: number;
  /** The AnchorRef snapshot the follow carried (null when the follow carried none). */
  anchor: AnchorRef | null;
  /** The tracking run the follow was fenced with. */
  runId: string;
  planId: string;
  planRevision: number;
}

/** What a confirm call carries besides its trigger. */
export interface ConfirmMeta {
  /**
   * step_done: the index the client advanced or skipped FROM, the index it moved to, and the step the confirm
   * checks (the asked step for an advance, the last `yes` step for a skip).
   */
  fromIndex?: number;
  toIndex?: number;
  stepId?: string;
  /** The second completion check (always a fresh capture). */
  recheck?: boolean;
  /** The follow's frame this confirm re-checks (set only by a follow answer). */
  followFrame?: FollowFrame;
  /** replan/unsure_twice: the last accepted follow's checklist, sent as `follow_checks`. */
  followChecks?: StepCheck[];
  /** replan/unsure_twice: the run's installed-replan count when it was asked (`replanDispatch`). */
  replanEpoch?: number;
  /** target_left: the first tracked frame of the run (sent as `before_scene`). */
  beforeFrame?: FollowFrame;
  /** target_left: the frame edge the last tracked box was at. */
  exitEdge?: ExitEdge;
}

/** Box within this distance of a frame edge (normalized) counts as leaving through that edge. */
export const EXIT_EDGE_MARGIN = 0.08;

/** The frame edge the last tracked box was nearest, or `none` when it was not near any. */
export function exitEdgeOf(box: Box): ExitEdge {
  const gaps: Array<[ExitEdge, number]> = [
    ['left', box.x],
    ['right', 1 - (box.x + box.width)],
    ['top', box.y],
    ['bottom', 1 - (box.y + box.height)],
  ];
  const [edge, gap] = gaps.reduce((best, next) => (next[1] < best[1] ? next : best));
  return gap <= EXIT_EDGE_MARGIN ? edge : 'none';
}

/**
 * The completion re-check, chosen when it fires: a target_left completion is re-checked the same way, and so is
 * any completion when the tracker has lost the target by then (`lost`: the run's starting frame and exit edge) —
 * a single fresh frame without the object cannot confirm a goal whose object has left (bug report attempt 1).
 */
export function recheckCall(
  trigger: string,
  meta: ConfirmMeta | undefined,
  lost?: { beforeFrame: FollowFrame; exitEdge?: ExitEdge } | null
): { trigger: 'goal_check' | 'target_left'; meta: ConfirmMeta } {
  if (trigger === 'target_left' && meta?.beforeFrame) {
    return { trigger: 'target_left', meta: { recheck: true, beforeFrame: meta.beforeFrame, exitEdge: meta.exitEdge } };
  }
  if (lost) return { trigger: 'target_left', meta: { recheck: true, beforeFrame: lost.beforeFrame, exitEdge: lost.exitEdge } };
  return { trigger: 'goal_check', meta: { recheck: true } };
}

export type ConfirmFrameSource = { kind: 'fresh' } | { kind: 'follow'; frame: FollowFrame } | { kind: 'drop' };

/**
 * Which frame a confirm leaves with. `current` is the guide's plan and the live tracking run at dispatch.
 */
export function confirmFrameSource(
  trigger: string,
  meta: ConfirmMeta,
  current: { planId: string | null; planRevision: number | null; runId: string | null }
): ConfirmFrameSource {
  const frame = meta.followFrame;
  if (!frame || meta.recheck || (trigger !== 'step_done' && trigger !== 'goal_check')) return { kind: 'fresh' };
  if (frame.planId !== current.planId || frame.planRevision !== current.planRevision || frame.runId !== current.runId) {
    return { kind: 'drop' };
  }
  return { kind: 'follow', frame };
}

/** The parts of a bound confirm answer the decision reads. */
export interface ConfirmAnswerFields {
  trigger: string;
  goal_status: { status: string };
  step_check?: 'yes' | 'no' | 'unsure' | null;
  step_id?: string | null;
  inferred_done?: 'yes' | 'no' | 'unsure' | null;
}

export type CompletionState = 'none' | 'checking' | 'confirmed' | 'user_confirmed';

export interface ConfirmDecision {
  /**
   * `start_check`: first `visually_satisfied` → "확인 중" and the 1.5 s re-check. `confirmed`: the re-check
   * agreed. `recheck_failed`: the re-check did not. `keep`: completion untouched.
   */
  completion: 'start_check' | 'confirmed' | 'recheck_failed' | 'keep';
  step:
    | { kind: 'none' }
    /** `step_check: no` for the step an advance or skip relied on: back to the step it left. */
    | { kind: 'revert'; toIndex: number };
  /** A goal_check/target_left that did not see (or infer) the goal while the tracker has lost the target. */
  lostNotice: boolean;
  /** The goal counted as satisfied only by inference (target_left `inferred_done: yes`), not by this frame. */
  inferred: boolean;
}

/** What a BOUND confirm answer does to completion and to the step, decided apart from each other. */
export function decideConfirm(
  answer: ConfirmAnswerFields,
  meta: ConfirmMeta,
  state: { stepIndex: number; completion: CompletionState; trackLost: boolean; userDoneIds?: ReadonlyArray<string> }
): ConfirmDecision {
  const visible = answer.goal_status.status === 'visually_satisfied';
  const inferred = !visible && answer.trigger === 'target_left' && answer.inferred_done === 'yes';
  const satisfied = visible || inferred;
  if (meta.recheck) {
    return {
      completion: satisfied && state.completion === 'checking' ? 'confirmed' : 'recheck_failed',
      step: { kind: 'none' },
      lostNotice: false,
      inferred,
    };
  }
  const completion: ConfirmDecision['completion'] = satisfied && state.completion === 'none' ? 'start_check' : 'keep';

  let step: ConfirmDecision['step'] = { kind: 'none' };
  const { fromIndex, toIndex, stepId } = meta;
  if (
    answer.trigger === 'step_done' &&
    typeof fromIndex === 'number' &&
    typeof toIndex === 'number' &&
    typeof stepId === 'string' &&
    toIndex !== fromIndex &&
    answer.step_id === stepId
  ) {
    if (answer.step_check === 'no' && state.stepIndex === toIndex && !(state.userDoneIds ?? []).includes(stepId)) {
      step = { kind: 'revert', toIndex: fromIndex };
    }
  }

  return {
    completion,
    step,
    lostNotice: (answer.trigger === 'goal_check' || answer.trigger === 'target_left') && !satisfied && state.trackLost,
    inferred,
  };
}

/**
 * A confirm dropped as overtaken (409 stale_plan after a talk/replan moved the plan on) that was the completion
 * re-check: while the run is still `checking`, the re-check is scheduled again (fresh frame, current revision),
 * otherwise `checking` would never resolve — follows and talks only ask for a goal check from `none`.
 */
export function retryRecheckAfterStale(meta: ConfirmMeta | undefined, completion: CompletionState): boolean {
  return Boolean(meta?.recheck) && completion === 'checking';
}

export type ConfirmPositionDecision = { kind: 'none' } | { kind: 'advance'; toIndex: number; skippedIds: string[] };

/**
 * Where a BOUND replan/unsure_twice confirm's `step_checks` puts the screen (pure; the hook applies it).
 *
 * - `replanInstalled` (this answer installed a new plan) → nothing: the checks judged the old plan.
 * - The screen moved while the call was out (`currentIndex !== askedIndex`) → nothing.
 * - `lastYes` = the latest `yes` step at or after the asked one (ids outside the plan are ignored). None → nothing.
 * - Target = the step after `lastYes`, clamped to the last step; already there (the last step `yes` while on
 *   it) → nothing: completion goes only through `goal_status`.
 * - Steps from the asked one up to `lastYes` whose check is not `yes` are `skippedIds` (shown as skipped, the
 *   same meaning as a follow skip): absence is not completion.
 */
export function decideConfirmPosition(
  steps: ReadonlyArray<{ id: string }>,
  askedIndex: number,
  currentIndex: number,
  checks: ReadonlyArray<StepCheck> | null | undefined,
  replanInstalled: boolean
): ConfirmPositionDecision {
  if (replanInstalled || !checks || checks.length === 0) return { kind: 'none' };
  if (!steps[askedIndex] || currentIndex !== askedIndex) return { kind: 'none' };
  let lastYes = -1;
  for (const check of checks) {
    if (check.visible !== 'yes') continue;
    const index = steps.findIndex((step) => step.id === check.step_id);
    if (index >= askedIndex && index > lastYes) lastYes = index;
  }
  if (lastYes < 0) return { kind: 'none' };
  const toIndex = Math.min(lastYes + 1, steps.length - 1);
  if (toIndex === askedIndex) return { kind: 'none' };
  const visible = new Map(checks.map((check) => [check.step_id, check.visible]));
  const skippedIds = steps
    .slice(askedIndex, lastYes)
    .filter((step) => visible.get(step.id) !== 'yes')
    .map((step) => step.id);
  return { kind: 'advance', toIndex, skippedIds };
}
