/**
 * What a bound follow checklist may do to the step (pure; the hook applies it). Design 2026-10-01 §3.C.
 *
 * The follower answers, for every step from the asked one (`current_step` at capture) to the last, whether
 * that step's `done_when` is visible now (`yes`/`no`/`unsure`). It does not choose the step; this does.
 *
 * - The screen moved while the call was in flight (`currentIndex !== askedIndex`) → nothing changes.
 * - `lastYes` = the latest step whose check is `yes`. None → nothing changes. `target` = the step after it,
 *   clamped to the last step.
 * - Adjacent advance (`lastYes === askedIndex`): one follow is enough → `advance` to `target`; the hook
 *   re-checks the asked step with confirm(step_done) and reverts on `step_check: no`.
 * - Skip (`lastYes > askedIndex`): only when the PREVIOUS accepted follow also showed that step (or a later
 *   one) `yes` → `skip` to `target`, re-checked by confirm(step_done) for the `lastYes` step. A first sighting
 *   is `pendingSkip` (recorded, the screen stays). Steps between the asked one and `lastYes` whose check is
 *   not `yes` are `skippedIds`: shown as skipped, never as done (absence is not completion).
 * - The last step `yes` while on it changes nothing: completion goes only through `goal_seen: yes` →
 *   confirm(goal_check).
 * - `unsure` and `no` never move the screen, forward or back.
 */
export type Visible = 'yes' | 'no' | 'unsure';

export interface StepCheck {
  step_id: string;
  visible: Visible;
}

export type FollowStepDecision =
  | { kind: 'none' }
  | { kind: 'pendingSkip'; lastYesIndex: number }
  | { kind: 'advance'; fromIndex: number; toIndex: number; stepId: string }
  | { kind: 'skip'; fromIndex: number; toIndex: number; stepId: string; skippedIds: string[] };

type Steps = ReadonlyArray<{ id: string }>;

/** Index of the latest `yes` step at or after `fromIndex` (ids outside the plan are ignored), or -1. */
function lastYesIndex(steps: Steps, checks: ReadonlyArray<StepCheck>, fromIndex: number): number {
  let last = -1;
  for (const check of checks) {
    if (check.visible !== 'yes') continue;
    const index = steps.findIndex((step) => step.id === check.step_id);
    if (index >= fromIndex && index > last) last = index;
  }
  return last;
}

/** The asked step's own entry in the checklist (null when the answer has none for it). */
export function currentStepVisible(steps: Steps, askedIndex: number, checks: ReadonlyArray<StepCheck>): Visible | null {
  const asked = steps[askedIndex];
  if (!asked) return null;
  return checks.find((check) => check.step_id === asked.id)?.visible ?? null;
}

export function decideFollowStep(
  steps: Steps,
  askedIndex: number,
  currentIndex: number,
  checks: ReadonlyArray<StepCheck>,
  previousChecks: ReadonlyArray<StepCheck> | null
): FollowStepDecision {
  const asked = steps[askedIndex];
  if (!asked || currentIndex !== askedIndex) return { kind: 'none' };
  const lastYes = lastYesIndex(steps, checks, askedIndex);
  if (lastYes < 0) return { kind: 'none' };
  const target = Math.min(lastYes + 1, steps.length - 1);
  if (lastYes === askedIndex) {
    if (target === askedIndex) return { kind: 'none' };
    return { kind: 'advance', fromIndex: askedIndex, toIndex: target, stepId: asked.id };
  }
  const seenBefore = previousChecks !== null && lastYesIndex(steps, previousChecks, lastYes) >= lastYes;
  if (!seenBefore) return { kind: 'pendingSkip', lastYesIndex: lastYes };
  const visible = new Map(checks.map((check) => [check.step_id, check.visible]));
  const skippedIds = steps
    .slice(askedIndex, lastYes)
    .filter((step) => visible.get(step.id) !== 'yes')
    .map((step) => step.id);
  return { kind: 'skip', fromIndex: askedIndex, toIndex: target, stepId: steps[lastYes].id, skippedIds };
}
