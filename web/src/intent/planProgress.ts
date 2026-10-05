/**
 * The plan card's per-step state (design 2026-10-01 §3.F, guide-talk §D): done ✓ / user-confirmed ✓ /
 * skipped / current / pending. Pure.
 *
 * Steps before the current one are done unless the user said they were done (talk `step_mark: done`, shown in
 * words as the user's confirmation) or the client skipped them (a skip never marks a step whose `done_when` was
 * not seen as done: absence is not completion). A revert or go-back forgets marks and skips at or after the
 * step it returns to; a replan starts with none.
 */
export type PlanStepState = 'done' | 'user_done' | 'skipped' | 'current' | 'pending';

export const PLAN_STEP_STATE_TEXT: Record<PlanStepState, string> = {
  done: '완료 ✓',
  user_done: '사용자 확인 ✓',
  skipped: '건너뜀',
  current: '현재',
  pending: '대기',
};

type Steps = ReadonlyArray<{ id: string }>;

export function planStepStates(
  steps: Steps,
  stepIndex: number,
  skippedIds: ReadonlyArray<string>,
  userDoneIds: ReadonlyArray<string> = []
): PlanStepState[] {
  const skipped = new Set(skippedIds);
  const userDone = new Set(userDoneIds);
  return steps.map((step, index) =>
    index === stepIndex
      ? 'current'
      : index > stepIndex
        ? 'pending'
        : userDone.has(step.id)
          ? 'user_done'
          : skipped.has(step.id)
            ? 'skipped'
            : 'done'
  );
}

/** `skipped` plus `ids`, without duplicates. */
export function addSkipped(skipped: ReadonlyArray<string>, ids: ReadonlyArray<string>): string[] {
  return [...new Set([...skipped, ...ids])];
}

/** The skipped ids of steps strictly before `index` (what survives a revert to `index`). */
export function skippedBefore(steps: Steps, skipped: ReadonlyArray<string>, index: number): string[] {
  const before = new Set(steps.slice(0, Math.max(0, index)).map((step) => step.id));
  return skipped.filter((id) => before.has(id));
}
