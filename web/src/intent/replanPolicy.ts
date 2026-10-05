/**
 * When the client asks the selected upper model to rewrite the plan (design 2026-10-01, §3.D). Pure; the hook applies it.
 *
 * - Stuck: `STUCK_TARGET_CHANGED_FOLLOWS` accepted follows fired by `target_changed` in a row with no step
 *   change in between → confirm('replan') on a fresh capture. The scene keeps changing but the plan does not
 *   move. `heartbeat`/`manual` (and every other trigger) neither count nor reset: a user holding still is not
 *   stuck. A step change resets the count.
 * - Going back is NOT detected automatically: a later step can legitimately undo an earlier step's
 *   `done_when` (open the lid → close it), so a `no` for a passed step is not evidence of going back. The
 *   manual "계획 다시 짜기" button covers it.
 * - Budget: replans (`replan` and `unsure_twice`, stuck or manual) at most `REPLAN_MAX_PER_RUN` per guide run
 *   and never within `REPLAN_MIN_GAP_MS` of the last plan or replan. Inside that, the dispatch gate's paid-call
 *   floor, spacing and budget still apply. A blocked replan only bumps a debug counter.
 * - A replan-capable talk (`replan_allowed: true`) RESERVES a slot while it is out (`reserveReplan`), so no
 *   manual or stuck replan can take the last slot in the meantime; the hook releases it when the call ends
 *   (`releaseReplan`), after the talk reducer counted a replan the answer carried. A replan confirm is checked
 *   again when it leaves (`replanDispatch`): dropped when a replan was installed after it was asked.
 */
export const STUCK_TARGET_CHANGED_FOLLOWS = 3;
/**
 * 3 -> 10 and 15 s -> 8 s on 2026-10-03 (operator): a replan/unsure_twice confirm now also places the step from
 * the large model's own checklist (`decideConfirmPosition`), so a call that does not replan still moves the guide
 * forward instead of being spent; asking more often is worth it. Not yet measured on a live run.
 */
export const REPLAN_MAX_PER_RUN = 10;
export const REPLAN_MIN_GAP_MS = 8_000;

/** One accepted (bound) follow → the next stuck count and whether the guide is now stuck. */
export function nextStuckCount(
  count: number,
  follow: { trigger: string; stepChanged: boolean }
): { count: number; stuck: boolean } {
  if (follow.stepChanged) return { count: 0, stuck: false };
  if (follow.trigger !== 'target_changed') return { count, stuck: false };
  const next = count + 1;
  return next >= STUCK_TARGET_CHANGED_FOLLOWS ? { count: 0, stuck: true } : { count: next, stuck: false };
}

export interface ReplanBudget {
  /** Replans sent in this guide run. */
  used: number;
  /** Slots held by a replan-capable talk that is still out (0 or 1). Absent means 0. */
  reserved?: number;
  /** `performance.now()` of the last plan install, replan request or installed replan. */
  lastPlanAt: number;
}

export type ReplanBlock = 'max' | 'cooldown' | null;

export function freshReplanBudget(planAt: number): ReplanBudget {
  return { used: 0, reserved: 0, lastPlanAt: planAt };
}

/** Whether a replan may be sent now; `waitMs` is how long a cooldown still lasts (0 otherwise). */
export function replanBlock(budget: ReplanBudget, nowMs: number): { block: ReplanBlock; waitMs: number } {
  if (budget.used + (budget.reserved ?? 0) >= REPLAN_MAX_PER_RUN) return { block: 'max', waitMs: 0 };
  const waitMs = budget.lastPlanAt + REPLAN_MIN_GAP_MS - nowMs;
  return waitMs > 0 ? { block: 'cooldown', waitMs } : { block: null, waitMs: 0 };
}

/** A replan was sent at `nowMs`. */
export function spendReplan(budget: ReplanBudget, nowMs: number): ReplanBudget {
  return { ...budget, used: budget.used + 1, lastPlanAt: nowMs };
}

/** A replan-capable talk is about to leave: hold one slot if any is left (`reserved` says whether it got one). */
export function reserveReplan(budget: ReplanBudget): { budget: ReplanBudget; reserved: boolean } {
  const reserved = budget.reserved ?? 0;
  if (budget.used + reserved >= REPLAN_MAX_PER_RUN) return { budget, reserved: false };
  return { budget: { ...budget, reserved: reserved + 1 }, reserved: true };
}

/** The talk that held a slot ended (with or without a replan, which the reducer already counted in `used`). */
export function releaseReplan(budget: ReplanBudget): ReplanBudget {
  return { ...budget, reserved: Math.max(0, (budget.reserved ?? 0) - 1) };
}

/**
 * A replan confirm is about to leave. `askedEpoch` is the run's replan count when it was asked (`undefined` for
 * a call that carries none): a replan installed since makes its question moot. Over the cap is refused too.
 */
export function replanDispatch(budget: ReplanBudget, askedEpoch: number | undefined, currentEpoch: number): 'send' | 'drop' {
  if (askedEpoch !== undefined && askedEpoch !== currentEpoch) return 'drop';
  return budget.used + (budget.reserved ?? 0) > REPLAN_MAX_PER_RUN ? 'drop' : 'send';
}

/** A replanned plan was installed at `nowMs`: the user gets the full gap with it. */
export function notePlanInstalled(budget: ReplanBudget, nowMs: number): ReplanBudget {
  return { ...budget, lastPlanAt: Math.max(budget.lastPlanAt, nowMs) };
}

/** The disabled manual button's reason. */
export function describeReplanBlock(block: ReplanBlock): string | null {
  if (block === 'max') return `이번 가이드에서는 계획을 더 다시 짤 수 없습니다(최대 ${REPLAN_MAX_PER_RUN}회).`;
  if (block === 'cooldown') return `계획을 방금 짰습니다. ${REPLAN_MIN_GAP_MS / 1000}초가 지나면 다시 짤 수 있습니다.`;
  return null;
}
