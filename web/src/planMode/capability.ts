/**
 * What plan execution policies this standalone demo can actually honor.
 *
 * The Live server owns `plan_policy`: it gates manual checks, procedural requirements, dependency graphs,
 * revocable state/event evidence and goal-vs-safety conditions. This demo has no such machine, so those
 * policies are refused outright instead of silently executing them as ordinary linear visual steps.
 * An ordinary all-visual, ungated plan runs as before.
 */
import type { GuideStep, MaterialInput } from '../generated/api.generated';

export const UNSUPPORTED_PLAN_NOTICE = '이 계획은 단계 확인·선행 조건을 지원하는 Live Plan 클라이언트에서 실행하세요.';

/** Whether the plan uses an execution policy this demo cannot run (anything beyond plain visual steps). */
export function hasUnsupportedExecution(steps: ReadonlyArray<GuideStep>): boolean {
  return steps.some(
    (step) => (step.check ?? 'visual') !== 'visual' || step.required === true || (step.requires?.length ?? 0) > 0
      || step.goal_required === false || step.condition_kind === 'event'
  );
}

/** The refusal notice, naming the reference materials so a blocked run's source is not silently lost. */
export function unsupportedPlanNotice(materials: ReadonlyArray<MaterialInput> | null): string {
  if (!materials || materials.length === 0) return UNSUPPORTED_PLAN_NOTICE;
  return `${UNSUPPORTED_PLAN_NOTICE} (참고자료: ${materials.map((material) => material.title).join(', ')})`;
}
