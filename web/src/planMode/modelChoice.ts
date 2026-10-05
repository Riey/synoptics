/**
 * The plan model chosen before starting a guide run (design: selectable `deepseek:high` vs `astra:high`).
 *
 * Both profiles plan with high reasoning; the only difference is which upper provider writes the plan. The
 * choice is frozen into the run that Start creates and sent as `plan_model` on that run's first plan
 * request only — confirm/replan/talk reuse the model stored with the accepted plan. A model whose provider
 * has no key is reported `unavailable` and holds Start; the client never silently plans on the other model.
 */

export const PLAN_MODELS = ['deepseek:high', 'astra:high'] as const;
export type PlanModel = (typeof PLAN_MODELS)[number];
/** The server's default; the control starts here and never invents a third option. */
export const DEFAULT_PLAN_MODEL: PlanModel = 'deepseek:high';

export interface PlanModelOption {
  /** Short provider name, used inside sentences about which model will plan. */
  provider: string;
  /** The option's own label. */
  label: string;
}

/** One entry per model: the closed set of choices and how each is named. */
export const PLAN_MODEL_OPTION: Record<PlanModel, PlanModelOption> = {
  'deepseek:high': { provider: 'DeepSeek', label: 'DeepSeek · high' },
  'astra:high': { provider: 'Astra', label: 'Astra · high' },
};

/**
 * Per-model readiness from `/api/health` (`plan_models`). `null` means the server did not report it at
 * all: no model is then assumed available from a missing entry.
 */
export type PlanModelAvailability = Record<PlanModel, boolean> | null;

/**
 * Read `plan_models` defensively: only the `true` a server actually sent counts as available, so a partial
 * or malformed payload can never enable a model that has no key.
 */
export function readPlanModelAvailability(raw: unknown): PlanModelAvailability {
  if (raw === null || typeof raw !== 'object') return null;
  const source = raw as Record<string, unknown>;
  return {
    'deepseek:high': source['deepseek:high'] === true,
    'astra:high': source['astra:high'] === true,
  };
}

export type PlanModelState = 'ready' | 'unavailable' | 'unknown';

export function planModelState(availability: PlanModelAvailability, model: PlanModel): PlanModelState {
  if (availability === null) return 'unknown';
  return availability[model] ? 'ready' : 'unavailable';
}

/** The one-word state shown next to an option. */
export const PLAN_MODEL_STATE_TEXT: Record<PlanModelState, string> = {
  ready: '사용 가능',
  unavailable: '키 미설정',
  unknown: '확인 중',
};

/**
 * Why the chosen model cannot start a run, or `null` when it can. The unavailable wording says the other
 * model is not used instead, so the block can never read as an automatic switch.
 */
export function planModelBlockedReason(availability: PlanModelAvailability, model: PlanModel): string | null {
  switch (planModelState(availability, model)) {
    case 'ready':
      return null;
    case 'unavailable':
      return `선택한 계획 모델(${PLAN_MODEL_OPTION[model].provider})의 키가 서버에 설정되지 않았습니다. 다른 계획 모델을 고르거나 서버 설정을 확인해 주세요. 다른 모델로 자동 전환하지 않습니다.`;
    case 'unknown':
      return '계획 모델 준비 상태를 확인하지 못했습니다. 새로고침해 주세요.';
  }
}
