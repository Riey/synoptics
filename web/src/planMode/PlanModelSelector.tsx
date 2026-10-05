import {
  PLAN_MODELS,
  PLAN_MODEL_OPTION,
  PLAN_MODEL_STATE_TEXT,
  planModelBlockedReason,
  planModelState,
  type PlanModel,
  type PlanModelAvailability,
} from './modelChoice';

export interface PlanModelSelectorProps {
  /** The chosen model; frozen into the next run Start creates and sent as `plan_model` on its plan request. */
  value: PlanModel;
  /** `health.plan_models` (null while it is unknown). Drives the per-option state and the block wording. */
  availability: PlanModelAvailability;
  /** True while a guide run is planning or running: the choice is frozen for that run. */
  disabled: boolean;
  onChange: (model: PlanModel) => void;
}

/**
 * The plan model picker: one native radio group in a labelled fieldset, both options always visible, each
 * showing whether the server has its key. A model without a key is stated in words and holds Start — the
 * other model is never used in its place, so a missing Astra key can never silently route to DeepSeek.
 */
export function PlanModelSelector({ value, availability, disabled, onChange }: PlanModelSelectorProps) {
  const blocked = planModelBlockedReason(availability, value);
  return (
    <fieldset className="mode-picker plan-model-picker" data-testid="plan-model-picker" disabled={disabled}>
      <legend className="form-label">계획 모델</legend>
      {PLAN_MODELS.map((model) => {
        const state = planModelState(availability, model);
        return (
          <label key={model} data-testid="plan-model-option" data-model={model} data-state={state}>
            <input
              type="radio"
              name="plan-model"
              value={model}
              checked={value === model}
              onChange={() => onChange(model)}
              aria-describedby="plan-model-note"
            />
            <span>{PLAN_MODEL_OPTION[model].label}</span>
            <span className="plan-model-status" aria-hidden="true">
              {PLAN_MODEL_STATE_TEXT[state]}
            </span>
          </label>
        );
      })}
      <p className="plan-model-note" id="plan-model-note" data-testid="plan-model-note" role="status">
        {blocked ?? `${PLAN_MODEL_OPTION[value].label}로 계획을 만듭니다.`}
      </p>
    </fieldset>
  );
}
