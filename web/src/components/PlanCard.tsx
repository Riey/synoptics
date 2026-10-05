/** Detailed plan stays available on demand; the camera's current action is the primary instruction. */
import type { IntentView } from '../intent/useIntentLoop';
import { PLAN_STEP_STATE_TEXT, planStepStates } from '../intent/planProgress';
import { materialTitleById } from '../planMode/materials';

export function PlanCard({ view }: { view: IntentView }) {
  const plan = view.plan;
  if (!plan) return null;
  // 완료 ✓ / 사용자 확인 ✓ / 건너뜀 / 현재 / 대기: a skipped step is labelled in words, never shown as done, and a
  // step the user said was done says so instead of claiming the camera saw it.
  const states = planStepStates(plan.steps, view.stepIndex, view.skippedStepIds, view.userDoneStepIds);
  return (
    <details className="card plan-card" data-testid="plan-card" data-plan-revision={plan.planRevision} data-core-mode={view.coreMode}>
      <summary className="plan-summary">전체 단계와 완료 조건 · {plan.steps.length}단계</summary>
      <p className="card-description" data-testid="plan-target">대상: {plan.target}</p>
      <ol className="plan-steps">
        {plan.steps.map((step, index) => {
          const state = states[index];
          const evidenceTitle = step.evidence ? materialTitleById(plan.materials, step.evidence.material_id) : null;
          return (
            <li
              key={`${plan.planRevision}-${step.id}`}
              className={`plan-step plan-step--${state}`}
              data-testid="plan-step"
              data-step-id={step.id}
              data-step-state={state}
              aria-current={state === 'current' ? 'step' : undefined}
            >
              <span className="plan-step-state" data-testid="plan-step-state">{PLAN_STEP_STATE_TEXT[state]}</span>
              <span className="plan-step-say">{step.say}</span>
              <span className="plan-step-done text-secondary">완료 조건: {step.done_when}</span>
              {step.evidence?.quote && (
                <span className="plan-step-done text-secondary">
                  자료 근거: “{step.evidence.quote}”{evidenceTitle ? ` — ${evidenceTitle}` : ''}
                </span>
              )}
            </li>
          );
        })}
      </ol>
      <p className="card-description text-secondary">전체 목표 조건: {plan.goalWhen}</p>
    </details>
  );
}
