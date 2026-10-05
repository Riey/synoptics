/** Compact status and accessible fallback; spatial action cues carry the normal instruction. */
import { useEffect, useState } from 'react';
import { type CompletionState, type IntentView } from '../intent/useIntentLoop';

const STAGE_PENDING_TEXT: Record<string, string> = {
  plan: '그림 안내 준비 중',
  follow: '단계 확인 중',
  confirm: '완료 확인 중',
};
const COMPLETION_TEXT: Record<CompletionState, string | null> = {
  none: null,
  checking: '완료로 보임 · 확인 중',
  confirmed: '완료로 보임(2회 확인)',
  user_confirmed: '완료 확인됨(사용자)',
};

export function formatElapsed(ms: number): string {
  return `${(Math.max(0, ms) / 1000).toFixed(1)}s`;
}

export function NarrationBar({
  view,
  instructionVisible = false,
  visualsEnabled = true,
  onConfirmCompletion,
}: {
  view: IntentView;
  instructionVisible?: boolean;
  visualsEnabled?: boolean;
  onConfirmCompletion?: () => void;
}) {
  const pending = view.pending;
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!pending) return undefined;
    setNow(performance.now());
    const timer = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(timer);
  }, [pending]);

  if (view.phase === 'idle' && !view.notice && !view.error && !view.plan) return null;
  const step = view.plan?.steps[view.stepIndex] ?? null;
  const say = view.phase === 'running' || view.phase === 'planning'
    ? step?.say ?? view.partialFirstSay ?? null : null;
  const streaming = !step && view.partialFirstSay !== null;
  const completion = COMPLETION_TEXT[view.completion];
  const visual = instructionVisible && !view.needsReselect && !view.textOnlyQualifier && !view.error;
  const waitingForDrawing = visualsEnabled && !visual && view.phase === 'running'
    && Boolean(step?.commands.some((command) => command.kind === 'action' || command.kind === 'label'));
  const notice = view.error ?? view.notice;

  return (
    <div
      className="camera-stage-advice narration-bar"
      data-testid="narration-bar"
      data-phase={view.phase}
      data-core-mode={view.coreMode}
      data-step-id={step?.id ?? ''}
      data-completion={view.completion}
      data-presentation={visual ? 'overlay' : 'fallback'}
      data-text-only={view.textOnlyQualifier ? 'true' : 'false'}
    >
      {/* The clock is deliberately outside the live region. Announce instruction changes, not ticks. */}
      <span className="visually-hidden" role="status" aria-live="polite" aria-atomic="true">
        {say ? `${view.textOnlyQualifier || waitingForDrawing ? '위치 확인 전 참고 지시. ' : ''}${view.stepIndex + 1}단계. ${say}` : ''}
        {completion} {notice}
      </span>
      <div className="narration-chips">
        {step && view.phase === 'running' && (
          <span className="narration-chip">{view.stepIndex + 1} / {view.plan?.steps.length} 단계</span>
        )}
        {pending && (
          <span className="narration-chip narration-chip-pending" data-testid="narration-elapsed">
            {STAGE_PENDING_TEXT[pending.stage] ?? '확인 중'} <span aria-hidden="true">{formatElapsed(now - pending.startedAt)}</span>
          </span>
        )}
        {completion && (
          <span className={`narration-chip narration-chip-${view.completion === 'checking' ? 'checking' : 'done'}`} data-testid="narration-completion">
            {completion}
          </span>
        )}
        {view.textOnlyQualifier && (
          <span className="narration-chip narration-chip-stale" data-testid="narration-text-only">{view.textOnlyQualifier}</span>
        )}
        {view.sequentialPending && view.phase === 'running' && (
          <span
            className="narration-chip narration-chip-pending"
            data-testid="core-sequential-waiting"
            data-core-mode={view.coreMode}
            data-waiting-step-id={view.sequentialPending.stepId}
            data-waiting-step-index={view.sequentialPending.stepIndex}
          >
            {view.sequentialPending.stepIndex + 1}단계 상위 확인 대기
          </span>
        )}
      </div>
      {view.partialTarget && !step && (
        <span className="narration-target" data-testid="narration-partial-target">대상: {view.partialTarget}</span>
      )}
      {waitingForDrawing && !view.needsReselect && !notice && (
        <span className="camera-stage-advice-body" data-testid="instruction-wait">그림 위치 확인 중 · 설명은 참고용</span>
      )}
      {say && ((visual || waitingForDrawing) ? (
        <details className="narration-details" key={`${view.plan?.planId}-${step?.id}`} data-testid="instruction-details">
          <summary>설명 보기</summary>
          <p className="narration-say" data-testid="narration-say">{say}</p>
        </details>
      ) : (
        <strong className="camera-stage-advice-headline narration-say" data-testid="narration-say" data-streaming={streaming ? 'true' : 'false'}>
          {say}
        </strong>
      ))}
      {notice && <span className="camera-stage-advice-body" data-testid="narration-notice">{notice}</span>}
      {view.budgetNotice && (
        <span className="camera-stage-advice-body narration-budget" data-testid="narration-budget">{view.budgetNotice}</span>
      )}
      {view.needsReselect && (
        <span className="camera-stage-advice-body narration-warn" data-testid="narration-reselect">대상이 다릅니다. 대상을 다시 지정하세요.</span>
      )}
      {view.completion === 'confirmed' && onConfirmCompletion && (
        <button type="button" className="btn btn-primary narration-confirm" data-testid="guide-confirm-complete" onClick={onConfirmCompletion}>완료 확인</button>
      )}
    </div>
  );
}
