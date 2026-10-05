/**
 * Grounded-run record: a page-lifetime, text-only log of what the demo guide actually did, plus user
 * feedback.
 *
 * The journal diffs `IntentView` transitions through the pure `runJournal` helpers, so a re-render never
 * records twice. Each event keeps the provenance it was written with (grounding/goal/material titles), so
 * a later stop, mode switch, or source edit cannot re-attribute old evidence to a new source. The current
 * step's evidence quote (and the material it came from) is shown only while the live materials and goal
 * still match the installed plan.
 *
 * No images, no autosending, no localStorage: everything lives in this component's state. This is the
 * demo's reference-based guidance record — the Live PWA owns the actual Plan review/approval flow.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { MaterialInput } from '../generated/api.generated';
import type { IntentView } from '../intent/useIntentLoop';
import { materialTextById, materialTitleById, quoteInMaterial, sameMaterials } from './materials';
import {
  buildRunRecordExport,
  checkText,
  eventsBetween,
  provenanceOf,
  shouldRecordTransition,
  snapshotOf,
  stepEvidenceQuote,
  type JournalEvent,
  type JournalEventInput,
  type JournalNote,
  type JournalProvenance,
  type RunSnapshot,
} from './runJournal';
import './plan-mode.css';

/** A page-lifetime cap: far beyond any real run, but keeps an idle tab bounded. */
const JOURNAL_MAX_EVENTS = 500;
const NOTE_MAX = 500;
const NOTE_MAX_COUNT = 200;

function formatClock(at: number): string {
  const date = new Date(at);
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function formatStamp(at: number): string {
  const date = new Date(at);
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

export function PlanRunRecord({
  view,
  materials,
  goal,
  active,
}: {
  view: IntentView;
  materials: readonly MaterialInput[];
  goal: string;
  active: boolean;
}) {
  const [events, setEvents] = useState<JournalEvent[]>([]);
  const [notes, setNotes] = useState<JournalNote[]>([]);
  const [noteDraft, setNoteDraft] = useState('');

  const previousRef = useRef<RunSnapshot | null>(null);
  const nextIdRef = useRef(1);
  const activeRef = useRef(active);
  activeRef.current = active;

  const push = useCallback((inputs: JournalEventInput[]) => {
    if (inputs.length === 0) return;
    const at = Date.now();
    const added: JournalEvent[] = inputs.map((input) => ({
      kind: input.kind,
      text: input.text,
      detail: input.detail,
      provenance: input.provenance ?? { grounded: null, goal: null, materialTitles: null },
      id: nextIdRef.current++,
      at,
    }));
    setEvents((current) => {
      const combined = [...current, ...added];
      return combined.length > JOURNAL_MAX_EVENTS ? combined.slice(combined.length - JOURNAL_MAX_EVENTS) : combined;
    });
  }, []);

  // The diff runs once per `view` object (React state identity), never per render; `push` is idempotent
  // for a stale snapshot, so React's StrictMode double-mount records nothing twice.
  useEffect(() => {
    const next = snapshotOf(view);
    const previous = previousRef.current;
    previousRef.current = next;
    if (!previous) {
      // Mounted mid-run (Main mounts continuously, so normally this is the idle snapshot): capture the
      // in-flight plan once, under the same grounding gate as every other transition.
      if (next.planId !== null && (next.planGrounded === true || (activeRef.current && next.planGrounded !== false))) {
        push([
          {
            kind: 'plan',
            text: `${next.planGrounded === true ? '참고자료 ' : ''}계획 (기록 시작 시점) · ${next.stepCount}단계 (rev ${next.planRevision ?? 0})`,
            detail: next.planTarget ? `대상: ${next.planTarget}` : undefined,
            provenance: provenanceOf(next, activeRef.current),
          },
        ]);
      }
      return;
    }
    // While the reference flow is selected, a goal-only run is not this record's business; while it is not,
    // only a grounded plan's transitions (including its retirement) are kept.
    if (!shouldRecordTransition(previous, next, activeRef.current)) return;
    push(eventsBetween(previous, next, view.plan?.steps ?? [], activeRef.current));
  }, [view, push]);

  const plan = view.plan;
  const planMaterials = plan?.materials ?? null;
  const step = plan ? (plan.steps[view.stepIndex] ?? null) : null;
  const evidence = step?.evidence ?? null;
  const quote = stepEvidenceQuote(step);
  const materialText = evidence ? materialTextById(planMaterials, evidence.material_id) : null;
  const materialTitle = evidence ? materialTitleById(planMaterials, evidence.material_id) : null;
  const materialsMatch = sameMaterials(planMaterials, materials);
  const goalMatches = plan !== null && plan.userGoal === goal.trim();
  const quoteGrounded =
    planMaterials !== null &&
    materialsMatch &&
    goalMatches &&
    quote !== null &&
    materialText !== null &&
    quoteInMaterial(quote, materialText);
  const quoteSuppressed = planMaterials !== null && (!materialsMatch || !goalMatches);
  const currentCheck = (view.stepChecks ?? []).find((check) => check.step_id === step?.id) ?? null;
  const skippedSteps = useMemo(() => {
    if (!plan) return [];
    const skipped = new Set(view.skippedStepIds);
    return plan.steps.filter((planStep) => skipped.has(planStep.id));
  }, [plan, view.skippedStepIds]);

  if (!active) return null;

  const newest = events.length > 0 ? events[events.length - 1] : null;
  const truncationNote = events.length >= JOURNAL_MAX_EVENTS;
  const runProvenance: JournalProvenance = plan
    ? {
        grounded: planMaterials !== null,
        goal: plan.userGoal,
        materialTitles: planMaterials?.map((material) => material.title) ?? null,
        planId: plan.planId,
        planRevision: plan.planRevision,
      }
    : newest
      ? newest.provenance
      : { grounded: null, goal: null, materialTitles: null };

  const addNote = () => {
    const text = noteDraft.trim();
    if (!text) return;
    setNotes((current) => (current.length >= NOTE_MAX_COUNT ? current : [...current, { at: Date.now(), text, provenance: runProvenance }]));
    setNoteDraft('');
  };

  const exportRecord = () => {
    const payload = buildRunRecordExport(events, notes, runProvenance, truncationNote);
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `synoptics-plan-run-${formatStamp(Date.now())}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const clearRecord = () => {
    setEvents([]);
    setNotes([]);
    setNoteDraft('');
  };

  return (
    <section className="card plan-mode-record" data-testid="plan-mode-run-record" data-grounded={planMaterials !== null ? 'true' : 'false'}>
      <div className="card-header-flex">
        <h2 className="card-title">참고자료 안내 기록 (데모)</h2>
        <span className="plan-mode-count">
          기록 {events.length}개 · 메모 {notes.length}개
        </span>
      </div>
      <p className="card-description">
        계획 수립·단계 이동·건너뜀·조건 검사·대화·오류를 시각과 함께 이 페이지에만 기록합니다(자동 전송·저장 없음).
      </p>
      <p className="card-description text-secondary" data-testid="plan-mode-record-demo-notice">
        이 기록은 이 데모 페이지의 참고자료 안내용입니다. 계획을 검토하고 승인·수정하는 실제 Plan 흐름은 Live PWA에서
        진행합니다.
      </p>
      <p className="visually-hidden" role="status" aria-live="polite">
        {newest ? newest.text : ''}
      </p>

      {planMaterials !== null && step && (
        <div className="plan-mode-current" data-testid="plan-mode-current-step">
          <p className="form-label">{view.phase === 'completed' ? '마지막 안내 단계와 자료 근거' : '현재 단계와 자료 근거'}</p>
          <p className="plan-mode-current__say">{step.say}</p>
          {quoteGrounded ? (
            <>
              <blockquote className="plan-mode-quote" data-testid="plan-mode-current-quote">
                &ldquo;{quote}&rdquo;
              </blockquote>
              <p className="plan-mode-count" data-testid="plan-mode-current-source">
                출처: {materialTitle ?? evidence?.material_id}
                {evidence?.version ? ` (${evidence.version})` : ''}
                {evidence?.locator ? ` · ${evidence.locator}` : ''}
              </p>
            </>
          ) : quoteSuppressed ? (
            <p className="plan-mode-count" data-testid="plan-mode-quote-suppressed">
              참고자료나 목표가 바뀌어 이전 계획의 인용은 표시하지 않습니다. 다시 시작하면 새 자료로 계획합니다.
            </p>
          ) : (
            <p className="plan-mode-count">이 단계에는 자료 인용이 없습니다.</p>
          )}
          {currentCheck && (
            <p className="plan-mode-check" data-testid="plan-mode-current-check" data-visible={currentCheck.visible}>
              현재 단계 조건 검사: <strong>{checkText(currentCheck.visible)}</strong>
              {currentCheck.visible === 'no' && (
                <span className="plan-mode-count"> — 화면에서 확인되지 않았다는 뜻이며 오류가 아닙니다.</span>
              )}
            </p>
          )}
        </div>
      )}

      {skippedSteps.length > 0 && (
        <div className="alert-card alert-warning" role="status" data-testid="plan-mode-skipped-warning">
          <span className="alert-icon" aria-hidden="true">!</span>
          <span className="alert-message">
            건너뜀 {skippedSteps.length}개: {skippedSteps.map((planStep) => planStep.say).join(' / ')} — 필수 조건이 화면에서
            확인되지 않았거나 순서가 불확실할 수 있습니다. 위반을 관찰한 것은 아닙니다.
          </span>
        </div>
      )}

      <details className="disclosure plan-mode-history" data-testid="plan-mode-history">
        <summary className="disclosure-summary">진행 이력 · 피드백 · 내보내기</summary>
      {events.length === 0 ? (
        <p className="card-description" data-testid="plan-mode-record-empty">
          아직 기록이 없습니다. “가이드 시작”을 누르면 단계 변화가 여기에 쌓입니다.
        </p>
      ) : (
        <ol className="plan-mode-events" data-testid="plan-mode-event-list">
          {events
            .slice()
            .reverse()
            .map((event) => (
              <li key={event.id} className={`plan-mode-event plan-mode-event--${event.kind}`} data-kind={event.kind}>
                <span className="plan-mode-event__time">{formatClock(event.at)}</span>
                <span className="plan-mode-event__body">
                  <span className="plan-mode-event__text">{event.text}</span>
                  {event.detail && <span className="plan-mode-event__detail">{event.detail}</span>}
                </span>
              </li>
            ))}
        </ol>
      )}
      {truncationNote && (
        <p className="plan-mode-count">기록이 최대 {JOURNAL_MAX_EVENTS}개로 제한되어 오래된 항목은 표시되지 않습니다.</p>
      )}

      <div className="form-group">
        <label className="form-label" htmlFor="plan-mode-note-input">
          피드백 메모 추가
        </label>
        <textarea
          id="plan-mode-note-input"
          className="input-text"
          rows={2}
          maxLength={NOTE_MAX}
          value={noteDraft}
          placeholder="예: 3단계 순서가 참고자료와 달랐음 · 화면에 없는 도구를 먼저 안내함"
          onChange={(event) => setNoteDraft(event.target.value)}
        />
        <div className="plan-mode-row">
          <button
            type="button"
            className="btn btn-secondary"
            data-testid="plan-mode-note-add"
            disabled={!noteDraft.trim() || notes.length >= NOTE_MAX_COUNT}
            onClick={addNote}
          >
            메모 추가
          </button>
          <span className="plan-mode-count">
            {noteDraft.length} / {NOTE_MAX}자
          </span>
        </div>
      </div>

      {notes.length > 0 && (
        <ol className="plan-mode-notes" data-testid="plan-mode-note-list">
          {notes
            .slice()
            .reverse()
            .map((note) => (
              <li key={`${note.at}-${note.text}`} className="plan-mode-note">
                <span className="plan-mode-event__time">{formatClock(note.at)}</span>
                <span>{note.text}</span>
              </li>
            ))}
        </ol>
      )}

      <div className="button-group">
        <button
          type="button"
          className="btn btn-secondary"
          data-testid="plan-mode-record-export"
          disabled={events.length === 0 && notes.length === 0}
          onClick={exportRecord}
        >
          기록 내보내기 (JSON)
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          data-testid="plan-mode-record-clear"
          disabled={events.length === 0 && notes.length === 0}
          onClick={clearRecord}
        >
          기록 지우기
        </button>
      </div>
      </details>
    </section>
  );
}
