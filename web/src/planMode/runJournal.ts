/**
 * Grounded-run journal: the observable `IntentView` transitions folded into a text-only event log.
 *
 * Pure and idempotent: `snapshotOf` captures what the log diffs, `eventsBetween` turns one transition
 * into events. The component owns the previous snapshot, so a re-render never records twice, and a
 * transition that reports the same stale/unsure check again records nothing (changes only).
 *
 * Nothing here judges a goal: a `no` step check is a screen observation, not an error, and a stop is
 * never labelled a completion.
 */
import type { GuideStep } from '../generated/api.generated';
import type { IntentView } from '../intent/useIntentLoop';
import { materialTitleById } from './materials';

export type JournalEventKind =
  | 'plan'
  | 'replan'
  | 'step'
  | 'skipped'
  | 'user_done'
  | 'check'
  | 'completion'
  | 'talk'
  | 'notice'
  | 'error'
  | 'phase'
  | 'stop';

/**
 * The run a record belongs to, frozen when the event was written (never read back from live props).
 * `grounded` is null while no plan is installed, false for a goal-only plan, true for a plan grounded in
 * reference materials.
 */
export interface JournalProvenance {
  grounded: boolean | null;
  goal: string | null;
  materialTitles: string[] | null;
  planId?: string | null;
  planRevision?: number | null;
}

export interface JournalEventInput {
  kind: JournalEventKind;
  text: string;
  detail?: string;
  /** Defaults to the next snapshot's provenance; a stop names the run it ended instead. */
  provenance?: JournalProvenance;
}

export interface JournalEvent extends JournalEventInput {
  id: number;
  at: number;
  provenance: JournalProvenance;
}

export type CheckVisible = 'yes' | 'no' | 'unsure';

export const CHECK_TEXT: Record<CheckVisible, string> = {
  yes: '조건 보임',
  no: '아직 충족 안 됨',
  unsure: '화면으로 확인 불가',
};

export function checkText(visible: string): string {
  if (visible === 'yes') return CHECK_TEXT.yes;
  if (visible === 'no') return CHECK_TEXT.no;
  if (visible === 'unsure') return CHECK_TEXT.unsure;
  return visible;
}

/** The step's evidence quote (the server only sets evidence for a plan grounded in materials). */
export function stepEvidenceQuote(step: GuideStep | null | undefined): string | null {
  const quote = step?.evidence?.quote;
  return typeof quote === 'string' && quote.trim().length > 0 ? quote : null;
}

export interface RunSnapshot {
  phase: string;
  /** The core the run executes (`classic` / `sequential`), so a record names what produced it. */
  coreMode: string;
  planId: string | null;
  planRevision: number | null;
  planGrounded: boolean | null;
  planTarget: string | null;
  planGoal: string | null;
  planMaterialTitles: string[] | null;
  stepCount: number;
  stepIndex: number;
  stepId: string | null;
  stepSay: string | null;
  stepQuote: string | null;
  stepEvidenceTitle: string | null;
  skipped: string[];
  userDone: string[];
  checks: [string, string][];
  completion: string;
  talkAt: number | null;
  talkUtterance: string | null;
  talkReply: string | null;
  notice: string | null;
  error: string | null;
}

export function snapshotOf(view: IntentView): RunSnapshot {
  const plan = view.plan;
  const step = plan?.steps[view.stepIndex] ?? null;
  const materialId = step?.evidence?.material_id ?? null;
  return {
    phase: view.phase,
    coreMode: view.coreMode,
    planId: plan?.planId ?? null,
    planRevision: plan?.planRevision ?? null,
    planGrounded: plan ? plan.materials !== null : null,
    planTarget: plan?.target ?? null,
    planGoal: plan?.userGoal ?? null,
    planMaterialTitles: plan?.materials?.map((material) => material.title) ?? null,
    stepCount: plan?.steps.length ?? 0,
    stepIndex: view.stepIndex,
    stepId: step?.id ?? null,
    stepSay: step?.say ?? null,
    stepQuote: stepEvidenceQuote(step),
    stepEvidenceTitle: materialId ? materialTitleById(plan?.materials ?? null, materialId) : null,
    skipped: [...view.skippedStepIds],
    userDone: [...view.userDoneStepIds],
    checks: (view.stepChecks ?? []).map((check) => [check.step_id, check.visible]),
    completion: view.completion,
    talkAt: view.talk?.at ?? null,
    talkUtterance: view.talk?.utterance ?? null,
    talkReply: view.talk?.reply ?? null,
    notice: view.notice,
    error: view.error,
  };
}

export function provenanceOf(snapshot: RunSnapshot, active: boolean): JournalProvenance {
  return {
    grounded: snapshot.planGrounded ?? (active ? true : null),
    goal: snapshot.planGoal,
    materialTitles: snapshot.planMaterialTitles,
    planId: snapshot.planId,
    planRevision: snapshot.planRevision,
  };
}

/**
 * Whether one transition belongs in this record. While the reference flow is selected, a goal-only run is
 * not this record's business; while it is not, only a grounded plan's transitions (including its
 * retirement) are kept, so the history survives a switch while a goal-only run never pollutes it.
 */
export function shouldRecordTransition(previous: RunSnapshot, next: RunSnapshot, active: boolean): boolean {
  return active
    ? previous.planGrounded !== false && next.planGrounded !== false
    : previous.planGrounded === true || next.planGrounded === true;
}

interface StepLike {
  id: string;
  say: string;
}

function joinDetail(parts: Array<string | null>): string | undefined {
  const kept = parts.filter((part): part is string => part !== null && part.length > 0);
  return kept.length > 0 ? kept.join(' · ') : undefined;
}

export function eventsBetween(
  previous: RunSnapshot,
  next: RunSnapshot,
  steps: ReadonlyArray<StepLike>,
  active: boolean
): JournalEventInput[] {
  const events: JournalEventInput[] = [];
  const sayOf = (id: string): string => steps.find((step) => step.id === id)?.say ?? id;
  const indexOf = (id: string): number => steps.findIndex((step) => step.id === id);
  const provOf = (snapshot: RunSnapshot): JournalProvenance => provenanceOf(snapshot, active);

  const planChanged = next.planId !== previous.planId || next.planRevision !== previous.planRevision;
  const errorChanged = Boolean(next.error && next.error !== previous.error);
  const planRemoved = planChanged && next.planId === null && previous.planId !== null;

  if (planChanged) {
    if (planRemoved) {
      // A stop is terminal evidence, never a completion; the error, when present, is recorded separately.
      if (!errorChanged) {
        events.push({
          kind: 'stop',
          text: previous.planGrounded === true ? '참고자료 안내가 정지되었습니다 (완료 아님).' : '안내가 정지되었습니다 (완료 아님).',
          provenance: provOf(previous),
        });
      }
    } else if (next.planId !== null) {
      const first = previous.planId === null;
      const prefix = next.planGrounded === true ? '참고자료 ' : '';
      events.push({
        kind: first ? 'plan' : 'replan',
        text: first
          ? `${prefix}계획 수립 · ${next.stepCount}단계 (rev ${next.planRevision ?? 0})`
          : `${prefix}계획 수정 (rev ${next.planRevision ?? 0}) · ${next.stepCount}단계`,
        detail: joinDetail([
          `코어: ${next.coreMode === 'sequential' ? '순차' : '기존'}`,
          next.planTarget ? `대상: ${next.planTarget}` : null,
          next.planGoal ? `목표: ${next.planGoal}` : null,
        ]),
        provenance: provOf(next),
      });
    }
  }

  if (next.stepId !== null && next.stepId !== previous.stepId) {
    const reverted = previous.stepId !== null && next.stepIndex < previous.stepIndex && next.planId === previous.planId;
    events.push({
      kind: 'step',
      text: reverted
        ? `${next.stepIndex + 1}단계로 되돌아감: ${next.stepSay ?? ''}`.trim()
        : `${next.stepIndex + 1}/${next.stepCount}단계: ${next.stepSay ?? ''}`.trim(),
      detail: next.stepQuote
        ? `자료 근거: “${next.stepQuote}”${next.stepEvidenceTitle ? ` — ${next.stepEvidenceTitle}` : ''}`
        : undefined,
      provenance: provOf(next),
    });
  }

  const skippedAdded = next.skipped.filter((id) => !previous.skipped.includes(id));
  if (skippedAdded.length > 0) {
    events.push({
      kind: 'skipped',
      text: `건너뜀 ${skippedAdded.length}개: ${skippedAdded.map(sayOf).join(' / ')}`,
      detail: '필수 조건이 화면에서 확인되지 않았거나 순서가 불확실할 수 있습니다. 위반을 관찰한 것은 아닙니다.',
      provenance: provOf(next),
    });
  }

  const doneAdded = next.userDone.filter((id) => !previous.userDone.includes(id));
  if (doneAdded.length > 0) {
    events.push({
      kind: 'user_done',
      text: `사용자 확인: ${doneAdded.map(sayOf).join(' / ')}`,
      provenance: provOf(next),
    });
  }

  const previousChecks = new Map(previous.checks);
  for (const [stepId, visible] of next.checks) {
    if (previousChecks.get(stepId) === visible) continue;
    const index = indexOf(stepId);
    events.push({
      kind: 'check',
      text: `${index >= 0 ? `${index + 1}단계` : `단계 ${stepId}`} 조건 검사: ${checkText(visible)}`,
      detail: visible === 'no' ? '화면에서 확인되지 않았다는 뜻이며, 그 자체로 오류나 위반은 아닙니다.' : undefined,
      provenance: provOf(next),
    });
  }

  if (next.completion !== previous.completion) {
    if (next.completion === 'confirmed') {
      events.push({
        kind: 'completion',
        text: '화면 판정: 완료로 보임(2회 확인)',
        detail: '화면 기반 판정이며, 참고자료 전체 준수를 보증하지 않습니다.',
        provenance: provOf(next),
      });
    } else if (next.completion === 'user_confirmed') {
      events.push({ kind: 'completion', text: '사용자 완료 확인', provenance: provOf(next) });
    } else if (next.completion === 'checking') {
      events.push({ kind: 'completion', text: '완료 여부 확인 중', provenance: provOf(next) });
    } else if (next.completion === 'none' && previous.completion !== 'none' && !planRemoved && !errorChanged) {
      // A reset that is part of a stop or an error is not a judgement change; those are recorded above.
      events.push({ kind: 'completion', text: '완료 판정이 해제되었습니다. 안내를 계속합니다.', provenance: provOf(next) });
    }
  }

  if (next.talkAt !== null && next.talkAt !== previous.talkAt) {
    events.push({
      kind: 'talk',
      text: `나: ${next.talkUtterance ?? ''}`.trim(),
      detail: next.talkReply ? `안내: ${next.talkReply}` : undefined,
      provenance: provOf(next),
    });
  }

  if (next.notice && next.notice !== previous.notice) {
    events.push({ kind: 'notice', text: next.notice, provenance: provOf(next) });
  }

  if (errorChanged && next.error) {
    events.push({ kind: 'error', text: `오류: ${next.error}`, provenance: provOf(planRemoved ? previous : next) });
  }

  if (next.phase !== previous.phase) {
    if (next.phase === 'planning') {
      events.push({ kind: 'phase', text: '계획을 준비하는 중입니다.', provenance: provOf(next) });
    } else if (next.phase === 'running' && previous.phase !== 'running') {
      events.push({ kind: 'phase', text: '안내를 시작했습니다.', provenance: provOf(next) });
    } else if (next.phase === 'completed' && next.completion === previous.completion) {
      events.push({ kind: 'phase', text: '안내 루프가 종료되었습니다.', provenance: provOf(next) });
    } else if (next.phase === 'idle' && previous.phase !== 'idle' && !planRemoved && !errorChanged) {
      events.push({ kind: 'stop', text: '안내가 정지되었습니다 (완료 아님).', provenance: provOf(previous) });
    } else if (next.phase === 'error' && !errorChanged) {
      events.push({ kind: 'error', text: '오류로 중단되었습니다.', provenance: provOf(next) });
    }
  }

  return events;
}

export interface JournalNote {
  at: number;
  text: string;
  provenance: JournalProvenance;
}

export interface RunRecordExport {
  app: 'synoptics-plan-run-record';
  version: 2;
  exported_at: string;
  run: JournalProvenance;
  journal_truncated: boolean;
  events: Array<{
    at: string;
    kind: JournalEventKind;
    text: string;
    detail?: string;
    grounded: boolean | null;
    goal: string | null;
    material_titles: string[] | null;
    plan_id: string | null;
    plan_revision: number | null;
  }>;
  notes: Array<{ at: string; text: string; provenance: JournalProvenance }>;
}

export function buildRunRecordExport(
  events: ReadonlyArray<JournalEvent>,
  notes: ReadonlyArray<JournalNote>,
  run: JournalProvenance,
  journalTruncated: boolean
): RunRecordExport {
  return {
    app: 'synoptics-plan-run-record',
    version: 2,
    exported_at: new Date().toISOString(),
    run,
    journal_truncated: journalTruncated,
    events: events.map((event) => {
      const row: RunRecordExport['events'][number] = {
        at: new Date(event.at).toISOString(),
        kind: event.kind,
        text: event.text,
        grounded: event.provenance.grounded,
        goal: event.provenance.goal,
        material_titles: event.provenance.materialTitles,
        plan_id: event.provenance.planId ?? null,
        plan_revision: event.provenance.planRevision ?? null,
      };
      if (event.detail) row.detail = event.detail;
      return row;
    }),
    notes: notes.map((note) => ({ at: new Date(note.at).toISOString(), text: note.text, provenance: note.provenance })),
  };
}
