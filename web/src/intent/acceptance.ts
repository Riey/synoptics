/**
 * The acceptance rule for a follow/confirm answer (design v4 "수락 규칙"), as one pure function.
 *
 * An answer carries the envelope the server echoed (`fence_echo`, `anchors_echo`, `intent_seq`,
 * `plan_revision`). It is compared with the request it answers and with the client's CURRENT state:
 *
 * - `bound`: session, task_epoch, run_id, anchor identity (id, track, generation), plan id + revision and
 *   the newest intent_seq all match → the answer may drive geometry, step changes and the completion path.
 * - `text_only`: session, task_epoch, run_id and plan still match, but the anchor's generation (or track)
 *   moved on → its words may be shown labelled with the capture age, never geometry or completion.
 * - `rejected`: anything else (a different task, run, plan or revision, an older intent_seq, a forged or
 *   mismatched echo) → nothing is shown; only its cost is recorded.
 *
 * Nothing is ever cleared at dispatch; this decides only what an arriving answer may do.
 */
import type { AnchorEcho, GuideFence } from '../generated/api.generated';

export type Acceptance = 'bound' | 'text_only' | 'rejected';

/** What was sent: copied when the request left, never re-read later. */
export interface SentStamp {
  sessionId: string;
  planId: string;
  planRevision: number;
  intentSeq: number;
  fence: GuideFence;
  anchors: AnchorEcho[];
}

/** The client's state when the answer lands. */
export interface CurrentIntentState {
  sessionId: string | null;
  taskEpoch: string | null;
  /** The live run (the store's current run), or null when no run exists. */
  runId: string | null;
  planId: string | null;
  planRevision: number | null;
  /** The newest intent_seq dispatched in this guide. */
  latestIntentSeq: number;
  /** The anchor as the live store describes it now (id bound + its track/generation), or null. */
  anchor: AnchorEcho | null;
}

/** The envelope fields of a follow/confirm response. */
export interface AnswerEnvelope {
  intent_seq: number;
  plan_revision: number;
  fence_echo: GuideFence;
  anchors_echo: AnchorEcho[];
}

export interface AcceptanceOptions {
  /**
   * A confirm answer that applied a replan returns the bumped revision. It is still the answer to THIS
   * plan when the revision went up from the one sent and the client has not moved on since.
   */
  replanApplied?: boolean;
}

function sameFence(a: GuideFence, b: GuideFence): boolean {
  return a.task_epoch === b.task_epoch && a.run_id === b.run_id;
}

function sameAnchors(a: AnchorEcho[], b: AnchorEcho[]): boolean {
  if (a.length !== b.length) return false;
  return a.every(
    (item, index) =>
      item.anchor_id === b[index].anchor_id &&
      item.track_id === b[index].track_id &&
      item.generation === b[index].generation
  );
}

export function classifyAnswer(
  sent: SentStamp,
  answer: AnswerEnvelope,
  current: CurrentIntentState,
  options: AcceptanceOptions = {}
): Acceptance {
  // The echo must be the request's own: a mismatch is not an answer to this request at all.
  if (answer.intent_seq !== sent.intentSeq) return 'rejected';
  if (!sameFence(answer.fence_echo, sent.fence)) return 'rejected';
  if (!sameAnchors(answer.anchors_echo, sent.anchors)) return 'rejected';

  // The question the answer is about must still be the question being asked.
  if (current.sessionId === null || current.sessionId !== sent.sessionId) return 'rejected';
  if (current.taskEpoch === null || current.taskEpoch !== sent.fence.task_epoch) return 'rejected';
  if (current.runId === null || current.runId !== sent.fence.run_id) return 'rejected';
  if (current.planId === null || current.planId !== sent.planId) return 'rejected';
  if (current.planRevision === null || current.planRevision !== sent.planRevision) return 'rejected';
  const revisionOk = options.replanApplied
    ? answer.plan_revision > sent.planRevision
    : answer.plan_revision === sent.planRevision;
  if (!revisionOk) return 'rejected';
  // Only the newest request may drive the screen; an older one would overwrite a newer decision.
  if (sent.intentSeq !== current.latestIntentSeq) return 'rejected';

  // Same task, run and plan. Geometry and completion additionally need the same anchor identity.
  if (sent.anchors.length === 0) return current.anchor === null ? 'bound' : 'text_only';
  const sentAnchor = sent.anchors[0];
  const now = current.anchor;
  if (
    now &&
    now.anchor_id === sentAnchor.anchor_id &&
    now.track_id === sentAnchor.track_id &&
    now.generation === sentAnchor.generation
  ) {
    return 'bound';
  }
  return 'text_only';
}

/** Whole seconds between the answer's capture and now, for the 'N초 전 화면 기준' qualifier (>= 0). */
export function captureAgeSeconds(capturedAtMs: number, nowMs: number): number {
  if (!Number.isFinite(capturedAtMs) || !Number.isFinite(nowMs)) return 0;
  return Math.max(0, Math.round((nowMs - capturedAtMs) / 1000));
}

/** The text-only qualifier the narration prints next to the step sentence. */
export function textOnlyQualifier(capturedAtMs: number, nowMs: number): string {
  return `${captureAgeSeconds(capturedAtMs, nowMs)}초 전 화면 기준`;
}
