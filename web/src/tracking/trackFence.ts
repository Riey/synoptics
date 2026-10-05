/**
 * Pure fence predicates for the tracker's paired-frame flow (contract v4 §2/§3/§4).
 *
 * The tracker answers one specific frame (`frame_id` + `frame_seq`) of one specific run
 * (`run_id`), each answer carrying a per-run `version`. These functions decide whether a
 * response may be applied to the panel, how an error maps onto run/frame state, and when a
 * scene change invalidates a run. They deliberately take plain fields instead of the wire
 * types: the generated contract types stay the single source of truth at the API boundary,
 * and this logic stays dependency-free and unit-testable.
 */

/** Why an update was accepted or dropped. The reason is surfaced in diagnostics only. */
export type TrackFenceReason = 'accepted' | 'no_run' | 'stale_run' | 'frame_mismatch' | 'stale_version';

export interface UpdateIdentity {
  runId: string;
  frameId: string;
  frameSeq: number;
  version: number;
}

export interface FenceState {
  /** The run the panel currently owns, or null when no run is active. */
  activeRunId: string | null;
  /** The exact frame whose answer the panel is waiting for, or null when none is in flight. */
  pairedFrameId: string | null;
  pairedFrameSeq: number | null;
  /** Highest version applied so far for `activeRunId`. */
  lastVersion: number;
}

export interface FenceDecision {
  accept: boolean;
  reason: TrackFenceReason;
}

/**
 * Decide whether a frame response may be applied.
 *
 * v4 §3: drop any response whose `run_id` is not the active run, regardless of `version`;
 * drop any response with `version <=` the last applied version for that run; and never apply
 * an answer to a frame the panel is not currently pairing (the server never answers an
 * un-ingested frame, so a mismatch means the frame was superseded).
 */
export function decideUpdate(update: UpdateIdentity, state: FenceState): FenceDecision {
  if (state.activeRunId === null) return { accept: false, reason: 'no_run' };
  if (update.runId !== state.activeRunId) return { accept: false, reason: 'stale_run' };
  if (state.pairedFrameId === null || update.frameId !== state.pairedFrameId) {
    return { accept: false, reason: 'frame_mismatch' };
  }
  if (state.pairedFrameSeq === null || update.frameSeq !== state.pairedFrameSeq) {
    return { accept: false, reason: 'frame_mismatch' };
  }
  if (update.version <= state.lastVersion) return { accept: false, reason: 'stale_version' };
  return { accept: true, reason: 'accepted' };
}

import type { Box } from '@visual-coach/visual-tools';

/**
 * v4 invariant 1: `(box != null) == (state == "tracking")`. A type guard, so callers that only want
 * to draw can narrow with it and a contradictory payload can never paint a box.
 */
export function isBoxDrawable(box: Box | null | undefined, state: string): box is Box {
  return state === 'tracking' && box !== null && box !== undefined;
}

/**
 * v4 §2: a terminal state ends the run. `lost` is sticky and never auto-re-acquires, and `unavailable`
 * cannot serve the run at all, so the caller must stop uploading and let the user reseed or restart.
 * `acquiring` and `occluded` are transient and keep the loop alive.
 */
export function isTerminalTrackState(state: string): boolean {
  return state === 'lost' || state === 'unavailable';
}

export type TrackErrorAction = 'retire_run' | 'drop_frame' | 'lost_start_race' | 'unavailable' | 'unknown';

/**
 * Map the v4 error taxonomy (§2) onto the panel's behaviour. A 409 never carries track state,
 * so it never results in a box.
 */
export function decideError(status: number, code: string): TrackErrorAction {
  if (status === 409) {
    if (code === 'stale_run' || code === 'not_active_run') return 'retire_run';
    if (code === 'stale_frame') return 'drop_frame';
    if (code === 'stale_start') return 'lost_start_race';
    if (code === 'seed_not_first_frame') return 'retire_run';
  }
  if (status === 503 || status === 504) return 'unavailable';
  return 'unknown';
}

/**
 * A run is invalidated by any change to the scene it was started against: camera source,
 * frame geometry, mirror, scene mode, or the task text (parent decisions 9/12 — source and
 * mirror changes are expressed as a new run). Callers compare this key to the one captured at
 * run start and retire the run when it differs.
 */
export function runFenceKey(parts: {
  sourceEpoch: number;
  cameraFrameSize: string;
  mirrorView: boolean;
  sceneMode: string;
  taskKey: string;
}): string {
  // A NUL separator can never appear in a numeric epoch, a WxH size, a mode name, or a typed target,
  // so two different scenes can never collide into one key.
  return [parts.sourceEpoch, parts.cameraFrameSize, parts.mirrorView ? 'mirror' : 'plain', parts.sceneMode, parts.taskKey].join('\u0000');
}
