/**
 * Event triggers and the dispatch gate for the plan-follow loop (design v4 "계획-추종 루프"). Pure: the
 * hook feeds it observations of the live track store plus the clock, and acts on what comes back.
 *
 * Triggers
 * - `acquired`: once per (run_id, generation), when that pair is first seen in `tracking`.
 * - `target_moved`: the box centre moved more than `movedFraction` of the frame width, or its area changed
 *   by more than `areaRatio`×, versus the box of the last ACCEPTED answer, and that held for
 *   `movedSustainMs`. Fires once per excursion; re-armed when a newer accepted box arrives or the box
 *   comes back.
 * - `anchor_lost`: the state has been `lost` for `lostMs`. The timer is cancelled by `tracking`, by a new
 *   run and by a fence change; it re-arms when `lost` is entered again.
 * - `target_changed`: the tracked object's appearance changed inside a box that may not have moved
 *   (`anchorAppearance.ts`); produced by the hook's sampler, not here.
 * - `heartbeat`: every `heartbeatLocalMs` (local follower) or `heartbeatRemoteMs` (DeepSeek follower)
 *   while tracking, measured from the start of tracking or the last FOLLOW dispatch (`noteDispatch`): it is
 *   skipped whenever a follow left within the last interval, for whatever reason. A confirm does not count.
 * - `manual`: the button; not produced here.
 *
 * Dispatch gate, per stage (`follow`, `confirm`): each stage has its own in-flight slot, its own pending
 * slot (coalesced: the newest wins unless a less important call would displace a more important one) and
 * its own floor, so a confirm(step_done) never holds the next follow back. Paid calls of both
 * stages additionally share: a spacing that keeps two paid calls from reaching the server's guide lane
 * closer than its cooldown, a rolling budget per guide run (`remoteBudget` per `remoteWindowMs`) and a
 * mirror of the server's per-minute guide ceiling. The plan stage is exclusive: while a plan is in flight
 * nothing else leaves. A talk (`markTalk`) counts as a paid call and restarts the confirm floor.
 */
import type { Box } from '@visual-coach/visual-tools';

export type FollowTriggerKind = 'acquired' | 'target_moved' | 'target_changed' | 'anchor_lost' | 'manual' | 'heartbeat';
export type ConfirmTriggerKind = 'goal_check' | 'step_done' | 'unsure_twice' | 'replan' | 'target_left';

export interface TriggerConfig {
  movedFraction: number;
  areaRatio: number;
  movedSustainMs: number;
  lostMs: number;
  heartbeatLocalMs: number;
  heartbeatRemoteMs: number;
}

export const DEFAULT_TRIGGER_CONFIG: TriggerConfig = {
  movedFraction: 0.2,
  areaRatio: 1.5,
  movedSustainMs: 300,
  lostMs: 1000,
  heartbeatLocalMs: 4000,
  heartbeatRemoteMs: 8000,
};

/** What the hook observed in the live store (null when the store is empty). */
export interface TrackObservation {
  runId: string;
  trackId: string;
  generation: number;
  state: 'acquiring' | 'tracking' | 'occluded' | 'lost' | 'unavailable';
  box: Box | null;
}

/** The box of the last accepted answer, with the anchor identity it was measured on. */
export interface AcceptedBox {
  runId: string;
  generation: number;
  box: Box;
}

export interface TriggerState {
  runId: string | null;
  fenceKey: string | null;
  acquiredKeys: string[];
  movedSince: number | null;
  movedFired: boolean;
  movedReference: AcceptedBox | null;
  lostSince: number | null;
  lostFired: boolean;
  /** Start of the current heartbeat interval (tracking start, last follow dispatch or last heartbeat); null while not tracking. */
  heartbeatFrom: number | null;
}

export const INITIAL_TRIGGER_STATE: TriggerState = {
  runId: null,
  fenceKey: null,
  acquiredKeys: [],
  movedSince: null,
  movedFired: false,
  movedReference: null,
  lostSince: null,
  lostFired: false,
  heartbeatFrom: null,
};

export interface TriggerInput {
  observation: TrackObservation | null;
  nowMs: number;
  /** Any change of this key (task, camera, mirror, consent…) cancels the timers. */
  fenceKey: string;
  accepted: AcceptedBox | null;
  followProvider: 'local' | 'clef' | 'deepseek' | null;
  /** Frame height / width of the uploaded image (for the width-relative centre shift); default 1. */
  frameAspect?: number;
}

export interface TriggerStep {
  state: TriggerState;
  fired: FollowTriggerKind[];
}

function centre(box: Box): { x: number; y: number } {
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

/**
 * True when `box` left `reference` by the configured centre shift or area ratio. The shift is measured in
 * frame-WIDTH units: normalized y differences are converted with `frameAspect` (height / width).
 */
export function hasMoved(
  box: Box,
  reference: Box,
  config: TriggerConfig = DEFAULT_TRIGGER_CONFIG,
  frameAspect: number = 1
): boolean {
  const a = centre(box);
  const b = centre(reference);
  const aspect = Number.isFinite(frameAspect) && frameAspect > 0 ? frameAspect : 1;
  const shift = Math.hypot(a.x - b.x, (a.y - b.y) * aspect);
  if (shift > config.movedFraction) return true;
  const areaA = box.width * box.height;
  const areaB = reference.width * reference.height;
  if (areaA <= 0 || areaB <= 0) return areaA !== areaB;
  const ratio = areaA > areaB ? areaA / areaB : areaB / areaA;
  return ratio > config.areaRatio;
}

export function stepTriggers(
  previous: TriggerState,
  input: TriggerInput,
  config: TriggerConfig = DEFAULT_TRIGGER_CONFIG
): TriggerStep {
  const { observation: obs, nowMs: now } = input;
  const fired: FollowTriggerKind[] = [];
  let state: TriggerState = { ...previous };

  // A new run or a fence change cancels every timer; acquisition keys are per run.
  const runId = obs?.runId ?? null;
  if (runId !== state.runId || input.fenceKey !== state.fenceKey) {
    state = {
      ...INITIAL_TRIGGER_STATE,
      runId,
      fenceKey: input.fenceKey,
      acquiredKeys: runId !== null && runId === state.runId ? state.acquiredKeys : [],
    };
  }

  const tracking = obs?.state === 'tracking' && obs.box !== null;

  // acquired: once per (run, generation).
  if (obs && tracking) {
    const key = `${obs.runId}|${obs.generation}`;
    if (!state.acquiredKeys.includes(key)) {
      state.acquiredKeys = [...state.acquiredKeys, key];
      fired.push('acquired');
    }
  }

  // anchor_lost: lost sustained; any other state cancels, re-entry re-arms.
  if (obs?.state === 'lost') {
    if (state.lostSince === null) {
      state.lostSince = now;
      state.lostFired = false;
    }
    if (!state.lostFired && now - state.lostSince >= config.lostMs) {
      state.lostFired = true;
      fired.push('anchor_lost');
    }
  } else {
    state.lostSince = null;
    state.lostFired = false;
  }

  // target_moved: against the last accepted answer's box on the same anchor identity.
  const reference = input.accepted;
  if (reference !== state.movedReference) {
    state.movedReference = reference;
    state.movedSince = null;
    state.movedFired = false;
  }
  if (
    obs &&
    tracking &&
    obs.box &&
    reference &&
    reference.runId === obs.runId &&
    reference.generation === obs.generation &&
    hasMoved(obs.box, reference.box, config, input.frameAspect ?? 1)
  ) {
    if (state.movedSince === null) state.movedSince = now;
    if (!state.movedFired && now - state.movedSince >= config.movedSustainMs) {
      state.movedFired = true;
      fired.push('target_moved');
    }
  } else {
    state.movedSince = null;
    state.movedFired = false;
  }

  // heartbeat: while tracking, from tracking start / the last follow dispatch. Another trigger firing in the
  // same tick takes the slot (its follow, once it leaves, restarts the interval through `noteDispatch`).
  if (tracking) {
    if (state.heartbeatFrom === null) {
      state.heartbeatFrom = now;
    } else if (fired.length === 0) {
      const local = input.followProvider === 'local' || input.followProvider === 'clef';
      const interval = local ? config.heartbeatLocalMs : config.heartbeatRemoteMs;
      if (now - state.heartbeatFrom >= interval) {
        state.heartbeatFrom = now;
        fired.push('heartbeat');
      }
    }
  } else {
    state.heartbeatFrom = null;
  }

  return { state, fired };
}

/** Restart the heartbeat interval: a FOLLOW was just dispatched (for any trigger). Confirms do not call this. */
export function noteDispatch(state: TriggerState, nowMs: number): TriggerState {
  return state.heartbeatFrom === null ? state : { ...state, heartbeatFrom: nowMs };
}

/**
 * A call of `stage` was just dispatched. Only a FOLLOW restarts the heartbeat interval: a confirm (e.g. the
 * step_done right after an advance) must not postpone the next follow (phase-4 E2E: a confirm-based reset
 * held the follow back 5 s and 28.0 was never followed inside the clip).
 */
export function noteStageDispatch(state: TriggerState, stage: 'follow' | 'confirm', nowMs: number): TriggerState {
  return stage === 'follow' ? noteDispatch(state, nowMs) : state;
}

// ------------------------------------------------------------------------------------------ dispatch gate

export type CallLane = 'local' | 'remote';
export type GateStage = 'follow' | 'confirm';

export interface PendingCall {
  kind: GateStage;
  trigger: FollowTriggerKind | ConfirmTriggerKind;
  lane: CallLane;
  /** Opaque payload the hook attaches (e.g. the step to revert to). */
  meta?: Record<string, unknown>;
}

export interface GateConfig {
  /** Floor between two local follows. */
  followLocalFloorMs: number;
  /** Floor between two DeepSeek follows. */
  followRemoteFloorMs: number;
  /** Floor between two confirms (paid upper model). */
  confirmFloorMs: number;
  /** Minimum spacing for ANY two paid dispatches (mirrors the server's guide-lane cooldown + margin). */
  remoteSpacingMs: number;
  /** Paid calls allowed per rolling `remoteWindowMs` (plan included). Local calls are free. */
  remoteBudget: number;
  remoteWindowMs: number;
  /** Mirror of the server's per-minute guide-lane ceiling: paid calls per rolling minute (plan included). */
  remotePerMinute: number;
}

export interface StageSlot {
  inFlight: PendingCall | null;
  pending: PendingCall | null;
  lastAt: number;
}

export interface GateState {
  follow: StageSlot;
  confirm: StageSlot;
  /** A plan is in flight: nothing else may leave. */
  planInFlight: boolean;
  lastRemoteAt: number;
  /** Dispatch times of paid calls (pruned to the budget window). */
  remoteTimes: number[];
}

const EMPTY_SLOT: StageSlot = { inFlight: null, pending: null, lastAt: Number.NEGATIVE_INFINITY };

export const INITIAL_GATE_STATE: GateState = {
  follow: EMPTY_SLOT,
  confirm: EMPTY_SLOT,
  planInFlight: false,
  lastRemoteAt: Number.NEGATIVE_INFINITY,
  remoteTimes: [],
};

export const GATE_STAGES: readonly GateStage[] = ['follow', 'confirm'];

/**
 * Higher wins a stage's pending slot. Follow: heartbeat is the least important. Confirm: the completion
 * recheck outranks a step re-check (which also judges the goal), which outranks a plain goal check.
 */
export function callPriority(call: PendingCall): number {
  if (call.kind === 'confirm') {
    if (call.meta && (call.meta as { recheck?: boolean }).recheck) return 4;
    if (call.trigger === 'goal_check' || call.trigger === 'target_left') return 2;
    return 3;
  }
  return call.trigger === 'heartbeat' ? 1 : 2;
}

function withSlot(state: GateState, stage: GateStage, slot: StageSlot): GateState {
  return stage === 'follow' ? { ...state, follow: slot } : { ...state, confirm: slot };
}

/** Put `call` in its stage's pending slot unless a more important call already holds it. */
export function enqueue(state: GateState, call: PendingCall): GateState {
  const slot = state[call.kind];
  if (slot.pending && callPriority(slot.pending) > callPriority(call)) return state;
  return withSlot(state, call.kind, { ...slot, pending: call });
}

export type GateDecision =
  | { kind: 'idle' }
  | { kind: 'busy' }
  | { kind: 'wait'; waitMs: number }
  /** The rolling paid-call budget is spent until `waitMs` from now; the pending call is kept. */
  | { kind: 'capped'; waitMs: number; limit: 'budget' | 'per_minute' }
  | { kind: 'dispatch'; call: PendingCall };

/** Paid dispatch times still inside `windowMs`. */
function recent(times: number[], nowMs: number, windowMs: number): number[] {
  return times.filter((at) => at > nowMs - windowMs);
}

function stageFloor(call: PendingCall, config: GateConfig): number {
  if (call.kind === 'confirm') return config.confirmFloorMs;
  return call.lane === 'local' ? config.followLocalFloorMs : config.followRemoteFloorMs;
}

/**
 * Whether `stage`'s pending call may leave now. Does not mutate; `markDispatched` commits a dispatch.
 * `remoteHeld`: a talk waits for its floor — no other paid call may take the slot in front of it
 * (each such call would restart the floor and starve the talk); the hook pumps again once the talk leaves.
 */
export function decideDispatch(
  state: GateState,
  stage: GateStage,
  nowMs: number,
  config: GateConfig,
  remoteHeld: boolean = false
): GateDecision {
  const slot = state[stage];
  const call = slot.pending;
  if (!call) return { kind: 'idle' };
  if (slot.inFlight || state.planInFlight) return { kind: 'busy' };
  if (remoteHeld && call.lane === 'remote') return { kind: 'busy' };
  if (call.lane === 'remote') {
    const inBudget = recent(state.remoteTimes, nowMs, config.remoteWindowMs);
    if (inBudget.length >= config.remoteBudget) {
      return { kind: 'capped', waitMs: Math.max(1, inBudget[0] + config.remoteWindowMs - nowMs), limit: 'budget' };
    }
    const inMinute = recent(state.remoteTimes, nowMs, 60_000);
    if (inMinute.length >= config.remotePerMinute) {
      return { kind: 'capped', waitMs: Math.max(1, inMinute[0] + 60_000 - nowMs), limit: 'per_minute' };
    }
  }
  let waitMs = slot.lastAt + stageFloor(call, config) - nowMs;
  if (call.lane === 'remote') waitMs = Math.max(waitMs, state.lastRemoteAt + config.remoteSpacingMs - nowMs);
  if (waitMs > 0) return { kind: 'wait', waitMs };
  return { kind: 'dispatch', call };
}

function noteRemote(state: GateState, nowMs: number, config: GateConfig | null): GateState {
  const keep = config ? Math.max(config.remoteWindowMs, 60_000) : Number.POSITIVE_INFINITY;
  return { ...state, lastRemoteAt: nowMs, remoteTimes: [...recent(state.remoteTimes, nowMs, keep), nowMs] };
}

export function markDispatched(state: GateState, call: PendingCall, nowMs: number, config: GateConfig | null = null): GateState {
  const slot = state[call.kind];
  let next = withSlot(state, call.kind, {
    inFlight: call,
    pending: slot.pending === call ? null : slot.pending,
    lastAt: nowMs,
  });
  if (call.lane === 'remote') next = noteRemote(next, nowMs, config);
  return next;
}

export function markSettled(state: GateState, stage: GateStage): GateState {
  return withSlot(state, stage, { ...state[stage], inFlight: null });
}

/** The plan left (true): counted as a paid call; or it settled (false). */
export function markPlan(state: GateState, inFlight: boolean, nowMs: number, config: GateConfig | null = null): GateState {
  const next = { ...state, planInFlight: inFlight };
  return inFlight ? noteRemote(next, nowMs, config) : next;
}

/**
 * A talk call left (`/api/guide/talk`): it is a paid call (budget + per-minute mirror + spacing) and
 * shares the confirm stage's floor, without taking the confirm slot (talk has its own single flight).
 */
export function markTalk(state: GateState, nowMs: number, config: GateConfig | null = null): GateState {
  return noteRemote(withSlot(state, 'confirm', { ...state.confirm, lastAt: nowMs }), nowMs, config);
}

/** Drop a stage's pending call. */
export function dropPending(state: GateState, stage: GateStage): GateState {
  return state[stage].pending ? withSlot(state, stage, { ...state[stage], pending: null }) : state;
}

/** Paid calls inside the budget window (for the narration's budget line). */
export function remoteCallsInWindow(state: GateState, nowMs: number, config: GateConfig): number {
  return recent(state.remoteTimes, nowMs, config.remoteWindowMs).length;
}
