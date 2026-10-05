/**
 * The plan → follow → confirm loop (design v4 "계획-추종 루프"), event driven and fenced.
 *
 * - `start()` ("가이드 시작"): capture → `POST /api/guide/plan` as an event stream → partial target and first
 *   sentence are shown at once → on the final plan the tracking run starts with `selection.target`.
 * - Two cores (`planMode/coreChoice.ts`), chosen before Start and sent as `core_mode` on the plan request:
 *   `classic` (default) keeps the loop below exactly as it was. `sequential` has the follower judge only the
 *   CURRENT step, whose `yes` merely NOMINATES it: the screen moves one position only when the upper model
 *   answers its own `step_done: yes` for that step, plan revision, step activation and frame
 *   (`sequentialCore.ts`), and a checklist in that core never positions the screen.
 * - The live track store drives the triggers (`triggers.ts`): acquired / target_moved / anchor_lost /
 *   heartbeat, plus `manual()`; every 250 ms while tracking the appearance inside the box is sampled and
 *   compared with the frame of the last accepted follow (`anchorAppearance.ts`) → `target_changed`.
 * - Follow and confirm are separate stages: each has one request in flight plus one pending (coalesced) and
 *   its own floor, so a confirm(step_done) never holds the next follow back. Paid calls share a
 *   spacing (the server lane's cooldown) and a rolling budget per guide run (30 per 2 minutes); the plan
 *   stage is exclusive.
 * - Every answer goes through the acceptance rule (`acceptance.ts`): bound / text_only / rejected. Nothing
 *   on screen is cleared when a request leaves.
 * - The follower answers a checklist: for each step from the asked one to the last, is its `done_when` visible
 *   now (`followStep.ts`). The asked step `yes` advances at once; a later step `yes` skips to the step after
 *   it only when the previous accepted follow showed it too (the steps passed over without a `yes` are shown
 *   as skipped, never as done). Either is immediate (the plan's own sentence and commands) and re-checked by
 *   an asynchronous confirm(step_done); a confirm whose `step_check` is `no` for that step reverts it.
 * - A confirm a follow answer raises (step_done, or goal_check on `goal_seen: yes`) is sent with that
 *   follow's own frame and AnchorRef snapshot (`confirmDecision.ts`): it re-checks what the follower saw,
 *   not a later state. anchor_lost goal_check, unsure_twice and the completion re-check capture fresh.
 * - A confirm answer's step decision and completion decision are separate: `step_check: no` reverts the
 *   step even when the same answer is `visually_satisfied`, and completion still takes the re-check.
 * - Replans (`replanPolicy.ts`): three accepted `target_changed` follows with no step change → confirm(replan)
 *   on a fresh frame; the asked step `unsure` twice in a row → confirm(unsure_twice); the "계획 다시 짜기"
 *   button → confirm(replan). Both carry the last accepted checklist (`follow_checks`) and the step the screen
 *   shows (`current_step`); the judge answers its own `step_checks` from there, and unless it also replans, its
 *   latest `yes` moves the screen at once (`decideConfirmPosition`, skipped steps as on a follow skip). At most
 *   ten per run and never within 8 s of the last plan or replan (`replanPolicy.ts`); a blocked one is only counted.
 * - Completion only from a bound confirm: `visually_satisfied` → "확인 중" → one more bound confirm on a fresh
 *   frame 1.5 s later → "완료로 보임(2회 확인)", loops stop, and the user may press "완료 확인". The follower,
 *   the tracker state and text-only answers never touch completion.
 * - Talk (`talk()`, design 2026-10-01 guide-talk): the user's words go to `/api/guide/talk` on a fresh
 *   capture, one at a time (never queued), within the shared paid-call budget (`talkPolicy.ts`). The answer
 *   goes through the acceptance rule and `applyTalk.ts`: a rewritten sentence, a user mark (done/skipped),
 *   going back, a new target noun (tracking restarts, no paid call) or a replan. A follow/confirm that a talk
 *   or replan overtook (409 stale_plan at the client's revision) is dropped, not fatal. A user-marked step is
 *   never reverted by a late step_done confirm.
 *
 * The anchored overlay reads `overlay` (an external store) per animation frame; the page re-renders only
 * when the view changes (a step, a call starting or ending, a notice).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type {
  Box,
  ExitEdge,
  GuideConfirmRequest,
  GuideConfirmResponse,
  GuideFollowRequest,
  GuideFollowResponse,
  GuidePlanResponse,
  GuideStep,
  GuideTalkRequest,
  GuideTalkResponse,
  MaterialInput,
  Usage,
} from '../generated/api.generated';
import { ApiError } from '../coach/api';
import {
  CameraFrameUnavailableError,
  CaptureSkippedError,
  captureVideoFrameAsync,
  sampleVideoAppearanceAsync,
} from '../coach/image-processor';
import {
  EMPTY_GUIDE_TRACE,
  appendGuideCall,
  bumpGuideCounter,
  markGuideMilestone,
  sampleGuideHide,
  startGuideTrace,
  type GuideCallAcceptance,
  type GuideCounter,
  type GuideTrace,
} from '../coach/debugTrace';
import { LIVE_MIN_DISPATCH_INTERVAL_MS } from './dispatchFloor';
import { newOpaqueId } from '../ids';
import { isLiveBoxDrawable, type LiveTrackSnapshot, type LiveTrackStore } from '../tracking/liveTrackStore';
import { classifyAnswer, textOnlyQualifier, type Acceptance, type SentStamp } from './acceptance';
import {
  PRIMARY_ANCHOR_ID,
  anchorLabel,
  bindStepCommands,
  bindingFromSnapshot,
  buildAnchorRef,
  echoOf,
  type AnchorBinding,
} from './anchorProject';
import {
  INITIAL_APPEARANCE_STATE,
  appearanceSampleDue,
  stepAppearance,
  type AppearanceReference,
  type AppearanceState,
} from './anchorAppearance';
import { currentStepVisible, decideFollowStep, type StepCheck } from './followStep';
import {
  currentStepChecks,
  dropSequentialAwaiting,
  freshSequentialState,
  nominateSequential,
  noteStepActivation,
  resetSequentialForPlan,
  settleSequential,
  type SequentialState,
} from './sequentialCore';
import { addSkipped, skippedBefore } from './planProgress';
import {
  describeReplanBlock,
  freshReplanBudget,
  nextStuckCount,
  notePlanInstalled,
  replanBlock,
  spendReplan,
  releaseReplan,
  replanDispatch,
  reserveReplan,
  type ReplanBudget,
} from './replanPolicy';
import {
  confirmFrameSource,
  decideConfirm,
  exitEdgeOf,
  recheckCall,
  retryRecheckAfterStale,
  type CompletionState,
  type ConfirmFrameSource,
  type ConfirmMeta,
  type ConfirmPositionDecision,
  type FollowFrame,
  decideConfirmPosition,
} from './confirmDecision';
import { createGuideOverlayStore, EMPTY_GUIDE_OVERLAY, type GuideOverlayStore } from './guideOverlayStore';
import { hasUnsupportedExecution, unsupportedPlanNotice } from '../planMode/capability';
import { DEFAULT_CORE_MODE, type CoreMode } from '../planMode/coreChoice';
import type { PlanModel } from '../planMode/modelChoice';
import { motionKey } from './actionMotion';
import { applyTalk, talkChangesPlan, type TalkAction, type TalkRunState } from './applyTalk';
import { noticeExpired, type NoticeStamp } from './noticePolicy';
import {
  TALK_UTTERANCE_MAX,
  decideTalk,
  describeTalkBlock,
  describeTalkError,
  isSupersededStale,
  talkSendStep,
} from './talkPolicy';
import { requestGuideConfirm, requestGuideFollow, requestGuidePlan, requestGuideTalk } from './intentClient';
import { IntentHttpError, type PlanPartial } from './planStream';
import {
  GATE_STAGES,
  INITIAL_GATE_STATE,
  INITIAL_TRIGGER_STATE,
  decideDispatch,
  enqueue,
  markDispatched,
  markPlan,
  markSettled,
  markTalk,
  noteStageDispatch,
  stepTriggers,
  type AcceptedBox,
  type ConfirmTriggerKind,
  type FollowTriggerKind,
  type GateConfig,
  type GateStage,
  type GateState,
  type PendingCall,
  type TriggerState,
} from './triggers';

/** Minimum spacing of two local follows (the local follower is free but single-flight on the server). */
export const LOCAL_FOLLOW_FLOOR_MS = 1000;
/** Wait before the second, confirming completion check on a fresh frame. */
export const COMPLETION_RECHECK_MS = 1500;
/** How often the loop re-reads the store when no frame is published (timers: lost, heartbeat, moved). */
const TICK_MS = 100;
/** An appearance sample that was not taken (`CaptureSkippedError`). */
const SKIPPED = Symbol('appearance-sample-skipped');
/** The trace (hide-ratio samples) is pushed to React at most this often. */
const TRACE_PUBLISH_MS = 1000;
/** Consecutive failed calls after which the loop stops instead of retrying forever. */
const MAX_CONSECUTIVE_FAILURES = 3;

/**
 * Shown while the sequential core waits: the follower called the current step done and the screen has NOT
 * moved — the upper model's own confirmation is what moves it (`sequentialCore.ts`).
 */
const SEQUENTIAL_WAIT_NOTICE = '단계가 끝난 것으로 보입니다. 상위 모델 확인을 기다리는 중이며, 확인되면 다음 단계로 넘어갑니다.';

/** Paid calls (plan, remote follows, confirms) per rolling window of one guide run. */
export const REMOTE_BUDGET = 30;
export const REMOTE_BUDGET_WINDOW_MS = 120_000;

export const GATE_CONFIG: GateConfig = {
  followLocalFloorMs: LOCAL_FOLLOW_FLOOR_MS,
  followRemoteFloorMs: LIVE_MIN_DISPATCH_INTERVAL_MS,
  confirmFloorMs: LIVE_MIN_DISPATCH_INTERVAL_MS,
  // The server's guide lane refuses a second paid call within 1.0 s (main.py GUIDE_ANALYSIS_COOLDOWN); the
  // margin covers the capture/encode time between the client's dispatch and the server's arrival.
  remoteSpacingMs: 1200,
  remoteBudget: REMOTE_BUDGET,
  remoteWindowMs: REMOTE_BUDGET_WINDOW_MS,
  // Mirror of the server's guide-lane ceiling (main.py GUIDE_ANALYSES_PER_MINUTE).
  remotePerMinute: 20,
};

export type FollowProvider = 'local' | 'clef' | 'deepseek';
export type GuideStageName = 'plan' | 'follow' | 'confirm' | 'talk';
export type IntentPhase = 'idle' | 'planning' | 'running' | 'completed' | 'error';
export type { CompletionState };

/** A call shown as pending in the narration bar (elapsed time ticks from `startedAt`). */
export interface PendingView {
  stage: GuideStageName;
  trigger: string;
  provider: FollowProvider;
  /** `performance.now()` when the call started (capture). */
  startedAt: number;
}

/**
 * The sequential core's waiting state: the follower called this step done and the upper model's own
 * `step_done` answer has not arrived yet, so the screen has NOT moved. It is shown while the candidate is
 * pending, not after it (an upper yes moves the step and clears it).
 */
export interface SequentialPending {
  stepId: string;
  stepIndex: number;
  /** The step activation the candidate belongs to. */
  activation: number;
}

export interface IntentView {
  phase: IntentPhase;
  /** The core this run executes, frozen at Start (the picker's choice for the NEXT run is the host's). */
  coreMode: CoreMode;
  /** Installed plan (text only), or null. */
  plan: {
    planId: string;
    planRevision: number;
    steps: GuideStep[];
    goalWhen: string;
    target: string;
    /** The reference materials frozen at Start (null for a goal-only run): the plan's evidence resolves against them. */
    materials: MaterialInput[] | null;
    userGoal: string;
  } | null;
  stepIndex: number;
  /** Steps before `stepIndex` the client skipped (shown as skipped, not done). */
  skippedStepIds: string[];
  /** Steps before `stepIndex` the user said were done by talk (사용자 확인 ✓, not 완료 ✓). */
  userDoneStepIds: string[];
  /** Latest accepted visual checklist, never inferred from step position or user claims. */
  stepChecks: StepCheck[] | null;
  /** Sequential core: the step awaiting the upper model's confirmation (null otherwise). */
  sequentialPending: SequentialPending | null;
  partialTarget: string | null;
  partialFirstSay: string | null;
  pending: PendingView | null;
  lastTrigger: string | null;
  /** Set while the newest shown answer was text-only: '…초 전 화면 기준'. */
  textOnlyQualifier: string | null;
  notice: string | null;
  /** Set while a paid call is waiting for the rolling budget (shown in the narration). */
  budgetNotice: string | null;
  completion: CompletionState;
  needsReselect: boolean;
  /** Why "계획 다시 짜기" is unavailable right now (the per-run maximum or the replan gap), or null. */
  replanBlockedReason: string | null;
  /** The latest applied exchange with the guide (its pair is sent as `prev_*` with the next utterance). */
  talk: { utterance: string; reply: string; spoken: string; at: number } | null;
  /** `performance.now()` when the waiting utterance was accepted for sending, or null (input unlocked). */
  talkPendingSince: number | null;
  /** Why an utterance cannot be sent right now, or null. */
  talkBlockedReason: string | null;
  /** Why the last utterance got no applied answer, or null. */
  talkError: string | null;
  error: string | null;
}

const INITIAL_VIEW: IntentView = {
  phase: 'idle',
  coreMode: DEFAULT_CORE_MODE,
  plan: null,
  stepIndex: 0,
  skippedStepIds: [],
  userDoneStepIds: [],
  stepChecks: null,
  sequentialPending: null,
  partialTarget: null,
  partialFirstSay: null,
  pending: null,
  lastTrigger: null,
  textOnlyQualifier: null,
  notice: null,
  budgetNotice: null,
  completion: 'none',
  needsReselect: false,
  replanBlockedReason: null,
  talk: null,
  talkPendingSince: null,
  talkBlockedReason: null,
  talkError: null,
  error: null,
};

/** One accounted guide call (fold it into the page's cost totals). */
export type GuideUsageEvent =
  | { kind: 'billed'; stage: GuideStageName; provider: string; model: string | null; usage: Usage | null }
  | { kind: 'unknown'; stage: GuideStageName; lane: 'local' | 'remote' };

export interface UseIntentLoopOptions {
  videoRef: RefObject<HTMLVideoElement | null>;
  mirror: boolean;
  liveTrack: LiveTrackStore;
  ensureSession: () => Promise<string>;
  /** The app's current session id (a different one retires every answer of the old one). */
  sessionId: string | null;
  consent: boolean;
  cameraLive: boolean;
  userGoal: string;
  context: string;
  /** Reference documents for the plan prompt (optional; <=4, each title 1..80 / text 1..4000). */
  materials?: MaterialInput[] | null;
  /** Any change (task text, camera source, mirror, scene mode, plan model) retires the running guide. */
  fenceKey: string;
  /** The chosen plan model; sent as `plan_model` on this run's first plan request only. */
  planModel: PlanModel;
  /** The chosen core; sent as `core_mode` on this run's first plan request only. */
  coreMode: CoreMode;
  /** `health.follow_provider`; null means the follower is not configured. */
  followProvider: FollowProvider | null;
  /** Start the tracking run for a known target noun (no provider call). */
  startTracking: (target: string) => Promise<void>;
  stopTracking: () => void;
  onUsage?: (event: GuideUsageEvent) => void;
  onSessionExpired?: () => void;
  /** The user pressed “완료 확인” after a twice-confirmed completion. Records only; never a goal status. */
  onUserConfirmedCompletion?: (info: { planId: string; userGoal: string }) => void;
}

interface GuideRun {
  id: number;
  taskEpoch: string;
  sessionId: string | null;
  userGoal: string;
  context: string | null;
  /** The reference materials sent with the plan (null: none). Non-null gates the grounded plan behaviors. */
  materials: MaterialInput[] | null;
  /** The plan model this run was started with; the server keeps it for confirm/replan/talk. */
  planModel: PlanModel;
  /** The core this run executes: the chosen one, or the one the plan response recorded. */
  coreMode: CoreMode;
  controller: AbortController;
  startedAt: number;
  planId: string | null;
  planRevision: number | null;
  steps: GuideStep[];
  goalWhen: string;
  target: string;
  stepIndex: number;
  /** Sequential core only: the step machine (`sequentialCore.ts`); null in the classic core. */
  seq: SequentialState | null;
  /** Last intent_seq handed out (one counter for both stages). */
  intentSeq: number;
  /** Newest intent_seq dispatched per stage: an answer is superseded only by a newer call of its own stage. */
  latestSeq: Record<GateStage | 'talk', number>;
  unsureStreak: number;
  /** The last bound follow's checklist (this plan revision only): a skip's second sighting, `follow_checks`. */
  lastChecks: StepCheck[] | null;
  /** Skipped step ids before `stepIndex`. */
  skipped: string[];
  /** Step ids the user said were done (talk); pruned to the steps before `stepIndex` on a step change. */
  userDone: string[];
  /** An utterance is waiting or in flight (one at a time, never queued). */
  talkPending: boolean;
  /** An accepted utterance waits for its floor; paid follows/confirms are held until it leaves. */
  talkWaiting: boolean;
  /** The last applied exchange: the next utterance's `prev_utterance`/`prev_reply`. */
  lastTalk: { utterance: string; reply: string } | null;
  /** Accepted target_changed follows since the last step change. */
  stuckCount: number;
  replanBudget: ReplanBudget;
  /** Replans installed in this run (confirm or talk): a replan confirm asked before the latest one is moot. */
  replanEpoch: number;
  consecutiveFailures: number;
  completion: CompletionState;
  /** Loops stopped (completion confirmed, or a fatal error); the plan stays on screen. */
  halted: boolean;
  binding: AnchorBinding | null;
  accepted: AcceptedBox | null;
  /** The appearance grid of the frame of the last accepted follow (target_changed reference). */
  appearanceRef: AppearanceReference | null;
  warn: boolean;
  firstBoxSeen: boolean;
  /** The first follow frame sent with the target tracked (since `acquired`): target_left's `before_scene`. */
  firstSeen: FollowFrame | null;
  /** The last live tracked box: where the target left the frame (`exit_edge`). */
  lastBox: Box | null;
}

/** What a call carries besides its trigger (a confirm's step and, when a follow raised it, its frame). */
type CallMeta = ConfirmMeta;

/** target_left's frames for this run: its first tracked frame and where the last live box was (null: none yet). */
function targetLeft(run: GuideRun): { beforeFrame: FollowFrame; exitEdge?: ExitEdge } | null {
  if (!run.firstSeen) return null;
  return { beforeFrame: run.firstSeen, exitEdge: run.lastBox ? exitEdgeOf(run.lastBox) : undefined };
}

/** What a follow was sent with (copied at capture). */
interface FollowSent {
  box: AcceptedBox | null;
  appearance: AppearanceReference | null;
  stepIndex: number;
  activation: number | null;
}

/** `step_check`/`step_id` arrive with the step_done confirm (read defensively: older servers lack them). */
type ConfirmStepFields = { step_check?: 'yes' | 'no' | 'unsure' | null; step_id?: string | null };

const FOLLOW_TRIGGER_TEXT: Record<string, string> = {
  acquired: '대상 잡힘',
  target_moved: '대상 이동',
  target_changed: '대상 변화',
  anchor_lost: '대상 놓침',
  manual: '직접 요청',
  heartbeat: '주기 확인',
  goal_check: '완료 확인',
  target_left: '화면 밖 확인',
  step_done: '단계 확인',
  unsure_twice: '판단 불확실',
  replan: '계획 재검토',
  start: '가이드 시작',
  talk: '사용자 말',
};

export function describeTrigger(trigger: string | null): string | null {
  if (!trigger) return null;
  return FOLLOW_TRIGGER_TEXT[trigger] ?? trigger;
}

/** The trace counter each applied talk action bumps. */
const TALK_ACTION_COUNTER: Record<TalkAction, GuideCounter | null> = {
  none: null,
  step_say: 'talkSay',
  target: 'talkTarget',
  step_mark: 'talkMark',
  go_to: 'talkGoTo',
  replan: 'talkReplan',
};

/** Whether a failed call may still have been billed by a paid provider (unknown spend, not zero). */
function mayHaveBilled(error: unknown): boolean {
  if (error instanceof ApiError) return error.status >= 500 && error.status !== 503;
  // A transport failure after dispatch: the provider may have run.
  return !(error instanceof DOMException && error.name === 'AbortError');
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

let guideRunCounter = 0;

export function useIntentLoop(options: UseIntentLoopOptions) {
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const [view, setViewState] = useState<IntentView>(INITIAL_VIEW);
  const [trace, setTrace] = useState<GuideTrace>(EMPTY_GUIDE_TRACE);
  const overlay = useMemo<GuideOverlayStore>(() => createGuideOverlayStore(), []);

  const runRef = useRef<GuideRun | null>(null);
  const gateRef = useRef<GateState>(INITIAL_GATE_STATE);
  const triggerRef = useRef<TriggerState>(INITIAL_TRIGGER_STATE);
  const appearanceRef = useRef<AppearanceState>(INITIAL_APPEARANCE_STATE);
  /** An appearance sample is in the capture worker; the next is not taken until it lands. */
  const appearanceSamplingRef = useRef(false);
  const traceRef = useRef<GuideTrace>(EMPTY_GUIDE_TRACE);
  const timersRef = useRef<Set<number>>(new Set());
  const gateTimerRef = useRef<{ handle: number; dueAt: number } | null>(null);
  /** A DeepSeek-backed call is waiting for the budget (the narration says so; counted once per episode). */
  const budgetCappedRef = useRef(false);
  const talkBlockRef = useRef<string | null>(null);
  /** The notice on screen and the step it was raised on (cleared on a step change or after 8 s). */
  const noticeRef = useRef<NoticeStamp | null>(null);
  /** The calls in flight per stage, as the narration shows them (the follow wins the single chip). */
  const pendingViewsRef = useRef<Partial<Record<GateStage, PendingView>>>({});
  const lastTracePublishRef = useRef(0);
  const mountedRef = useRef(true);

  const setView = useCallback((patch: Partial<IntentView> | ((previous: IntentView) => Partial<IntentView>)) => {
    if (!mountedRef.current) return;
    setViewState((previous) => ({ ...previous, ...(typeof patch === 'function' ? patch(previous) : patch) }));
  }, []);

  const updateTrace = useCallback((change: (previous: GuideTrace) => GuideTrace, publish = true) => {
    traceRef.current = change(traceRef.current);
    if (publish && mountedRef.current) {
      lastTracePublishRef.current = performance.now();
      setTrace(traceRef.current);
    }
  }, []);

  const later = useCallback((fn: () => void, ms: number) => {
    const handle = window.setTimeout(() => {
      timersRef.current.delete(handle);
      fn();
    }, ms);
    timersRef.current.add(handle);
    return handle;
  }, []);

  const clearTimers = useCallback(() => {
    for (const handle of timersRef.current) window.clearTimeout(handle);
    timersRef.current.clear();
    gateTimerRef.current = null;
  }, []);

  const setPendingView = useCallback(
    (stage: GateStage, pending: PendingView | null) => {
      if (pending) pendingViewsRef.current = { ...pendingViewsRef.current, [stage]: pending };
      else {
        const next = { ...pendingViewsRef.current };
        delete next[stage];
        pendingViewsRef.current = next;
      }
      const views = pendingViewsRef.current;
      setView({ pending: views.follow ?? views.confirm ?? null });
    },
    [setView]
  );

  const publishOverlay = useCallback(
    (run: GuideRun | null) => {
      if (!run || run.halted) {
        overlay.set(EMPTY_GUIDE_OVERLAY);
        return;
      }
      const step = run.steps[run.stepIndex] ?? null;
      overlay.set({
        binding: run.binding,
        commands: run.binding ? bindStepCommands(step, run.binding.anchorId) : [],
        warn: run.warn,
        motionKey:
          step && run.binding
            ? motionKey({
                runId: run.id,
                planRevision: run.planRevision,
                stepId: step.id,
                anchorId: run.binding.anchorId,
                trackRunId: run.binding.runId,
                trackId: run.binding.trackId,
                generation: run.binding.generation,
              })
            : null,
      });
    },
    [overlay]
  );

  const isCurrent = (run: GuideRun) => runRef.current === run && !run.controller.signal.aborted;

  /** Retire the guide: abort the in-flight call, stop every timer, keep nothing drawable. */
  const retire = useCallback(
    (reason: { error?: string; notice?: string; keepView?: boolean } = {}) => {
      const run = runRef.current;
      runRef.current = null;
      run?.controller.abort();
      clearTimers();
      gateRef.current = INITIAL_GATE_STATE;
      triggerRef.current = INITIAL_TRIGGER_STATE;
      appearanceRef.current = INITIAL_APPEARANCE_STATE;
      pendingViewsRef.current = {};
      budgetCappedRef.current = false;
      talkBlockRef.current = null;
      overlay.set(EMPTY_GUIDE_OVERLAY);
      if (reason.keepView) {
        setView({
          pending: null,
          budgetNotice: null,
          sequentialPending: null,
          talkPendingSince: null,
          talkBlockedReason: null,
          phase: reason.error ? 'error' : 'idle',
          error: reason.error ?? null,
          notice: reason.notice ?? null,
        });
      } else {
        setView({
          ...INITIAL_VIEW,
          phase: reason.error ? 'error' : 'idle',
          error: reason.error ?? null,
          notice: reason.notice ?? null,
        });
      }
      setTrace(traceRef.current);
    },
    [clearTimers, overlay, setView]
  );

  // ------------------------------------------------------------------------------------------- dispatch

  const pumpRef = useRef<() => void>(() => {});

  const enqueueCall = useCallback((call: PendingCall) => {
    const run = runRef.current;
    if (!run || run.halted || run.planId === null) return;
    const previousFrame = (gateRef.current.confirm.pending?.meta as CallMeta | undefined)?.followFrame?.frameId;
    gateRef.current = enqueue(gateRef.current, call);
    if (run.seq && call.kind === 'confirm') {
      const retained = (gateRef.current.confirm.pending?.meta as CallMeta | undefined)?.followFrame?.frameId;
      const incomingFrame = (call.meta as CallMeta | undefined)?.followFrame?.frameId;
      const discarded = previousFrame && previousFrame !== retained ? previousFrame :
        call.trigger === 'step_done' && incomingFrame !== retained ? incomingFrame : null;
      if (discarded) {
        run.seq = dropSequentialAwaiting(run.seq, discarded);
        publishSequential(run);
      }
    }
    pumpRef.current();
  }, []);

  // Both loopback followers (llama.cpp `local`, Clef) are unbilled: the local lane, outside the remote budget.
  const followLane = (): 'local' | 'remote' => {
    const provider = optionsRef.current.followProvider;
    return provider === 'local' || provider === 'clef' ? 'local' : 'remote';
  };

  const enqueueFollow = useCallback(
    (trigger: FollowTriggerKind) => {
      enqueueCall({ kind: 'follow', trigger, lane: followLane() });
    },
    [enqueueCall]
  );

  const enqueueConfirm = useCallback(
    (trigger: ConfirmTriggerKind, meta?: CallMeta) => {
      enqueueCall({ kind: 'confirm', trigger, lane: 'remote', meta: meta as Record<string, unknown> | undefined });
    },
    [enqueueCall]
  );

  const currentAnchorEcho = (snapshot: LiveTrackSnapshot | null) =>
    snapshot ? { anchor_id: PRIMARY_ANCHOR_ID, track_id: snapshot.trackId, generation: snapshot.generation } : null;

  const classify = (
    run: GuideRun,
    stage: GateStage | 'talk',
    stamp: SentStamp,
    answer: GuideFollowResponse | GuideConfirmResponse | GuideTalkResponse,
    replan: boolean
  ): Acceptance => {
    const snapshot = optionsRef.current.liveTrack.current();
    return classifyAnswer(
      stamp,
      answer,
      {
        sessionId: optionsRef.current.sessionId ?? run.sessionId,
        taskEpoch: isCurrent(run) ? run.taskEpoch : null,
        runId: snapshot?.runId ?? null,
        planId: run.planId,
        planRevision: run.planRevision,
        latestIntentSeq: run.latestSeq[stage],
        anchor: currentAnchorEcho(snapshot),
      },
      { replanApplied: replan }
    );
  };

  const recordCall = (
    stage: GuideStageName,
    trigger: string,
    provider: string | null,
    startedAt: number,
    accepted: GuideCallAcceptance,
    errorCode: string | null
  ) => {
    updateTrace((previous) =>
      appendGuideCall(previous, {
        id: newOpaqueId('gcall'),
        stage,
        trigger,
        provider,
        msToFinal: Math.round(performance.now() - startedAt),
        accepted,
        errorCode,
        startedAtWall: Date.now(),
      })
    );
  };

  /**
   * What the narration and the browser see of the sequential core: the newest candidate still waiting for the
   * upper model's answer. Nothing is shown once it resolves — the step has moved, so the wait is over.
   */
  const publishSequential = (run: GuideRun) => {
    const waiting = run.seq?.awaiting[0] ?? null;
    setView((previous) => ({
      coreMode: run.coreMode,
      sequentialPending: waiting
        ? { stepId: waiting.stepId, stepIndex: waiting.stepIndex, activation: waiting.activation }
        : null,
      ...(!waiting && previous.notice === SEQUENTIAL_WAIT_NOTICE ? { notice: null } : {}),
    }));
  };

  /** Move the screen to step `index`. Skips at or after `index` are forgotten unless `skipped` says otherwise. */
  const setStep = (run: GuideRun, index: number, skipped: string[] = skippedBefore(run.steps, run.skipped, index)) => {
    // A step change is a new step activation: a sequential candidate of the old one can never move it.
    if (run.seq && index !== run.stepIndex) run.seq = noteStepActivation(run.seq);
    run.stepIndex = index;
    run.skipped = skipped;
    run.userDone = skippedBefore(run.steps, run.userDone, index);
    run.stuckCount = 0;
    run.unsureStreak = 0;
    publishOverlay(run);
    publishSequential(run);
    setView({ stepIndex: index, skippedStepIds: skipped, userDoneStepIds: run.userDone, textOnlyQualifier: null });
  };

  /** Publish whether "계획 다시 짜기" is available; re-checks itself when a cooldown ends. */
  const refreshReplanBlock = (run: GuideRun) => {
    if (!isCurrent(run)) return;
    const { block, waitMs } = replanBlock(run.replanBudget, performance.now());
    setView({ replanBlockedReason: describeReplanBlock(block) });
    if (block === 'cooldown') later(() => refreshReplanBlock(run), waitMs + 20);
  };

  /** Send a replan-capable confirm within the per-run budget; a blocked one is only counted. */
  const requestReplan = (run: GuideRun, trigger: 'replan' | 'unsure_twice', checks: StepCheck[] | null): boolean => {
    const now = performance.now();
    if (replanBlock(run.replanBudget, now).block !== null) {
      updateTrace((previous) => bumpGuideCounter(previous, 'replanLimited'));
      return false;
    }
    run.replanBudget = spendReplan(run.replanBudget, now);
    enqueueConfirm(trigger, { ...(checks && checks.length > 0 ? { followChecks: checks } : {}), replanEpoch: run.replanEpoch });
    refreshReplanBlock(run);
    return true;
  };

  const installReplan = (run: GuideRun, answer: GuideConfirmResponse) => {
    if (!answer.replan || run.planRevision === null || answer.plan_revision <= run.planRevision) return false;
    run.steps = answer.replan.steps;
    run.planRevision = answer.plan_revision;
    run.stepIndex = 0;
    run.unsureStreak = 0;
    run.lastChecks = null;
    run.skipped = [];
    run.userDone = [];
    run.stuckCount = 0;
    run.replanBudget = notePlanInstalled(run.replanBudget, performance.now());
    run.replanEpoch += 1;
    // A replan is a new plan: the sequential step machine starts over on the new steps.
    if (run.seq) run.seq = resetSequentialForPlan(run.seq);
    publishOverlay(run);
    publishSequential(run);
    setView((previous) => ({
      plan: previous.plan ? { ...previous.plan, steps: answer.replan!.steps, planRevision: answer.plan_revision } : previous.plan,
      stepIndex: 0,
      skippedStepIds: [],
      userDoneStepIds: [],
      stepChecks: null,
      notice: '계획이 다시 짜였습니다. 새 1단계부터 안내합니다.',
    }));
    refreshReplanBlock(run);
    return true;
  };

  const applyFollow = (
    run: GuideRun,
    answer: GuideFollowResponse,
    acceptance: Acceptance,
    sent: FollowSent,
    capturedAt: number,
    followFrame: FollowFrame
  ) => {
    if (acceptance === 'rejected') return;
    if (acceptance === 'text_only') {
      // Words only: the step sentence stays, labelled with the capture age. No geometry, no step change.
      setView({ textOnlyQualifier: textOnlyQualifier(capturedAt, Date.now()) });
      return;
    }
    if (run.seq && (sent.activation !== run.seq.activation || sent.stepIndex !== run.stepIndex)) return;
    if (sent.box) run.accepted = sent.box;
    // The target_changed reference is the frame the follower judged (null when it could not be sampled).
    run.appearanceRef = sent.appearance;
    run.warn = answer.needs_reselect;
    // The sequential core has the current step judged and nothing else, so only its entry is kept either
    // side: a future step's reading is not this client's to show or to act on.
    const checks = run.seq ? currentStepChecks(run.steps, sent.stepIndex, answer.step_checks) : answer.step_checks;
    publishOverlay(run);
    setView({ needsReselect: answer.needs_reselect, textOnlyQualifier: null, stepChecks: checks });

    const previousChecks = run.lastChecks;
    run.lastChecks = checks;
    const stepBefore = run.stepIndex;

    // The asked step's own entry: `yes` is counted, `unsure` twice in a row asks the selected upper model to replan.
    const current = currentStepVisible(run.steps, sent.stepIndex, checks ?? []);
    let replanAsked = false;
    if (run.seq === null && current === 'yes') updateTrace((previous) => bumpGuideCounter(previous, 'followDone'));
    if (current === 'unsure') {
      updateTrace((previous) => bumpGuideCounter(previous, 'followUnsure'));
      run.unsureStreak += 1;
      if (run.unsureStreak >= 2) {
        run.unsureStreak = 0;
        replanAsked = true;
        requestReplan(run, 'unsure_twice', checks);
      }
    } else {
      run.unsureStreak = 0;
    }

    // How this answer moves the screen. The classic core moves it here and lets an upper confirm revert; the
    // sequential core only NOMINATES the step, and the upper model's own step_done answer moves it.
    const asked = run.steps[sent.stepIndex];
    let awaitingConfirm = false;
    if (run.seq) {
      if (!answer.needs_reselect && current === 'yes' && asked && sent.stepIndex === run.stepIndex && run.planRevision !== null) {
        const nominated = nominateSequential(run.seq, {
          planRevision: run.planRevision,
          stepIndex: sent.stepIndex,
          stepId: asked.id,
          frameId: followFrame.frameId,
          askedAt: performance.now(),
          activation: sent.activation!,
        });
        // Refused when this step's confirmation already committed: nothing is left to ask for it.
        if (nominated !== run.seq) {
          run.seq = nominated;
          updateTrace((previous) => bumpGuideCounter(previous, 'sequentialNominate'));
          publishSequential(run);
          enqueueConfirm('step_done', { stepId: asked.id, followFrame });
          setView({ notice: SEQUENTIAL_WAIT_NOTICE });
          awaitingConfirm = true;
        }
      }
    } else {
      // Judged against the step the request ASKED about; the screen may have moved on while it was in flight.
      const decision = decideFollowStep(run.steps, sent.stepIndex, run.stepIndex, answer.step_checks, previousChecks);
      if (decision.kind === 'pendingSkip') {
        updateTrace((previous) => bumpGuideCounter(previous, 'followPendingSkip'));
      } else if (decision.kind === 'advance' || decision.kind === 'skip') {
        if (decision.kind === 'skip') {
          updateTrace((previous) => bumpGuideCounter(previous, 'followSkip'));
          setStep(run, decision.toIndex, addSkipped(run.skipped, decision.skippedIds));
        } else {
          setStep(run, decision.toIndex);
        }
        // The large model re-checks the step the move relied on (the asked step, or the last `yes` step of a
        // skip) ON THE FRAME THE FOLLOWER JUDGED (and sees the goal on it too); a fresh capture would judge a
        // later state. Its `step_check: no` reverts to `fromIndex`.
        enqueueConfirm('step_done', { fromIndex: decision.fromIndex, toIndex: decision.toIndex, stepId: decision.stepId, followFrame });
        awaitingConfirm = true;
      }
    }

    // Stuck: the scene keeps changing (target_changed) but no step moves. Heartbeat/manual do not count. In
    // the sequential core a nomination IS the guide moving forward (the step is done and waits for the upper
    // model), so it counts as progress too — otherwise the scene moving would ask for a replan mid-step.
    const stuck = nextStuckCount(run.stuckCount, {
      trigger: answer.trigger,
      stepChanged: run.stepIndex !== stepBefore || awaitingConfirm,
    });
    run.stuckCount = stuck.count;
    if (stuck.stuck && !replanAsked && requestReplan(run, 'replan', checks)) {
      updateTrace((previous) => bumpGuideCounter(previous, 'stuckReplan'));
    }

    // A step_done confirm already holds the confirm slot (it outranks a goal check), so asking for one here
    // would be lost: the goal is re-checked on the next follow that sees it.
    if (awaitingConfirm) return;
    if (answer.goal_seen === 'yes' && run.completion === 'none') enqueueConfirm('goal_check', { followFrame });
  };

  /** The completion re-check, its trigger chosen when it fires (`recheckCall`: target_left once the target is lost). */
  const scheduleRecheck = (run: GuideRun, trigger: string, meta: CallMeta | undefined) => {
    later(() => {
      if (!isCurrent(run) || run.completion !== 'checking') return;
      const lost = optionsRef.current.liveTrack.current()?.state === 'lost';
      const recheck = recheckCall(trigger, meta, lost ? targetLeft(run) : null);
      enqueueConfirm(recheck.trigger, recheck.meta);
    }, COMPLETION_RECHECK_MS);
  };

  /** `askedIndex`: the step the screen showed when the confirm was dispatched (a replan's `current_step`). */
  const applyConfirm = (
    run: GuideRun,
    answer: GuideConfirmResponse,
    acceptance: Acceptance,
    meta: CallMeta,
    capturedAt: number,
    askedIndex: number
  ) => {
    // A replan is plan text, not geometry: it is installed whenever it answers this task's plan.
    const sameTaskPlan =
      answer.fence_echo.task_epoch === run.taskEpoch && isCurrent(run) && run.planId !== null;
    // A replan this client cannot execute is refused, not installed; the run retires with its source named.
    if (sameTaskPlan && acceptance !== 'rejected' && answer.replan && hasUnsupportedExecution(answer.replan.steps)) {
      retire({ notice: unsupportedPlanNotice(run.materials) });
      return;
    }
    const replanInstalled = sameTaskPlan && acceptance !== 'rejected' && installReplan(run, answer);

    if (acceptance !== 'bound') {
      // A confirm that could not be applied cannot commit a sequential candidate: release the one it was
      // asked about, so the wait ends and a later fresh-frame follow may nominate again.
      if (run.seq && meta.followFrame) {
        run.seq = dropSequentialAwaiting(run.seq, meta.followFrame.frameId);
        publishSequential(run);
      }
      if (meta.recheck && run.completion === 'checking') {
        run.completion = 'none';
        setView({ completion: 'none', notice: '완료 재확인이 다른 화면 기준이라 확정하지 않았습니다.' });
      }
      if (acceptance === 'text_only') setView({ textOnlyQualifier: textOnlyQualifier(capturedAt, Date.now()) });
      return;
    }

    const stepFields = answer as GuideConfirmResponse & ConfirmStepFields;
    const answeredStepId = stepFields.step_id ?? null;
    const answeredCheck = stepFields.step_check ?? null;
    if (!replanInstalled) {
      // The sequential core keeps only the current step's observation, so a future step's entry is dropped
      // rather than shown as if this client had judged the whole plan.
      const reported = run.seq ? currentStepChecks(run.steps, run.stepIndex, answer.step_checks ?? null) : answer.step_checks;
      if (reported) setView({ stepChecks: reported });
      else if (answeredStepId && answeredCheck && (run.seq === null || run.steps[run.stepIndex]?.id === answeredStepId)) {
        setView((previous) => ({ stepChecks: [
          ...(previous.stepChecks ?? []).filter((existing) => existing.step_id !== answeredStepId),
          { step_id: answeredStepId, visible: answeredCheck },
        ] }));
      }
    }

    // The sequential core's step decision comes from this answer alone: an exact-step `yes` on the frame the
    // candidate was nominated with commits the step and moves it exactly ONE position; every other answer
    // (no, unsure, a replay, a stale activation/revision, a different step) only releases the candidate.
    if (run.seq) {
      const settlement = settleSequential(run.seq, {
        frameId: meta.followFrame?.frameId ?? null,
        planRevision: run.planRevision ?? -1,
        stepIndex: run.stepIndex,
        stepCheck: answeredCheck,
        stepId: answeredStepId,
        stepCount: run.steps.length,
      });
      run.seq = settlement.state;
      publishSequential(run);
      if (settlement.outcome.kind === 'advance' || settlement.outcome.kind === 'final') {
        updateTrace((previous) => bumpGuideCounter(previous, 'sequentialAdvance'));
        if (settlement.outcome.kind === 'advance') setStep(run, settlement.outcome.toIndex);
        else setView({ notice: '마지막 단계가 상위 모델 확인으로 확정되었습니다. 목표 완료는 완료 확인이 따로 판정합니다.' });
      } else if (settlement.outcome.kind === 'hold') {
        updateTrace((previous) => bumpGuideCounter(previous, 'sequentialStale'));
      }
    }
    // Step and completion are decided apart: a `step_check: no` reverts even on a `visually_satisfied` answer.
    const decision = decideConfirm(
      {
        trigger: answer.trigger,
        goal_status: answer.goal_status,
        step_check: answeredCheck,
        step_id: answeredStepId,
        inferred_done: answer.inferred_done,
      },
      meta,
      {
        stepIndex: run.stepIndex,
        completion: run.completion,
        trackLost: optionsRef.current.liveTrack.current()?.state === 'lost',
        userDoneIds: run.userDone,
      }
    );

    // A question about where the object went is moot once its leaving was read as the goal reached.
    if (answer.needs_clarification && answer.clarification_prompt && !decision.inferred) {
      setView({ notice: answer.clarification_prompt });
    }

    if (
      run.materials !== null &&
      (run.skipped.length > 0 || run.stepIndex < run.steps.length - 1) &&
      (decision.completion === 'start_check' || decision.completion === 'confirmed')
    ) {
      run.completion = 'none';
      setView({
        completion: 'none',
        notice: '최종 모습이 보여도 미확인·건너뛴 절차가 있어 참고자료 완료를 확정하지 않았습니다. 대화로 해당 단계로 돌아가 확인해 주세요.',
      });
      return;
    }

    if (decision.completion === 'confirmed') {
      run.completion = 'confirmed';
      run.halted = true;
      clearTimers();
      gateRef.current = INITIAL_GATE_STATE;
      publishOverlay(run);
      setView({
        completion: 'confirmed',
        phase: 'completed',
        pending: null,
        notice: decision.inferred ? '대상이 화면 밖으로 나갔지만, 앞뒤 화면을 보고 완료된 것으로 판단했습니다.' : null,
      });
      return;
    }
    if (decision.completion === 'recheck_failed') {
      run.completion = 'none';
      setView({ completion: 'none', notice: '재확인에서 완료로 보이지 않았습니다. 안내를 계속합니다.' });
      return;
    }

    if (decision.step.kind === 'revert') {
      updateTrace((previous) => bumpGuideCounter(previous, 'confirmReversals'));
      setStep(run, decision.step.toIndex);
      setView({ notice: '확인해 보니 이전 단계가 아직 끝나지 않았습니다. 이전 단계로 돌아갑니다.' });
    }

    // replan/unsure_twice: the large model's checklist on this fresh frame places the screen directly (no second
    // sighting, no step_done re-check). A replan in the same answer wins: the checks judged the plan it replaces.
    // The sequential core never moves the screen from a checklist — only its own step_done confirmation does —
    // so a future step the judge reports as `yes` cannot advance or skip the execution here.
    const position: ConfirmPositionDecision = run.seq
      ? { kind: 'none' }
      : decideConfirmPosition(
          run.steps,
          askedIndex,
          run.stepIndex,
          answer.step_checks,
          replanInstalled || Boolean(answer.replan)
        );
    if (position.kind === 'advance') {
      updateTrace((previous) => bumpGuideCounter(previous, 'confirmAdvance'));
      setStep(run, position.toIndex, addSkipped(run.skipped, position.skippedIds));
    }

    if (decision.completion === 'start_check') {
      // The second, confirming check always looks at a fresh frame (that is its purpose).
      run.completion = 'checking';
      setView({ completion: 'checking' });
      scheduleRecheck(run, answer.trigger, meta);
    }

    if (decision.lostNotice) {
      setView({ notice: '대상을 놓쳤습니다. 화면에 대상을 다시 보여주거나 “대상 직접 지정”으로 다시 지정하세요.' });
    }
  };

  const handleCallError = (run: GuideRun, call: PendingCall, error: unknown) => {
    if (!(error instanceof ApiError)) {
      run.consecutiveFailures += 1;
      if (run.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        retire({ error: '안내 서버에 연결하지 못해 가이드를 멈췄습니다.', keepView: true });
      }
      return;
    }
    const body = (error instanceof IntentHttpError ? error.body : null) ?? {};
    if (error.status === 401) {
      optionsRef.current.onSessionExpired?.();
      retire({ error: '세션이 만료되었습니다. 가이드를 다시 시작해주세요.', keepView: true });
      return;
    }
    if (error.status === 409) {
      if (error.code === 'stale_plan' && isSupersededStale(body, run)) {
        // A talk or replan answer moved the plan on while this call was out: its question is moot. A completion
        // re-check is asked again on the current plan, or `checking` would never resolve.
        updateTrace((previous) => bumpGuideCounter(previous, 'staleSuperseded'));
        if (call.kind === 'confirm' && retryRecheckAfterStale(call.meta as CallMeta | undefined, run.completion)) {
          scheduleRecheck(run, call.trigger, call.meta as CallMeta | undefined);
        }
        return;
      }
      const message =
        error.code === 'task_changed'
          ? '작업 내용이 바뀌어 가이드를 멈췄습니다. 다시 시작해주세요.'
          : typeof body.plan_id === 'string' && body.plan_id === run.planId
            ? '서버의 계획이 갱신되었습니다. 가이드를 다시 시작해주세요.'
            : '다른 계획이 시작되어 이 가이드를 멈췄습니다.';
      retire({ error: message, keepView: true });
      return;
    }
    if (error.status === 503 && error.code === 'service_unavailable') {
      // Not busy: the lane or the follower is not configured. Retrying would only repeat the refusal.
      retire({ error: '안내 서버의 가이드(추종) 경로를 사용할 수 없어 가이드를 멈췄습니다.', keepView: true });
      return;
    }
    if (error.status === 503 || error.status === 429) {
      // Busy or rate-limited: try the same call again once the server says so (default 1 s).
      const wait = error.retryAfterMs ?? 1000;
      later(() => {
        if (isCurrent(run) && !run.halted && !gateRef.current[call.kind].pending) enqueueCall(call);
      }, wait);
      return;
    }
    run.consecutiveFailures += 1;
    if (run.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
      retire({ error: `안내 응답이 연속으로 실패해 가이드를 멈췄습니다(${error.code}).`, keepView: true });
    }
  };

  const runCall = useCallback(
    async (run: GuideRun, call: PendingCall) => {
      const opts = optionsRef.current;
      const startedAt = performance.now();
      const stage: GuideStageName = call.kind;
      const providerView: FollowProvider =
        call.lane === 'local' ? (optionsRef.current.followProvider === 'clef' ? 'clef' : 'local') : 'deepseek';
      const meta = (call.meta ?? {}) as CallMeta;
      // A follow-raised confirm's frame leaves the gate with this dispatch: from here on only this call (its
      // request body, and a 503/429 retry) holds the bytes; they go when it settles.
      let followFrameHeld: FollowFrame | null = meta.followFrame ?? null;
      if (call.meta && meta.followFrame) call.meta = { ...meta, followFrame: undefined };
      let dispatched = false;
      try {
        const video = opts.videoRef.current;
        const stepIndex = run.stepIndex;
        const activation = run.seq?.activation ?? null;
        const step = run.steps[stepIndex];
        if (!step || run.planId === null || run.planRevision === null || !run.sessionId) return;
        // The anchor is read at capture time: the snapshot here, the frame in the same synchronous turn.
        const snapshot = opts.liveTrack.current();
        const source: ConfirmFrameSource =
          call.kind === 'confirm'
            ? confirmFrameSource(call.trigger, meta, {
                planId: run.planId,
                planRevision: run.planRevision,
                runId: snapshot?.runId ?? null,
              })
            : { kind: 'fresh' };
        if (source.kind === 'drop') {
          // The plan or the tracking run changed since the follow: its question is moot, nothing is paid for.
          followFrameHeld = null;
          updateTrace((previous) => bumpGuideCounter(previous, 'confirmFollowFrameDropped'));
          return;
        }
        if (call.kind === 'confirm' && call.trigger === 'step_done' && run.seq && meta.followFrame) {
          // A sequential step_done confirm is only ever about the candidate it was asked with. Once that
          // candidate is resolved, or its step/revision/activation ended, the question is moot: drop it
          // instead of paying for an answer that could never move the current step.
          const askedFrame = meta.followFrame;
          const candidate = run.seq.awaiting.find((entry) => entry.frameId === askedFrame.frameId);
          const live =
            candidate !== undefined &&
            candidate.activation === run.seq.activation &&
            candidate.stepIndex === run.stepIndex &&
            candidate.planRevision === run.planRevision;
          if (!live) {
            if (candidate) {
              run.seq = dropSequentialAwaiting(run.seq, candidate.frameId);
              publishSequential(run);
            }
            followFrameHeld = null;
            updateTrace((previous) => bumpGuideCounter(previous, 'sequentialStale'));
            return;
          }
        }
        if (
          call.kind === 'confirm' &&
          (call.trigger === 'replan' || call.trigger === 'unsure_twice') &&
          replanDispatch(run.replanBudget, meta.replanEpoch, run.replanEpoch) === 'drop'
        ) {
          // A replan was installed after this one was asked (or the cap is spent): its slot comes back unused.
          run.replanBudget = { ...run.replanBudget, used: Math.max(0, run.replanBudget.used - 1) };
          updateTrace((previous) => bumpGuideCounter(previous, 'replanLimited'));
          refreshReplanBlock(run);
          return;
        }

        let frameId: string;
        let imageBase64: string;
        let capturedAt: number;
        let anchor: ReturnType<typeof buildAnchorRef>;
        let fenceRunId: string;
        let sentBox: AcceptedBox | null = null;
        let sentAppearance: AppearanceReference | null = null;
        if (source.kind === 'follow') {
          // The follow's own frame and AnchorRef snapshot: the confirm judges exactly what the follower saw.
          frameId = source.frame.frameId;
          imageBase64 = source.frame.imageBase64;
          capturedAt = source.frame.capturedAt;
          anchor = source.frame.anchor;
          fenceRunId = source.frame.runId;
        } else {
          if (!video || !snapshot) return; // no run to fence the request with
          anchor = buildAnchorRef(snapshot, anchorLabel(run.target, snapshot.target));
          sentBox =
            snapshot.state === 'tracking' && snapshot.box
              ? { runId: snapshot.runId, generation: snapshot.generation, box: { ...snapshot.box } }
              : null;
          // A follow also asks for the appearance grid inside the sent box, reduced from the same frozen
          // snapshot as the JPEG: the reference describes the very frame that is sent.
          const appearanceBox = call.kind === 'follow' && sentBox ? sentBox.box : null;
          const frame = await captureVideoFrameAsync(video, { mirror: opts.mirror, appearanceBox });
          if (!isCurrent(run)) return;
          if (appearanceBox && frame.appearanceGrid) {
            sentAppearance = {
              runId: snapshot.runId,
              trackId: snapshot.trackId,
              generation: snapshot.generation,
              grid: frame.appearanceGrid,
            };
          }
          frameId = newOpaqueId('guide');
          imageBase64 = frame.rawBase64;
          capturedAt = frame.capturedAt;
          fenceRunId = snapshot.runId;
        }
        if (run.seq && (activation !== run.seq.activation || stepIndex !== run.stepIndex)) return;

        run.intentSeq += 1;
        run.latestSeq = { ...run.latestSeq, [call.kind]: run.intentSeq };
        const stamp: SentStamp = {
          sessionId: run.sessionId,
          planId: run.planId,
          planRevision: run.planRevision,
          intentSeq: run.intentSeq,
          fence: { task_epoch: run.taskEpoch, run_id: fenceRunId },
          anchors: echoOf(anchor),
        };
        const scene = { frame_id: frameId, image_base64: imageBase64, label: '카메라 현재 화면' };
        setPendingView(call.kind, { stage, trigger: call.trigger, provider: providerView, startedAt });
        setView({ lastTrigger: call.trigger });
        // Only a FOLLOW restarts the heartbeat interval; a confirm never postpones the next follow.
        triggerRef.current = noteStageDispatch(triggerRef.current, call.kind, Date.now());
        if (call.trigger === 'target_moved') updateTrace((previous) => bumpGuideCounter(previous, 'targetMoved'));
        if (call.trigger === 'target_changed') updateTrace((previous) => bumpGuideCounter(previous, 'targetChanged'));
        if (source.kind === 'follow') updateTrace((previous) => bumpGuideCounter(previous, 'confirmFollowFrame'));

        dispatched = true;
        if (call.kind === 'follow') {
          const payload: GuideFollowRequest = {
            session_id: run.sessionId,
            consent_ai: true,
            scene,
            plan_id: run.planId,
            plan_revision: run.planRevision,
            current_step: step.id,
            anchors: anchor ? [anchor] : [],
            trigger: call.trigger as FollowTriggerKind,
            intent_seq: stamp.intentSeq,
            fence: stamp.fence,
          };
          const answer = await requestGuideFollow(payload, run.controller.signal);
          opts.onUsage?.({ kind: 'billed', stage, provider: answer.provider, model: answer.model ?? null, usage: answer.usage ?? null });
          if (!isCurrent(run)) {
            recordCall(stage, call.trigger, answer.provider, startedAt, 'rejected', null);
            return;
          }
          run.consecutiveFailures = 0;
          const acceptance = classify(run, 'follow', stamp, answer, false);
          recordCall(stage, call.trigger, answer.provider, startedAt, acceptance, null);
          // What a confirm this answer raises is sent with (held only until that confirm leaves or is superseded).
          const followFrame: FollowFrame = {
            frameId,
            imageBase64,
            capturedAt,
            anchor,
            runId: fenceRunId,
            planId: stamp.planId,
            planRevision: stamp.planRevision,
          };
          // The run's first frame with the target tracked is the "before" a later target_left confirm is judged
          // against: the starting state. A later one may already show the end state (the tracker can stick to
          // the face or a hand for seconds after the object is gone).
          if (sentBox && !run.firstSeen) run.firstSeen = followFrame;
          applyFollow(run, answer, acceptance, { box: sentBox, appearance: sentAppearance, stepIndex, activation }, capturedAt, followFrame);
        } else {
          const payload: GuideConfirmRequest & { current_step?: string } = {
            session_id: run.sessionId,
            consent_ai: true,
            scene,
            plan_id: run.planId,
            plan_revision: run.planRevision,
            user_goal: run.userGoal,
            anchors: anchor ? [anchor] : [],
            trigger: call.trigger as ConfirmTriggerKind,
            intent_seq: stamp.intentSeq,
            fence: stamp.fence,
          };
          // step_done names the step it checks (the one the client advanced from, or the last `yes` step of a
          // skip); a replan carries the follower's last checklist so the judge sees where the guide got stuck.
          if (call.trigger === 'step_done' && meta.stepId) payload.current_step = meta.stepId;
          // target_left: the run's first tracked frame, where the target left, and the step under way.
          if (call.trigger === 'target_left' && meta.beforeFrame) {
            payload.before_scene = {
              frame_id: meta.beforeFrame.frameId,
              image_base64: meta.beforeFrame.imageBase64,
              label: '대상 추적을 시작할 때의 화면',
            };
            if (meta.exitEdge) payload.exit_edge = meta.exitEdge;
            payload.current_step = step.id;
          }
          if (call.trigger === 'replan' || call.trigger === 'unsure_twice') {
            // The judge's step_checks start at the step the screen shows (applyConfirm places the screen from them).
            payload.current_step = step.id;
            if (meta.followChecks?.length) payload.follow_checks = meta.followChecks;
          }
          const answer = await requestGuideConfirm(payload, run.controller.signal);
          followFrameHeld = null;
          opts.onUsage?.({ kind: 'billed', stage, provider: answer.provider, model: answer.model ?? null, usage: answer.usage ?? null });
          if (!isCurrent(run)) {
            recordCall(stage, call.trigger, answer.provider, startedAt, 'rejected', null);
            return;
          }
          run.consecutiveFailures = 0;
          const acceptance = classify(run, 'confirm', stamp, answer, Boolean(answer.replan));
          recordCall(stage, call.trigger, answer.provider, startedAt, acceptance, null);
          applyConfirm(run, answer, acceptance, meta, capturedAt, stepIndex);
        }
      } catch (error) {
        if (dispatched && mayHaveBilled(error)) optionsRef.current.onUsage?.({ kind: 'unknown', stage, lane: call.lane });
        if (isAbort(error) || !isCurrent(run)) return;
        if (error instanceof CameraFrameUnavailableError) return; // transient; the next trigger retries
        recordCall(stage, call.trigger, null, startedAt, 'failed', error instanceof ApiError ? error.code : 'network');
        // A call that is not coming back (only a busy 503/429 is retried) can never commit its candidate:
        // release it so the wait ends and the next accepted follow may nominate again on a fresh frame.
        const retried =
          error instanceof ApiError &&
          (error.status === 429 || (error.status === 503 && error.code !== 'service_unavailable'));
        if (run.seq && call.trigger === 'step_done' && meta.followFrame && !retried) {
          run.seq = dropSequentialAwaiting(run.seq, meta.followFrame.frameId);
          publishSequential(run);
        }
        // A busy/rate-limited retry re-sends the SAME follow frame (the question is still about it).
        const retryCall: PendingCall = followFrameHeld ? { ...call, meta: { ...meta, followFrame: followFrameHeld } } : call;
        handleCallError(run, retryCall, error);
      } finally {
        if (!dispatched && run.seq && meta.followFrame) {
          run.seq = dropSequentialAwaiting(run.seq, meta.followFrame.frameId);
          publishSequential(run);
        }
        followFrameHeld = null;
        if (runRef.current === run) {
          gateRef.current = markSettled(gateRef.current, call.kind);
          setPendingView(call.kind, null);
          pumpRef.current();
        }
      }
    },
    // The helpers above read refs and stable callbacks only, so the dispatcher is created once.
    []
  );

  /** Wake the pump in `waitMs` (an earlier wake replaces a later one). */
  const wakePumpIn = (waitMs: number) => {
    const dueAt = performance.now() + waitMs;
    const current = gateTimerRef.current;
    if (current && current.dueAt <= dueAt) return;
    if (current) {
      window.clearTimeout(current.handle);
      timersRef.current.delete(current.handle);
    }
    const handle = later(() => {
      if (gateTimerRef.current?.handle === handle) gateTimerRef.current = null;
      pumpRef.current();
    }, waitMs);
    gateTimerRef.current = { handle, dueAt };
  };

  pumpRef.current = () => {
    const run = runRef.current;
    if (!run || run.halted) return;
    let soonest = Number.POSITIVE_INFINITY;
    let capped: { waitMs: number; limit: 'budget' | 'per_minute' } | null = null;
    for (const stage of GATE_STAGES) {
      const now = performance.now();
      const decision = decideDispatch(gateRef.current, stage, now, GATE_CONFIG, run.talkWaiting);
      if (decision.kind === 'dispatch') {
        gateRef.current = markDispatched(gateRef.current, decision.call, now, GATE_CONFIG);
        void runCall(run, decision.call);
      } else if (decision.kind === 'wait') {
        soonest = Math.min(soonest, decision.waitMs);
      } else if (decision.kind === 'capped') {
        soonest = Math.min(soonest, decision.waitMs);
        if (!capped || decision.waitMs < capped.waitMs) capped = { waitMs: decision.waitMs, limit: decision.limit };
      }
    }
    if (capped) {
      const seconds = Math.ceil(capped.waitMs / 1000);
      const text =
        capped.limit === 'budget'
          ? `AI 호출 한도(${Math.round(GATE_CONFIG.remoteWindowMs / 60_000)}분에 ${GATE_CONFIG.remoteBudget}회)에 도달했습니다. ${seconds}초 뒤에 이어서 확인합니다.`
          : `서버 호출 한도(1분에 ${GATE_CONFIG.remotePerMinute}회)에 도달했습니다. ${seconds}초 뒤에 이어서 확인합니다.`;
      if (!budgetCappedRef.current) {
        budgetCappedRef.current = true;
        updateTrace((previous) => bumpGuideCounter(previous, 'budgetCapped'));
      }
      setView({ budgetNotice: text });
    } else if (budgetCappedRef.current) {
      budgetCappedRef.current = false;
      setView({ budgetNotice: null });
    }
    if (Number.isFinite(soonest)) wakePumpIn(Math.max(1, soonest));
  };

  // --------------------------------------------------------------------------------------- the tick

  const tick = useCallback(() => {
    const run = runRef.current;
    if (!run || run.halted || run.planId === null) return;
    const opts = optionsRef.current;
    const snapshot = opts.liveTrack.current();
    const nowWall = Date.now();
    const video = opts.videoRef.current;
    const aspect = video && video.videoWidth > 0 ? video.videoHeight / video.videoWidth : 1;
    const step = stepTriggers(triggerRef.current, {
      observation: snapshot
        ? { runId: snapshot.runId, trackId: snapshot.trackId, generation: snapshot.generation, state: snapshot.state, box: snapshot.box }
        : null,
      nowMs: nowWall,
      fenceKey: opts.fenceKey,
      accepted: run.accepted,
      followProvider: opts.followProvider,
      frameAspect: aspect,
    });
    triggerRef.current = step.state;

    const talkReason = describeTalkBlock(
      decideTalk({ running: true, talkPending: run.talkPending, gate: gateRef.current, nowMs: performance.now(), config: GATE_CONFIG }).block,
      GATE_CONFIG
    );
    if (talkReason !== talkBlockRef.current) {
      talkBlockRef.current = talkReason;
      setView({ talkBlockedReason: talkReason });
    }
    const notice = noticeRef.current;
    if (notice && noticeExpired(notice, run.stepIndex, performance.now())) {
      noticeRef.current = null;
      setView((previous) => (previous.notice === notice.text ? { notice: null } : {}));
    }

    if (snapshot?.state === 'tracking') {
      if (snapshot.box) run.lastBox = { ...snapshot.box };
      const drawable = isLiveBoxDrawable(snapshot, nowWall);
      updateTrace((previous) => sampleGuideHide(previous, !drawable), false);
      if (drawable && !run.firstBoxSeen) {
        run.firstBoxSeen = true;
        updateTrace((previous) => markGuideMilestone(previous, 'msStartToFirstBox', performance.now() - run.startedAt));
      }
    }
    if (performance.now() - lastTracePublishRef.current >= TRACE_PUBLISH_MS) updateTrace((previous) => previous);

    // target_changed: the appearance inside the (padded) live box vs the frame of the last accepted follow.
    // The sample is frozen and reduced in the capture worker, so it lands a few ms after this tick; one is
    // in flight at a time. It is stepped with the time it was TAKEN and the reference it was taken
    // against, and dropped if the guide run was retired or that reference was replaced meanwhile (an
    // `acquired`, a retarget or a newer accepted answer), so a sample never scores against a reference
    // that no longer stands.
    const nowPerf = performance.now();
    let appearanceFired = false;
    if (!appearanceSamplingRef.current && appearanceSampleDue(appearanceRef.current, nowPerf)) {
      const drawableBox =
        snapshot && snapshot.state === 'tracking' && snapshot.box && video && isLiveBoxDrawable(snapshot, nowWall)
          ? snapshot.box
          : null;
      if (drawableBox && snapshot && video) {
        const anchor = { runId: snapshot.runId, trackId: snapshot.trackId, generation: snapshot.generation };
        const reference = run.appearanceRef;
        appearanceSamplingRef.current = true;
        void sampleVideoAppearanceAsync(video, drawableBox, opts.mirror)
          // Skipped (the tracking lane is behind, or a newer sample replaced it): no sample at all, not a
          // "no evidence" one — the streak stands and the next tick tries again.
          .catch((error: unknown) => (error instanceof CaptureSkippedError ? SKIPPED : null))
          .then((grid) => {
            appearanceSamplingRef.current = false;
            if (grid === SKIPPED) return;
            if (!isCurrent(run) || run.halted || run.appearanceRef !== reference) return;
            const sampled = stepAppearance(appearanceRef.current, { nowMs: nowPerf, anchor, grid, reference });
            appearanceRef.current = sampled.state;
            if (sampled.fired) enqueueFollow('target_changed');
          });
      } else {
        const sampled = stepAppearance(appearanceRef.current, {
          nowMs: nowPerf,
          anchor: null,
          grid: null,
          reference: run.appearanceRef,
        });
        appearanceRef.current = sampled.state;
        appearanceFired = sampled.fired;
      }
    }

    for (const fired of step.fired) {
      if (fired === 'acquired') {
        // Bind the role to the anchor the store reports now; a fresh verdict is pending.
        run.binding = bindingFromSnapshot(snapshot);
        run.warn = false;
        run.accepted = null;
        run.appearanceRef = null;
        run.firstSeen = null;
        run.lastBox = null;
        publishOverlay(run);
        setView({ needsReselect: false });
        enqueueFollow('acquired');
      } else if (fired === 'anchor_lost') {
        // With a frame from before the loss, the judge can infer the goal was reached out of view.
        const left = targetLeft(run);
        if (left) {
          enqueueConfirm('target_left', left);
        } else {
          enqueueConfirm('goal_check');
        }
      } else {
        enqueueFollow(fired);
      }
    }
    if (appearanceFired) enqueueFollow('target_changed');
  }, [enqueueConfirm, enqueueFollow, publishOverlay, setView, updateTrace]);

  // The store's publishes drive the tick at frame rate; the interval covers silence (lost, heartbeat).
  useEffect(() => {
    const unsubscribe = options.liveTrack.subscribe(tick);
    const interval = window.setInterval(tick, TICK_MS);
    return () => {
      unsubscribe();
      window.clearInterval(interval);
    };
  }, [options.liveTrack, tick]);

  // ---------------------------------------------------------------------------------------- actions

  const start = useCallback(async () => {
    const opts = optionsRef.current;
    if (runRef.current && runRef.current.planId === null) return; // a plan is already being made
    const goal = opts.userGoal.trim();
    if (!opts.consent) {
      setView({ ...INITIAL_VIEW, phase: 'error', error: 'AI 동의가 필요합니다.' });
      return;
    }
    if (!goal) {
      setView({ ...INITIAL_VIEW, phase: 'error', error: '목표를 먼저 입력해주세요.' });
      return;
    }
    const video = opts.videoRef.current;
    if (!video || !opts.cameraLive) {
      setView({ ...INITIAL_VIEW, phase: 'error', error: '카메라가 켜져 있어야 가이드를 시작할 수 있습니다.' });
      return;
    }

    retire();
    // The previous run (if any) is retired too: the new plan names its own target.
    opts.stopTracking();
    const startedAt = performance.now();
    // Materials are optional: an empty selection is a goal-only (ungrounded) run, never a blocking error.
    const materials =
      opts.materials && opts.materials.length > 0 ? opts.materials.map((material) => ({ ...material })) : null;
    const run: GuideRun = {
      id: (guideRunCounter += 1),
      taskEpoch: newOpaqueId('task'),
      sessionId: null,
      userGoal: goal,
      context: opts.context.trim() || null,
      materials,
      planModel: opts.planModel,
      coreMode: opts.coreMode,
      controller: new AbortController(),
      startedAt,
      planId: null,
      planRevision: null,
      steps: [],
      goalWhen: '',
      target: '',
      stepIndex: 0,
      seq: null,
      intentSeq: 0,
      latestSeq: { follow: 0, confirm: 0, talk: 0 },
      unsureStreak: 0,
      lastChecks: null,
      skipped: [],
      userDone: [],
      talkPending: false,
      talkWaiting: false,
      lastTalk: null,
      stuckCount: 0,
      replanBudget: freshReplanBudget(startedAt),
      replanEpoch: 0,
      consecutiveFailures: 0,
      completion: 'none',
      halted: false,
      binding: null,
      accepted: null,
      appearanceRef: null,
      warn: false,
      firstBoxSeen: false,
      firstSeen: null,
      lastBox: null,
    };
    runRef.current = run;
    traceRef.current = startGuideTrace(Date.now());
    setTrace(traceRef.current);
    setView({
      ...INITIAL_VIEW,
      phase: 'planning',
      coreMode: run.coreMode,
      pending: { stage: 'plan', trigger: 'start', provider: 'deepseek', startedAt },
      lastTrigger: 'start',
    });

    let dispatched = false;
    try {
      const capture = captureVideoFrameAsync(video, { mirror: opts.mirror });
      const sessionId = await opts.ensureSession();
      const frame = await capture;
      if (!isCurrent(run)) return;
      run.sessionId = sessionId;
      const onPartial = (partial: PlanPartial) => {
        if (!isCurrent(run)) return;
        if (partial.target !== undefined) setView({ partialTarget: partial.target });
        if (partial.firstSay !== undefined) {
          updateTrace((previous) => markGuideMilestone(previous, 'msStartToFirstSay', performance.now() - startedAt));
          setView({ partialFirstSay: partial.firstSay });
        }
      };
      dispatched = true;
      // The plan stage is exclusive and counts against the shared paid-call budget and spacing.
      gateRef.current = markPlan(gateRef.current, true, performance.now(), GATE_CONFIG);
      const plan: GuidePlanResponse = await requestGuidePlan(
        {
          session_id: sessionId,
          consent_ai: true,
          scene: { frame_id: newOpaqueId('guide'), image_base64: frame.rawBase64, label: '카메라 현재 화면' },
          user_goal: goal,
          context: run.context,
          plan_model: run.planModel,
          core_mode: run.coreMode,
          ...(run.materials ? { materials: run.materials } : {}),
        },
        { signal: run.controller.signal, onPartial }
      );
      opts.onUsage?.({ kind: 'billed', stage: 'plan', provider: plan.provider, model: plan.model ?? null, usage: plan.usage ?? null });
      if (runRef.current === run) gateRef.current = markPlan(gateRef.current, false, performance.now(), GATE_CONFIG);
      if (!isCurrent(run)) {
        recordCall('plan', 'start', plan.provider, startedAt, 'rejected', null);
        return;
      }
      updateTrace((previous) => markGuideMilestone(previous, 'msStartToFinalPlan', performance.now() - startedAt));
      const target = plan.selection.status === 'selected' ? (plan.selection.target ?? '').trim() : '';
      if (!target || plan.needs_clarification || plan.steps.length === 0) {
        recordCall('plan', 'start', plan.provider, startedAt, 'bound', null);
        const why =
          plan.clarification_prompt ??
          plan.selection.rationale ??
          '이 화면에서 다룰 대상을 고르지 못했습니다. 대상을 화면에 보여주고 다시 시작해주세요.';
        retire({ notice: why });
        return;
      }
      // Fail closed on an execution policy this demo cannot run (no tracking start, no step auto-advance).
      if (hasUnsupportedExecution(plan.steps)) {
        recordCall('plan', 'start', plan.provider, startedAt, 'bound', null);
        retire({ notice: unsupportedPlanNotice(materials) });
        return;
      }
      // The server echoes the core it built this plan for, and that choice is fixed for the plan's whole
      // task. A plan for the OTHER core is a different task, not one to run here: the client keeps the run's
      // selected core and refuses instead of silently switching policy under the same plan.
      if (plan.core_mode !== run.coreMode) {
        recordCall('plan', 'start', plan.provider, startedAt, 'bound', null);
        retire({
          notice: '서버가 선택한 코어와 다른 코어로 만든 계획을 돌려주었습니다. 그 계획으로는 시작하지 않았습니다. 코어를 확인하고 다시 시작해 주세요.',
        });
        return;
      }
      recordCall('plan', 'start', plan.provider, startedAt, 'bound', null);
      run.planId = plan.plan_id;
      run.planRevision = plan.plan_revision;
      run.steps = plan.steps;
      run.goalWhen = plan.goal_when;
      run.target = target;
      // The plan's own core was checked against the selected one above; the run keeps its selected core.
      run.seq = run.coreMode === 'sequential' ? freshSequentialState() : null;
      run.replanBudget = freshReplanBudget(performance.now());
      // Without a streamed first sentence, the final plan is when the first sentence appears.
      updateTrace((previous) => markGuideMilestone(previous, 'msStartToFirstSay', performance.now() - startedAt));
      setView({
        phase: 'running',
        pending: null,
        coreMode: run.coreMode,
        sequentialPending: null,
        plan: { planId: plan.plan_id, planRevision: plan.plan_revision, steps: plan.steps, goalWhen: plan.goal_when, target,
          materials: run.materials, userGoal: run.userGoal },
        stepIndex: 0,
        skippedStepIds: [],
        partialTarget: null,
        partialFirstSay: null,
      });
      refreshReplanBlock(run);
      publishOverlay(run);
      await opts.startTracking(target);
    } catch (error) {
      if (runRef.current === run) gateRef.current = markPlan(gateRef.current, false, performance.now(), GATE_CONFIG);
      if (dispatched && mayHaveBilled(error)) opts.onUsage?.({ kind: 'unknown', stage: 'plan', lane: 'remote' });
      if (isAbort(error) || !isCurrent(run)) return;
      recordCall('plan', 'start', null, startedAt, 'failed', error instanceof ApiError ? error.code : 'network');
      if (error instanceof ApiError && error.status === 401) opts.onSessionExpired?.();
      const message =
        error instanceof CameraFrameUnavailableError
          ? error.message
          : error instanceof ApiError
            ? describePlanError(error)
            : error instanceof Error
              ? error.message
              : '계획을 받지 못했습니다.';
      retire({ error: message });
    }
  }, [publishOverlay, retire, setView, updateTrace]);

  const stop = useCallback(() => {
    retire();
  }, [retire]);

  const manual = useCallback(() => {
    const run = runRef.current;
    if (!run || run.halted || run.planId === null) return;
    enqueueFollow('manual');
  }, [enqueueFollow]);

  /** "계획 다시 짜기": confirm(replan) on a fresh frame, within the per-run replan budget. */
  const replan = useCallback(() => {
    const run = runRef.current;
    if (!run || run.halted || run.planId === null) return;
    requestReplan(run, 'replan', run.lastChecks);
    // The helpers read refs and stable callbacks only (same pattern as the dispatcher).
  }, []);

  const confirmCompletion = useCallback(() => {
    const run = runRef.current;
    if (!run || run.completion !== 'confirmed' || run.planId === null) return;
    run.completion = 'user_confirmed';
    setView({ completion: 'user_confirmed' });
    optionsRef.current.onUserConfirmedCompletion?.({ planId: run.planId, userGoal: run.userGoal });
  }, [setView]);

  // ------------------------------------------------------------------------------------------- talk

  const talkStateOf = (run: GuideRun, planRevision: number): TalkRunState => ({
    steps: run.steps,
    stepIndex: run.stepIndex,
    planRevision,
    target: run.target,
    skipped: run.skipped,
    userDone: run.userDone,
    unsureStreak: run.unsureStreak,
    stuckCount: run.stuckCount,
    lastChecks: run.lastChecks,
    replanBudget: run.replanBudget,
  });

  const adoptTalkState = (run: GuideRun, next: TalkRunState) => {
    run.steps = next.steps;
    run.stepIndex = next.stepIndex;
    run.planRevision = next.planRevision;
    run.target = next.target;
    run.skipped = next.skipped;
    run.userDone = next.userDone;
    run.unsureStreak = next.unsureStreak;
    run.stuckCount = next.stuckCount;
    run.lastChecks = next.lastChecks;
    run.replanBudget = next.replanBudget;
  };

  /** Send one utterance (the gate already passed) and apply the answer. Never retried: the user says it again. */
  const runTalk = useCallback(
    async (run: GuideRun, text: string, firstWaitMs: number, accepted: { planId: string; stepId: string; doneWhen: string }) => {
      const opts = optionsRef.current;
      const startedAt = performance.now();
      let holdsReplan = false;
      run.talkPending = true;
      setView({ talkPendingSince: startedAt, talkError: null });
      let dispatched = false;
      try {
        // A floor or the spacing only delays the utterance. Meanwhile the remote lane is held, so no paid call can
        // restart the floor in front of it (glass e2e 2026-10-02: a talk starved behind confirms until the clip
        // ended); only a budget spent meanwhile refuses it.
        run.talkWaiting = true;
        if (firstWaitMs > 0) {
          await new Promise<void>((resolve) => later(() => resolve(), firstWaitMs));
          if (!isCurrent(run) || run.halted) return;
          const again = decideTalk({ running: true, talkPending: false, gate: gateRef.current, nowMs: performance.now(), config: GATE_CONFIG });
          if (again.block) {
            setView({ talkError: describeTalkBlock(again.block, GATE_CONFIG) });
            return;
          }
        }
        const video = opts.videoRef.current;
        const snapshot = opts.liveTrack.current();
        if (!video || !snapshot || !run.sessionId) {
          setView({ talkError: '대상 추적이 시작된 뒤에 말할 수 있습니다.' });
          return;
        }
        const capture = captureVideoFrameAsync(video, { mirror: opts.mirror });
        const anchor = buildAnchorRef(snapshot, anchorLabel(run.target, snapshot.target));
        const frame = await capture;
        if (!isCurrent(run) || run.halted || run.planId === null || run.planRevision === null) return;
        // The step the user was looking at when they pressed send, even if a follow moved the screen since.
        const sentStepId = talkSendStep(accepted, { planId: run.planId, steps: run.steps });
        if (sentStepId === null) {
          setView({ talkError: '말하는 사이 계획이 바뀌어 보내지 않았습니다. 다시 말해 주세요.' });
          return;
        }
        // A replan-capable talk holds one replan slot until it ends; with none left it may not replan at all.
        const reservation = reserveReplan(run.replanBudget);
        run.replanBudget = reservation.budget;
        holdsReplan = reservation.reserved;
        refreshReplanBlock(run);

        gateRef.current = markTalk(gateRef.current, performance.now(), GATE_CONFIG);
        run.talkWaiting = false;
        pumpRef.current();
        run.intentSeq += 1;
        run.latestSeq = { ...run.latestSeq, talk: run.intentSeq };
        const stamp: SentStamp = {
          sessionId: run.sessionId,
          planId: run.planId,
          planRevision: run.planRevision,
          intentSeq: run.intentSeq,
          fence: { task_epoch: run.taskEpoch, run_id: snapshot.runId },
          anchors: echoOf(anchor),
        };
        const payload: GuideTalkRequest = {
          session_id: run.sessionId,
          consent_ai: true,
          scene: { frame_id: newOpaqueId('guide'), image_base64: frame.rawBase64, label: '카메라 현재 화면' },
          plan_id: run.planId,
          plan_revision: run.planRevision,
          current_step: sentStepId,
          anchors: anchor ? [anchor] : [],
          utterance: text,
          // No replan slot could be held: the model is offered no replan at all (the server refuses one).
          replan_allowed: holdsReplan,
          intent_seq: stamp.intentSeq,
          fence: stamp.fence,
        };
        if (run.lastTalk) {
          payload.prev_utterance = run.lastTalk.utterance;
          payload.prev_reply = run.lastTalk.reply;
        }
        if (run.lastChecks && run.lastChecks.length > 0) payload.follow_checks = run.lastChecks;
        dispatched = true;
        updateTrace((previous) => bumpGuideCounter(previous, 'talk'));
        const answer = await requestGuideTalk(payload, run.controller.signal);
        opts.onUsage?.({ kind: 'billed', stage: 'talk', provider: answer.provider, model: answer.model ?? null, usage: answer.usage ?? null });
        if (!isCurrent(run)) {
          recordCall('talk', 'talk', answer.provider, startedAt, 'rejected', null);
          return;
        }
        const acceptance = classify(run, 'talk', stamp, answer, talkChangesPlan(answer));
        recordCall('talk', 'talk', answer.provider, startedAt, acceptance, null);
        const outcome = applyTalk(talkStateOf(run, run.planRevision ?? stamp.planRevision), sentStepId, answer, acceptance, performance.now());
        if (!outcome.applied) {
          updateTrace((previous) => bumpGuideCounter(previous, 'talkRejected'));
          setView({ talkError: '답이 지금 화면·계획과 맞지 않아 적용하지 않았습니다. 다시 말해 주세요.' });
          return;
        }
        // A talk replan this client cannot execute is refused, not adopted; the run retires with its source named.
        if (outcome.actions.includes('replan') && hasUnsupportedExecution(outcome.state.steps)) {
          retire({ notice: unsupportedPlanNotice(run.materials) });
          return;
        }
        for (const taken of outcome.actions) {
          const counter = TALK_ACTION_COUNTER[taken];
          if (counter) updateTrace((previous) => bumpGuideCounter(previous, counter));
        }
        const stepBefore = run.stepIndex;
        adoptTalkState(run, outcome.state);
        if (run.seq) {
          // A rewritten plan starts the step machine over; a user reposition is a new step activation.
          // Either way, a candidate raised for the previous one can never move the step it left.
          run.seq =
            answer.replan || answer.step_say
              ? resetSequentialForPlan(run.seq)
              : run.stepIndex !== stepBefore || Boolean(answer.step_mark || answer.go_to || answer.target)
                ? noteStepActivation(run.seq)
                : run.seq;
          publishSequential(run);
        }
        run.lastTalk = { utterance: text, reply: answer.reply };
        let notice: string | null = null;
        for (const effect of outcome.effects) {
          if (effect.kind === 'retarget') {
            // The new run binds on its own `acquired`; until then nothing is drawn and follows go without an anchor.
            run.binding = null;
            run.accepted = null;
            run.appearanceRef = null;
            run.warn = false;
            void opts.startTracking(effect.target);
          } else if (effect.kind === 'checkGoal') {
            if (run.completion === 'none') enqueueConfirm('goal_check');
          } else {
            notice = effect.text;
          }
        }
        publishOverlay(run);
        setView((previous) => ({
          plan: previous.plan
            ? { ...previous.plan, steps: run.steps, planRevision: run.planRevision ?? previous.plan.planRevision, target: run.target }
            : previous.plan,
          stepIndex: run.stepIndex,
          skippedStepIds: run.skipped,
          userDoneStepIds: run.userDone,
          stepChecks: outcome.actions.includes('replan') ? null : previous.stepChecks,
          needsReselect: outcome.actions.includes('target') ? false : previous.needsReselect,
          talk: { utterance: text, reply: answer.reply, spoken: answer.spoken, at: Date.now() },
          talkError: null,
          ...(notice ? { notice } : {}),
          ...(run.stepIndex !== stepBefore ? { textOnlyQualifier: null } : {}),
        }));
        if (outcome.actions.includes('replan')) run.replanEpoch += 1;
      } catch (error) {
        if (dispatched && mayHaveBilled(error)) optionsRef.current.onUsage?.({ kind: 'unknown', stage: 'talk', lane: 'remote' });
        if (isAbort(error) || !isCurrent(run)) return;
        if (error instanceof CameraFrameUnavailableError) {
          setView({ talkError: error.message });
          return;
        }
        recordCall('talk', 'talk', null, startedAt, 'failed', error instanceof ApiError ? error.code : 'network');
        if (error instanceof ApiError && error.status === 401) {
          optionsRef.current.onSessionExpired?.();
          retire({ error: '세션이 만료되었습니다. 가이드를 다시 시작해주세요.', keepView: true });
          return;
        }
        if (error instanceof ApiError && error.status === 409 && error.code === 'task_changed') {
          retire({ error: '작업 내용이 바뀌어 가이드를 멈췄습니다. 다시 시작해주세요.', keepView: true });
          return;
        }
        setView({
          talkError: describeTalkError(error instanceof ApiError ? { status: error.status, code: error.code, reason: error.reason } : null),
        });
      } finally {
        if (holdsReplan) {
          run.replanBudget = releaseReplan(run.replanBudget);
          refreshReplanBlock(run);
        }
        if (runRef.current === run) {
          run.talkPending = false;
          setView({ talkPendingSince: null });
          if (run.talkWaiting) {
            run.talkWaiting = false;
            pumpRef.current();
          }
        }
      }
    },
    // The helpers read refs and stable callbacks only (same pattern as the dispatcher).
    []
  );

  /**
   * The user's words to the guide. True when the utterance is on its way (the input may be cleared); false when
   * it was not accepted — the reason is in `view.talkBlockedReason`. Never queued.
   */
  const talk = useCallback(
    (utterance: string): boolean => {
      const run = runRef.current;
      const text = utterance.trim().slice(0, TALK_UTTERANCE_MAX);
      if (!text) return false;
      const decision = decideTalk({
        running: Boolean(run && !run.halted && run.planId !== null),
        talkPending: Boolean(run?.talkPending),
        gate: gateRef.current,
        nowMs: performance.now(),
        config: GATE_CONFIG,
      });
      if (!run || decision.block) {
        updateTrace((previous) => bumpGuideCounter(previous, 'talkBlocked'));
        const reason = describeTalkBlock(decision.block ?? 'not_running', GATE_CONFIG);
        talkBlockRef.current = reason;
        setView({ talkBlockedReason: reason });
        return false;
      }
      const step = run.steps[run.stepIndex];
      if (!step || run.planId === null) return false;
      void runTalk(run, text, decision.waitMs, { planId: run.planId, stepId: step.id, doneWhen: step.done_when });
      return true;
    },
    [runTalk, setView, updateTrace]
  );

  // Stamp each notice with the step it was raised on; the tick clears it on a step change or after 8 s.
  useEffect(() => {
    noticeRef.current = view.notice ? { text: view.notice, stepIndex: view.stepIndex, at: performance.now() } : null;
    // Only a new notice restamps; the step the guide is on when it appears is read from the same view.
  }, [view.notice]);

  // A hard fence (task text, camera, mirror, scene mode, plan model, consent, camera loss) retires the guide.
  const fenceSignature = `${options.fenceKey}|${options.consent}|${options.cameraLive}`;
  const fenceRef = useRef(fenceSignature);
  useEffect(() => {
    if (fenceRef.current === fenceSignature) return;
    fenceRef.current = fenceSignature;
    if (runRef.current) retire({ notice: '작업·카메라·모델·동의가 바뀌어 가이드를 멈췄습니다.' });
  }, [fenceSignature, retire]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      const run = runRef.current;
      runRef.current = null;
      run?.controller.abort();
      for (const handle of timersRef.current) window.clearTimeout(handle);
      timersRef.current.clear();
    };
  }, []);

  return { view, trace, overlay, start, stop, manual, replan, confirmCompletion, talk };
}

export type IntentLoop = ReturnType<typeof useIntentLoop>;

function describePlanError(error: ApiError): string {
  switch (error.code) {
    case 'service_unavailable':
      return '가이드(계획) 서버를 사용할 수 없습니다. 선택한 계획 모델의 API 설정을 확인해 주세요.';
    case 'provider_busy':
      return '이전 계획 요청이 아직 진행 중입니다. 잠시 후 다시 시도해주세요.';
    case 'rate_limited':
      return '요청이 너무 잦습니다. 잠시 후 다시 시도해주세요.';
    case 'ai_consent_required':
      return 'AI 동의가 필요합니다.';
    case 'invalid_provider_output':
      return `모델 응답이 계약을 통과하지 못했습니다${error.reason ? `(${error.reason})` : ''}. 다시 시도해주세요.`;
    case 'session_expired':
    case 'session_mismatch':
      return '세션이 만료되었습니다. 다시 시도해주세요.';
    default:
      return `계획을 받지 못했습니다(${error.status} ${error.code}).`;
  }
}
