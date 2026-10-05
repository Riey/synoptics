/**
 * Sanitized trace of the guide lane, for the in-app debug panel: milestone timings from "가이드 시작",
 * the newest guide calls and the counters that explain why a step did or did not move.
 *
 * Records are page memory only and carry no image and no model text. Everything here is pure, so the
 * accounting the panel shows can be exercised without a browser.
 *
 * The per-request trace of the removed live guidance lane (/api/guidance) lived here until 2026-10-01.
 */

// ------------------------------------------------------------------------------------ guide lane timings

/** How many guide calls the panel keeps (newest first). */
export const GUIDE_CALL_LIMIT = 12;

export type GuideCallAcceptance = 'bound' | 'text_only' | 'rejected' | 'failed';

/** One guide-lane call as the debug panel shows it. No image, no text from the model. */
export interface GuideCallRecord {
  id: string;
  /** plan | follow | confirm | talk */
  stage: 'plan' | 'follow' | 'confirm' | 'talk';
  trigger: string;
  /** 'deepseek' | 'local' | another provider token, or null when nothing came back. */
  provider: string | null;
  /** Request start (capture) → terminal answer, ms. */
  msToFinal: number | null;
  accepted: GuideCallAcceptance;
  /** Server error code for a failed call, else null. */
  errorCode: string | null;
  startedAtWall: number;
}

/**
 * Timings and counters of one guide run (from “가이드 시작”). Milestones are ms from the start press;
 * null until they happen. `hideSamples`/`hiddenSamples` sample the 300 ms freshness cut while tracking.
 */
export interface GuideTrace {
  startedAtWall: number | null;
  msStartToFirstSay: number | null;
  msStartToFinalPlan: number | null;
  msStartToFirstBox: number | null;
  calls: GuideCallRecord[];
  followDone: number;
  followUnsure: number;
  confirmReversals: number;
  /** Screen moves from a replan/unsure_twice confirm's `step_checks` (the large model placed the step). */
  confirmAdvance: number;
  targetMoved: number;
  /** `target_changed` follows dispatched (appearance inside the box changed). */
  targetChanged: number;
  /** Skips applied: a later step's done_when seen `yes` on two consecutive accepted follows. */
  followSkip: number;
  /** First sightings of a later step `yes` (recorded, the screen stayed). */
  followPendingSkip: number;
  /** confirm('replan') sent because target_changed follows kept coming with no step change. */
  stuckReplan: number;
  /** Replans (stuck, unsure_twice or manual) not sent: the per-run maximum or the replan gap (`replanPolicy.ts`). */
  replanLimited: number;
  /** Confirms a follow answer raised that were sent with that follow's own frame and anchor snapshot. */
  confirmFollowFrame: number;
  /** Such confirms dropped before dispatch: the plan revision or tracking run changed since the follow. */
  confirmFollowFrameDropped: number;
  /** Times a DeepSeek-backed call had to wait for the rolling budget. */
  budgetCapped: number;
  /** Utterances sent (`/api/guide/talk`). */
  talk: number;
  /** Applied talk answers by action. */
  talkSay: number;
  talkTarget: number;
  talkMark: number;
  talkGoTo: number;
  talkReplan: number;
  /** Talk answers discarded by the acceptance rule. */
  talkRejected: number;
  /** Utterances not sent: not running, one already waiting, or the DeepSeek budget. */
  talkBlocked: number;
  /** Follow/confirm 409 stale_plan answers dropped because a talk or replan had moved the plan on. */
  staleSuperseded: number;
  /** Sequential core: the follower's `yes` nominated the current step for the upper model's confirmation. */
  sequentialNominate: number;
  /** Sequential core: an upper `step_done: yes` committed the step (it advanced or was the last step). */
  sequentialAdvance: number;
  /** Sequential core: a candidate was dropped without moving the step (no/unsure, replay, stale, failure). */
  sequentialStale: number;
  hideSamples: number;
  hiddenSamples: number;
}

export const EMPTY_GUIDE_TRACE: GuideTrace = {
  startedAtWall: null,
  msStartToFirstSay: null,
  msStartToFinalPlan: null,
  msStartToFirstBox: null,
  calls: [],
  followDone: 0,
  followUnsure: 0,
  confirmReversals: 0,
  confirmAdvance: 0,
  targetMoved: 0,
  targetChanged: 0,
  followSkip: 0,
  followPendingSkip: 0,
  stuckReplan: 0,
  replanLimited: 0,
  confirmFollowFrame: 0,
  confirmFollowFrameDropped: 0,
  budgetCapped: 0,
  talk: 0,
  talkSay: 0,
  talkTarget: 0,
  talkMark: 0,
  talkGoTo: 0,
  talkReplan: 0,
  talkRejected: 0,
  talkBlocked: 0,
  staleSuperseded: 0,
  sequentialNominate: 0,
  sequentialAdvance: 0,
  sequentialStale: 0,
  hideSamples: 0,
  hiddenSamples: 0,
};

export function startGuideTrace(nowWall: number): GuideTrace {
  return { ...EMPTY_GUIDE_TRACE, startedAtWall: nowWall };
}

export type GuideMilestone = 'msStartToFirstSay' | 'msStartToFinalPlan' | 'msStartToFirstBox';

/** Record a milestone once: the first occurrence wins, later ones are ignored. Negative input clamps to 0. */
export function markGuideMilestone(trace: GuideTrace, milestone: GuideMilestone, ms: number): GuideTrace {
  if (trace[milestone] !== null || !Number.isFinite(ms)) return trace;
  return { ...trace, [milestone]: Math.max(0, Math.round(ms)) };
}

export function appendGuideCall(trace: GuideTrace, call: GuideCallRecord, limit: number = GUIDE_CALL_LIMIT): GuideTrace {
  return { ...trace, calls: [call, ...trace.calls].slice(0, limit) };
}

export type GuideCounter =
  | 'followDone'
  | 'followUnsure'
  | 'confirmReversals'
  | 'confirmAdvance'
  | 'targetMoved'
  | 'targetChanged'
  | 'followSkip'
  | 'followPendingSkip'
  | 'stuckReplan'
  | 'replanLimited'
  | 'confirmFollowFrame'
  | 'confirmFollowFrameDropped'
  | 'budgetCapped'
  | 'talk'
  | 'talkSay'
  | 'talkTarget'
  | 'talkMark'
  | 'talkGoTo'
  | 'talkReplan'
  | 'talkRejected'
  | 'talkBlocked'
  | 'staleSuperseded'
  | 'sequentialNominate'
  | 'sequentialAdvance'
  | 'sequentialStale';

export function bumpGuideCounter(trace: GuideTrace, counter: GuideCounter): GuideTrace {
  return { ...trace, [counter]: trace[counter] + 1 };
}

/** One sample of the freshness cut while the run was tracking: `hidden` = the box was not drawable. */
export function sampleGuideHide(trace: GuideTrace, hidden: boolean): GuideTrace {
  return { ...trace, hideSamples: trace.hideSamples + 1, hiddenSamples: trace.hiddenSamples + (hidden ? 1 : 0) };
}

/** Hidden share of the sampled tracking time, or null before any sample. */
export function guideHideRatio(trace: GuideTrace): number | null {
  return trace.hideSamples === 0 ? null : trace.hiddenSamples / trace.hideSamples;
}

/** Calls by acceptance outcome, for the panel's summary line. */
export function countGuideCalls(trace: GuideTrace): Record<GuideCallAcceptance, number> {
  const out: Record<GuideCallAcceptance, number> = { bound: 0, text_only: 0, rejected: 0, failed: 0 };
  for (const call of trace.calls) out[call.accepted] += 1;
  return out;
}
