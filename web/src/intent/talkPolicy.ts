/**
 * When a user utterance may be sent, how a refusal or failure is worded, and which stale answers a talk has
 * overtaken (design 2026-10-01 guide-talk §C/§E). Pure; the hook applies it.
 *
 * - Talk is a paid lane: one utterance waiting or in flight, never a queue (the input is locked meanwhile).
 * - It shares the run's paid-call budget (`remoteBudget` per `remoteWindowMs`), the per-minute mirror of the
 *   server ceiling, the paid-call spacing and the confirm stage's 2.2 s floor. A spent budget refuses the
 *   utterance (shown, not sent, not queued); a floor or the spacing only delays it by `waitMs`, and while it
 *   waits the hook holds every other paid call (`decideDispatch(..., remoteHeld)`) so it cannot starve.
 * - A follow/confirm answered `409 stale_plan` naming the plan the client is on, at the revision the client
 *   already holds (or a newer one while a talk is in flight to install it), was overtaken by a talk or replan
 *   answer: it is dropped, not treated as a lost plan.
 */
import { remoteCallsInWindow, type GateConfig, type GateState } from './triggers';

/** Mirror of the server's bound (`guide_contracts.TALK_UTTERANCE_MAX`). */
export const TALK_UTTERANCE_MAX = 200;

export type TalkBlock = 'not_running' | 'pending' | 'budget' | 'per_minute';

export interface TalkGateInput {
  running: boolean;
  talkPending: boolean;
  gate: GateState;
  nowMs: number;
  config: GateConfig;
}

export function decideTalk(input: TalkGateInput): { block: TalkBlock | null; waitMs: number } {
  const { gate, nowMs, config } = input;
  if (!input.running) return { block: 'not_running', waitMs: 0 };
  if (input.talkPending) return { block: 'pending', waitMs: 0 };
  if (remoteCallsInWindow(gate, nowMs, config) >= config.remoteBudget) return { block: 'budget', waitMs: 0 };
  if (remoteCallsInWindow(gate, nowMs, { ...config, remoteWindowMs: 60_000 }) >= config.remotePerMinute) {
    return { block: 'per_minute', waitMs: 0 };
  }
  const waitMs = Math.max(0, gate.confirm.lastAt + config.confirmFloorMs - nowMs, gate.lastRemoteAt + config.remoteSpacingMs - nowMs);
  return { block: null, waitMs };
}

export function describeTalkBlock(block: TalkBlock | null, config: GateConfig): string | null {
  switch (block) {
    case 'not_running':
      return '가이드가 진행 중일 때만 말할 수 있습니다.';
    case 'pending':
      return '앞의 말에 대한 답을 기다리는 중입니다.';
    case 'budget':
      return `AI 호출 한도(${Math.round(config.remoteWindowMs / 60_000)}분에 ${config.remoteBudget}회)에 도달했습니다. 잠시 후 다시 말해 주세요.`;
    case 'per_minute':
      return `서버 호출 한도(1분에 ${config.remotePerMinute}회)에 도달했습니다. 잠시 후 다시 말해 주세요.`;
    default:
      return null;
  }
}

/** A failed talk, in words (`null` = no HTTP answer at all). 401 and 409 task_changed stop the guide instead. */
export function describeTalkError(error: { status: number; code: string; reason: string | null } | null): string {
  if (!error) return '안내 서버에 연결하지 못했습니다. 다시 말해 주세요.';
  if (error.status === 409) return '그사이 계획이 바뀌어 답을 적용하지 못했습니다. 다시 말해 주세요.';
  if (error.status === 429 || error.code === 'provider_busy') return '안내 서버가 바쁩니다. 잠시 후 다시 말해 주세요.';
  if (error.code === 'service_unavailable') return '대화 경로를 사용할 수 없습니다. 선택한 계획 모델의 API 설정을 확인해 주세요.';
  if (error.code === 'provider_timeout') return '답이 늦어 취소됐습니다. 다시 말해 주세요.';
  if (error.code === 'invalid_provider_output') {
    return `답을 이해하지 못했습니다${error.reason ? `(${error.reason})` : ''}. 다른 말로 다시 말해 주세요.`;
  }
  return `말을 보내지 못했습니다(${error.status} ${error.code}).`;
}

export function isSupersededStale(
  body: Record<string, unknown>,
  run: { planId: string | null; planRevision: number | null; talkPending: boolean }
): boolean {
  if (typeof body.plan_id !== 'string' || body.plan_id !== run.planId) return false;
  if (typeof body.plan_revision !== 'number' || run.planRevision === null) return false;
  if (body.plan_revision === run.planRevision) return true;
  return run.talkPending && body.plan_revision > run.planRevision;
}

/**
 * The step an utterance is sent about: the step on screen when the user pressed send, as long as the same plan
 * still has that step (same id and `done_when`; a sentence rewrite keeps it, a replan does not). A follow may
 * have moved the screen during the floor wait, but the user was talking about what they saw. `null`: the plan
 * changed under the utterance; it is not sent.
 */
export function talkSendStep(
  accepted: { planId: string; stepId: string; doneWhen: string },
  current: { planId: string | null; steps: ReadonlyArray<{ id: string; done_when: string }> }
): string | null {
  if (current.planId !== accepted.planId) return null;
  const step = current.steps.find((candidate) => candidate.id === accepted.stepId);
  return step && step.done_when === accepted.doneWhen ? step.id : null;
}
