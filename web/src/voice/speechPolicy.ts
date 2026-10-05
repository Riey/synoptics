/**
 * What the guide says out loud, derived from consecutive `IntentView` snapshots (no timers, no DOM).
 *
 * Spoken, each once, when it changes: the current step's `say` (the plan's accessible spoken fallback,
 * `STEP_ACTION_RULE`), the guide's talk reply, the loop's notices and errors, the completion states, and the
 * reselect warning. The spoken text is exactly what the narration bar / talk bar shows — speech is a second
 * channel for the same words, never new words. One exception, by design: a talk answer is read as its short
 * `spoken` line (one sentence the model wrote for the voice) plus a pointer to the screen when the shown reply
 * says more — a 400-character reply is far too long to listen to (operator, 2026-10-02).
 *
 * `priority`: a new step or a new reply (`replace`) cuts whatever is still being read, because the old
 * sentence is stale; notices and errors (`append`) queue behind the current sentence.
 */
import type { IntentView } from '../intent/useIntentLoop';

export interface SpokenLine {
  text: string;
  priority: 'replace' | 'append';
}

export const COMPLETION_SPEECH = '목표를 달성한 것으로 보입니다. 완료 확인을 눌러 주세요.';
export const USER_CONFIRMED_SPEECH = '완료를 확인했습니다. 수고하셨습니다.';
export const RESELECT_SPEECH = '대상이 다릅니다. 대상을 다시 지정하세요.';
export const PLANNING_SPEECH = '계획을 세우는 중입니다.';
export const TALK_DETAIL_SPEECH = '자세한 건 화면에 있어요.';

/** What the voice reads of a talk answer: the short spoken line, plus a pointer when the screen shows more. */
export function talkSpeech(talk: { reply: string; spoken: string }): string {
  const spoken = talk.spoken.trim() || talk.reply.trim();
  return talk.reply.trim() === spoken ? spoken : `${spoken} ${TALK_DETAIL_SPEECH}`;
}

/** The step sentence the narration bar shows for a running guide (`NarrationBar`), or null. */
export function stepSpeech(view: IntentView): string | null {
  if (view.phase !== 'running' && view.phase !== 'planning') return null;
  const say = view.plan?.steps[view.stepIndex]?.say ?? view.partialFirstSay;
  if (!say) return null;
  const total = view.plan?.steps.length;
  return `${view.stepIndex + 1}단계${total ? ` / ${total}` : ''}. ${say}`;
}

/** The lines to speak on the transition `previous` -> `next`; `previous === null` is the first snapshot. */
export function spokenLines(previous: IntentView | null, next: IntentView): SpokenLine[] {
  const lines: SpokenLine[] = [];
  const step = stepSpeech(next);
  // A step is read when its sentence changes: a new plan, the next step, a replan rewriting this step, or the
  // streamed first sentence settling into the installed plan. Not while the first sentence is still streaming
  // in character by character (speech would stutter): only once the plan is installed.
  if (step !== null && next.phase === 'running' && (previous === null || stepSpeech(previous) !== step || previous.phase !== 'running')) {
    lines.push({ text: step, priority: 'replace' });
  }
  if (next.phase === 'planning' && previous?.phase !== 'planning') {
    lines.push({ text: PLANNING_SPEECH, priority: 'replace' });
  }
  if (next.talk && next.talk.at !== previous?.talk?.at) {
    lines.push({ text: talkSpeech(next.talk), priority: 'replace' });
  }
  if (next.completion === 'confirmed' && previous?.completion !== 'confirmed') {
    lines.push({ text: COMPLETION_SPEECH, priority: 'append' });
  }
  if (next.completion === 'user_confirmed' && previous?.completion !== 'user_confirmed') {
    lines.push({ text: USER_CONFIRMED_SPEECH, priority: 'replace' });
  }
  if (next.needsReselect && !previous?.needsReselect) {
    lines.push({ text: RESELECT_SPEECH, priority: 'append' });
  }
  if (next.notice && next.notice !== previous?.notice) {
    lines.push({ text: next.notice, priority: 'append' });
  }
  if (next.talkError && next.talkError !== previous?.talkError) {
    lines.push({ text: next.talkError, priority: 'append' });
  }
  if (next.error && next.error !== previous?.error) {
    lines.push({ text: next.error, priority: 'append' });
  }
  return lines;
}
