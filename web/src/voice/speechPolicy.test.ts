/**
 * Which guide words are spoken on a view transition, and which cut the sentence being read. Loaded through
 * Vite (runtime sibling import), like the intent policies.
 */
import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const WEB_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const server = await createServer({
  configFile: false,
  root: WEB_ROOT,
  logLevel: 'silent',
  server: { middlewareMode: true, hmr: false, watch: null },
});
after(async () => {
  await server.close();
});
const { spokenLines, stepSpeech, COMPLETION_SPEECH, USER_CONFIRMED_SPEECH, RESELECT_SPEECH, PLANNING_SPEECH, TALK_DETAIL_SPEECH } =
  await server.ssrLoadModule('/src/voice/speechPolicy.ts');

const step = (id: string, say: string) => ({ id, say, commands: [], done_when: `${id} done` });
const plan = {
  planId: 'p1',
  planRevision: 1,
  steps: [step('s1', '안경을 집으세요'), step('s2', '안경을 쓰세요')],
  goalWhen: '안경을 쓴 상태',
  target: 'glasses',
};
const view = (overrides: Record<string, unknown> = {}) => ({
  phase: 'running',
  plan,
  stepIndex: 0,
  skippedStepIds: [],
  userDoneStepIds: [],
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
  ...overrides,
});
const texts = (lines: { text: string }[]) => lines.map((line) => line.text);

test('the step sentence is the narration bar wording', () => {
  assert.equal(stepSpeech(view()), '1단계 / 2. 안경을 집으세요');
  assert.equal(stepSpeech(view({ stepIndex: 1 })), '2단계 / 2. 안경을 쓰세요');
  assert.equal(stepSpeech(view({ phase: 'completed' })), null);
  assert.equal(stepSpeech(view({ phase: 'planning', plan: null, partialFirstSay: '안경을' })), '1단계. 안경을');
});

test('the first running snapshot reads the current step, and only a step change reads again', () => {
  const first = spokenLines(null, view());
  assert.deepEqual(first, [{ text: '1단계 / 2. 안경을 집으세요', priority: 'replace' }]);
  assert.deepEqual(spokenLines(view(), view()), []);
  assert.deepEqual(spokenLines(view(), view({ lastTrigger: 'heartbeat', pending: { stage: 'follow' } })), []);
  assert.deepEqual(texts(spokenLines(view(), view({ stepIndex: 1 }))), ['2단계 / 2. 안경을 쓰세요']);
});

test('planning is announced once; the streamed first sentence is not read until the plan is installed', () => {
  const idle = view({ phase: 'idle', plan: null });
  const planning = view({ phase: 'planning', plan: null, partialFirstSay: '안경을 집' });
  assert.deepEqual(texts(spokenLines(idle, planning)), [PLANNING_SPEECH]);
  assert.deepEqual(spokenLines(planning, view({ phase: 'planning', plan: null, partialFirstSay: '안경을 집으세' })), []);
  assert.deepEqual(texts(spokenLines(planning, view())), ['1단계 / 2. 안경을 집으세요']);
});

test('a replan that rewrites the current step reads the new sentence at the same index', () => {
  const rewritten = { ...plan, planRevision: 2, steps: [step('s1', '안경을 손에 드세요'), plan.steps[1]] };
  assert.deepEqual(texts(spokenLines(view(), view({ plan: rewritten }))), ['1단계 / 2. 안경을 손에 드세요']);
});

test('a talk reply is read once, when it arrives, and cuts the sentence being read', () => {
  const before = view();
  const replied = view({ talk: { utterance: '이미 했어', reply: '다음 단계로 갈게요.', spoken: '다음 단계로 갈게요.', at: 10 } });
  assert.deepEqual(spokenLines(before, replied), [{ text: '다음 단계로 갈게요.', priority: 'replace' }]);
  assert.deepEqual(spokenLines(replied, replied), []);
  const again = view({ talk: { utterance: '이미 했어', reply: '다음 단계로 갈게요.', spoken: '다음 단계로 갈게요.', at: 20 } });
  assert.deepEqual(texts(spokenLines(replied, again)), ['다음 단계로 갈게요.']);
});

test('a long reply is not read out: the voice reads the short spoken line and points at the screen', () => {
  // Operator, 2026-10-02: the detailed reply is good on screen but too long to listen to.
  const reply = '뚜껑이 미끄러워 힘이 안 실려요.\n1. 마른 수건으로 감싸 돌리기\n2. 고무장갑 끼기\n3. 뚜껑만 따뜻한 물에 30초\n칼·가위는 쓰지 마세요.';
  const spoken = '마른 수건으로 뚜껑을 감싸 돌려 보세요.';
  const lines = spokenLines(view(), view({ talk: { utterance: '안 열려', reply, spoken, at: 10 } }));
  assert.deepEqual(lines, [{ text: `${spoken} ${TALK_DETAIL_SPEECH}`, priority: 'replace' }]);
  assert.ok(!lines[0].text.includes('고무장갑'));
  // A reply that is the spoken line itself gets no pointer.
  assert.deepEqual(texts(spokenLines(view(), view({ talk: { utterance: '응', reply: spoken, spoken, at: 11 } }))), [spoken]);
});

test('completion, the user confirmation, the reselect warning, notices and errors are each spoken once', () => {
  assert.deepEqual(spokenLines(view(), view({ completion: 'confirmed' })), [{ text: COMPLETION_SPEECH, priority: 'append' }]);
  assert.deepEqual(spokenLines(view({ completion: 'confirmed' }), view({ completion: 'confirmed' })), []);
  assert.deepEqual(texts(spokenLines(view({ completion: 'confirmed' }), view({ phase: 'completed', completion: 'user_confirmed' }))), [USER_CONFIRMED_SPEECH]);
  assert.deepEqual(texts(spokenLines(view(), view({ needsReselect: true }))), [RESELECT_SPEECH]);
  assert.deepEqual(spokenLines(view({ needsReselect: true }), view({ needsReselect: true })), []);
  assert.deepEqual(spokenLines(view(), view({ notice: '대상을 놓쳤습니다.' })), [{ text: '대상을 놓쳤습니다.', priority: 'append' }]);
  assert.deepEqual(texts(spokenLines(view(), view({ talkError: '답이 지금 화면·계획과 맞지 않아 적용하지 않았습니다.' }))), ['답이 지금 화면·계획과 맞지 않아 적용하지 않았습니다.']);
  assert.deepEqual(texts(spokenLines(view(), view({ phase: 'error', error: '세션이 만료되었습니다.' }))), ['세션이 만료되었습니다.']);
  assert.deepEqual(spokenLines(view({ notice: 'x' }), view({ notice: null })), []);
});

test('a step change and a notice in one transition keep the step first', () => {
  const lines = spokenLines(view(), view({ stepIndex: 1, notice: '새 1단계부터 안내합니다.' }));
  assert.deepEqual(lines.map((line) => line.priority), ['replace', 'append']);
});
