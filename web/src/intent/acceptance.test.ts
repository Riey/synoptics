/**
 * The bound / text_only / rejected acceptance rule (pure).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { captureAgeSeconds, classifyAnswer, textOnlyQualifier } from './acceptance.ts';

const FENCE = { task_epoch: 'task-1', run_id: 'run-1' };
const ANCHOR = { anchor_id: 'a1', track_id: 'trk-1', generation: 2 };

const sent = {
  sessionId: 'sess-1',
  planId: 'plan-1',
  planRevision: 3,
  intentSeq: 7,
  fence: FENCE,
  anchors: [ANCHOR],
};
const answer = { intent_seq: 7, plan_revision: 3, fence_echo: FENCE, anchors_echo: [ANCHOR] };
const current = {
  sessionId: 'sess-1',
  taskEpoch: 'task-1',
  runId: 'run-1',
  planId: 'plan-1',
  planRevision: 3,
  latestIntentSeq: 7,
  anchor: ANCHOR,
};

test('everything matching is bound', () => {
  assert.equal(classifyAnswer(sent, answer, current), 'bound');
});

test('same session, task, run and plan but a newer anchor generation is text_only', () => {
  assert.equal(classifyAnswer(sent, answer, { ...current, anchor: { ...ANCHOR, generation: 3 } }), 'text_only');
  assert.equal(classifyAnswer(sent, answer, { ...current, anchor: { ...ANCHOR, track_id: 'trk-2' } }), 'text_only');
  assert.equal(classifyAnswer(sent, answer, { ...current, anchor: null }), 'text_only');
});

test('a different task, run, plan, revision or session is rejected', () => {
  assert.equal(classifyAnswer(sent, answer, { ...current, taskEpoch: 'task-2' }), 'rejected');
  assert.equal(classifyAnswer(sent, answer, { ...current, taskEpoch: null }), 'rejected');
  assert.equal(classifyAnswer(sent, answer, { ...current, runId: 'run-2' }), 'rejected');
  assert.equal(classifyAnswer(sent, answer, { ...current, runId: null }), 'rejected');
  assert.equal(classifyAnswer(sent, answer, { ...current, planId: 'plan-2' }), 'rejected');
  assert.equal(classifyAnswer(sent, answer, { ...current, planRevision: 4 }), 'rejected');
  assert.equal(classifyAnswer(sent, answer, { ...current, sessionId: 'sess-2' }), 'rejected');
});

test('an older intent_seq is rejected even when everything else matches', () => {
  assert.equal(classifyAnswer(sent, answer, { ...current, latestIntentSeq: 8 }), 'rejected');
});

test('an echo that is not this request\'s own is rejected', () => {
  assert.equal(classifyAnswer(sent, { ...answer, intent_seq: 6 }, current), 'rejected');
  assert.equal(classifyAnswer(sent, { ...answer, fence_echo: { ...FENCE, run_id: 'run-x' } }, current), 'rejected');
  assert.equal(classifyAnswer(sent, { ...answer, anchors_echo: [] }, current), 'rejected');
  assert.equal(classifyAnswer(sent, { ...answer, plan_revision: 2 }, current), 'rejected');
});

test('a confirm that applied a replan answers this plan only with a higher revision', () => {
  assert.equal(classifyAnswer(sent, { ...answer, plan_revision: 4 }, current, { replanApplied: true }), 'bound');
  assert.equal(classifyAnswer(sent, { ...answer, plan_revision: 3 }, current, { replanApplied: true }), 'rejected');
  assert.equal(classifyAnswer(sent, { ...answer, plan_revision: 4 }, current), 'rejected');
});

test('a request sent without anchors is bound only while no anchor exists', () => {
  const bare = { ...sent, anchors: [] };
  const bareAnswer = { ...answer, anchors_echo: [] };
  assert.equal(classifyAnswer(bare, bareAnswer, { ...current, anchor: null }), 'bound');
  assert.equal(classifyAnswer(bare, bareAnswer, current), 'text_only');
});

test('the text-only qualifier states the capture age in whole seconds', () => {
  assert.equal(captureAgeSeconds(10_000, 13_400), 3);
  assert.equal(captureAgeSeconds(10_000, 9_000), 0);
  assert.equal(captureAgeSeconds(Number.NaN, 1), 0);
  assert.equal(textOnlyQualifier(10_000, 12_000), '2초 전 화면 기준');
});
