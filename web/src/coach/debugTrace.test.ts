/**
 * Behaviour of the guide-lane trace behind the debug panel.
 *
 * Loaded through Vite, so the module under test is the one the app ships. The assertions are about the
 * accounting, never about the Korean wording the panel prints.
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

const {
  EMPTY_GUIDE_TRACE,
  GUIDE_CALL_LIMIT,
  appendGuideCall,
  bumpGuideCounter,
  countGuideCalls,
  guideHideRatio,
  markGuideMilestone,
  sampleGuideHide,
  startGuideTrace,
} = await server.ssrLoadModule('/src/coach/debugTrace.ts');

test('guide milestones are recorded once (first wins) and rounded', () => {
  let trace = startGuideTrace(1_000);
  assert.equal(trace.startedAtWall, 1_000);
  assert.equal(trace.msStartToFirstSay, null);
  trace = markGuideMilestone(trace, 'msStartToFirstSay', 1538.4);
  trace = markGuideMilestone(trace, 'msStartToFirstSay', 2600);
  trace = markGuideMilestone(trace, 'msStartToFinalPlan', 2523.6);
  trace = markGuideMilestone(trace, 'msStartToFirstBox', -5);
  assert.equal(trace.msStartToFirstSay, 1538);
  assert.equal(trace.msStartToFinalPlan, 2524);
  assert.equal(trace.msStartToFirstBox, 0);
  assert.equal(markGuideMilestone(trace, 'msStartToFirstBox', Number.NaN), trace);
});

test('guide calls are bounded newest first and counted by acceptance', () => {
  let trace = EMPTY_GUIDE_TRACE;
  const kinds = ['bound', 'text_only', 'rejected', 'failed', 'bound'];
  for (let i = 0; i < GUIDE_CALL_LIMIT + 3; i += 1) {
    trace = appendGuideCall(trace, {
      id: `c${i}`,
      stage: 'follow',
      trigger: 'heartbeat',
      provider: 'local',
      msToFinal: 300 + i,
      accepted: kinds[i % kinds.length],
      errorCode: null,
      startedAtWall: i,
    });
  }
  assert.equal(trace.calls.length, GUIDE_CALL_LIMIT);
  assert.equal(trace.calls[0].id, `c${GUIDE_CALL_LIMIT + 2}`);
  const counts = countGuideCalls(trace);
  assert.equal(counts.bound + counts.text_only + counts.rejected + counts.failed, GUIDE_CALL_LIMIT);
});

test('guide counters and the sampled 300 ms hide ratio', () => {
  let trace = EMPTY_GUIDE_TRACE;
  assert.equal(guideHideRatio(trace), null);
  for (const hidden of [false, false, true, false]) trace = sampleGuideHide(trace, hidden);
  assert.equal(guideHideRatio(trace), 0.25);
  trace = bumpGuideCounter(bumpGuideCounter(trace, 'followDone'), 'followDone');
  trace = bumpGuideCounter(trace, 'confirmReversals');
  trace = bumpGuideCounter(trace, 'targetMoved');
  trace = bumpGuideCounter(trace, 'followUnsure');
  assert.deepEqual(
    [trace.followDone, trace.followUnsure, trace.confirmReversals, trace.targetMoved],
    [2, 1, 1, 1]
  );
  assert.equal(EMPTY_GUIDE_TRACE.followDone, 0, 'helpers never mutate');
});

test('target_changed and budget waits are counted separately', () => {
  let trace = startGuideTrace(0);
  assert.deepEqual([trace.targetChanged, trace.budgetCapped], [0, 0]);
  trace = bumpGuideCounter(trace, 'targetChanged');
  trace = bumpGuideCounter(trace, 'budgetCapped');
  assert.deepEqual([trace.targetChanged, trace.budgetCapped, trace.targetMoved], [1, 1, 0]);
  assert.equal('followOtherStep' in trace, false, 'the checklist follow has no other-step answers');
});

test('skips, pending skips, stuck replans and replan-limit hits are counted apart', () => {
  let trace = startGuideTrace(0);
  assert.deepEqual([trace.followSkip, trace.followPendingSkip, trace.stuckReplan, trace.replanLimited], [0, 0, 0, 0]);
  trace = bumpGuideCounter(bumpGuideCounter(trace, 'followPendingSkip'), 'followPendingSkip');
  trace = bumpGuideCounter(trace, 'followSkip');
  trace = bumpGuideCounter(trace, 'stuckReplan');
  trace = bumpGuideCounter(bumpGuideCounter(bumpGuideCounter(trace, 'replanLimited'), 'replanLimited'), 'replanLimited');
  assert.deepEqual([trace.followSkip, trace.followPendingSkip, trace.stuckReplan, trace.replanLimited], [1, 2, 1, 3]);
  assert.equal(EMPTY_GUIDE_TRACE.replanLimited, 0, 'helpers never mutate');
});

test('confirms on the follow frame and dropped follow-frame confirms are counted apart from reversals', () => {
  let trace = startGuideTrace(0);
  assert.deepEqual([trace.confirmFollowFrame, trace.confirmFollowFrameDropped], [0, 0]);
  trace = bumpGuideCounter(bumpGuideCounter(trace, 'confirmFollowFrame'), 'confirmFollowFrame');
  trace = bumpGuideCounter(trace, 'confirmFollowFrameDropped');
  assert.deepEqual([trace.confirmFollowFrame, trace.confirmFollowFrameDropped, trace.confirmReversals], [2, 1, 0]);
});
