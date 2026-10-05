/**
 * Behaviour of the page's paid-call cost accounting (startup selection and guide lane).
 *
 * Run with `npm run test:web` (node's built-in test runner, which strips the type annotations). Kept out
 * of the app tsconfig so the browser build never sees `node:` imports.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  EMPTY_USAGE_TOTALS,
  addProviderUsage,
  addUsage,
  EMPTY_GUIDE_USAGE,
  GUIDE_LOCAL_COST_TEXT,
  GUIDE_STAGES,
  addGuideCall,
  addGuideUnknownAttempt,
  describeGuideStageCost,
  guideStageCostUsd,
  peakCostUsd,
} from './liveCost.ts';

const SIX_DECIMALS = 1e-6;

test('a complete usage report is priced at peak rates and flagged complete', () => {
  const totals = addUsage(EMPTY_USAGE_TOTALS, { input_tokens: 3600, cached_input_tokens: 2400, output_tokens: 320 });
  assert.equal(totals.unknownAttempts, 0);
  assert.equal(totals.input, 3600);
  assert.equal(totals.cachedInput, 2400);
  assert.equal(totals.output, 320);
  // 2400 cached * $0.006 + 1200 miss * $0.30 + 320 out * $1.20, per million tokens.
  assert.ok(Math.abs(peakCostUsd(totals) - 0.0007584) < SIX_DECIMALS, String(peakCostUsd(totals)));
});

test('usage missing the output half still counts the input, and is not presented as a complete cost', () => {
  const totals = addUsage(EMPTY_USAGE_TOTALS, { input_tokens: 3600, cached_input_tokens: 2400 });
  assert.equal(totals.input, 3600);
  assert.equal(totals.output, 0);
  // The attempt is flagged, so "every priced attempt is accounted for" cannot be read off the total.
  assert.equal(totals.unknownAttempts, 1);
  // And the tokens that were reported are still real spend, not dropped.
  assert.ok(Math.abs(peakCostUsd(totals) - 0.0003744) < SIX_DECIMALS, String(peakCostUsd(totals)));
});

test('usage missing the input half still counts the output, and is flagged the same way', () => {
  const totals = addUsage(EMPTY_USAGE_TOTALS, { output_tokens: 320 });
  assert.equal(totals.input, 0);
  assert.equal(totals.output, 320);
  assert.equal(totals.unknownAttempts, 1);
  assert.ok(Math.abs(peakCostUsd(totals) - 0.000384) < SIX_DECIMALS, String(peakCostUsd(totals)));
});

test('no usage at all counts as unknown rather than as a free call', () => {
  for (const missing of [null, undefined, {}]) {
    const totals = addUsage(EMPTY_USAGE_TOTALS, missing);
    assert.equal(totals.unknownAttempts, 1);
    assert.equal(peakCostUsd(totals), 0);
  }
});

test('a missing cache-hit split is priced as misses, not as a discount', () => {
  const totals = addUsage(EMPTY_USAGE_TOTALS, { input_tokens: 3000, output_tokens: 300 });
  assert.equal(totals.unknownAttempts, 0);
  // 3000 * $0.30 (all misses) + 300 * $1.20.
  assert.ok(Math.abs(peakCostUsd(totals) - 0.00126) < SIX_DECIMALS, String(peakCostUsd(totals)));
});

test('totals accumulate across attempts without losing the incomplete ones', () => {
  let totals = EMPTY_USAGE_TOTALS;
  totals = addUsage(totals, { input_tokens: 3600, cached_input_tokens: 2400, output_tokens: 320 });
  totals = addUsage(totals, { input_tokens: 3600, cached_input_tokens: 2400 });
  totals = addUsage(totals, { output_tokens: 320 });
  assert.equal(totals.input, 7200);
  assert.equal(totals.output, 640);
  assert.equal(totals.unknownAttempts, 2);
});

test('a startup analysis by the priced provider is priced at the peak rates', () => {
  const totals = addProviderUsage(EMPTY_USAGE_TOTALS, 'deepseek', 'deepseek-flash', {
    input_tokens: 3600,
    cached_input_tokens: 2400,
    output_tokens: 320,
  });
  assert.equal(totals.input, 3600);
  assert.equal(totals.cachedInput, 2400);
  assert.equal(totals.output, 320);
  assert.equal(totals.unpricedStages, 0);
  assert.equal(totals.unknownAttempts, 0);
  assert.ok(Math.abs(peakCostUsd(totals) - 0.0007584) < SIX_DECIMALS, String(peakCostUsd(totals)));
});

test('a startup analysis by another provider is unpriced, never folded into the priced bucket', () => {
  const totals = addProviderUsage(EMPTY_USAGE_TOTALS, 'cerebras', 'qwen-3.8-27b', {
    input_tokens: 5190,
    output_tokens: 800,
  });
  assert.equal(peakCostUsd(totals), 0);
  assert.equal(totals.input, 0);
  assert.equal(totals.unpricedStages, 1);
  assert.equal(totals.unpricedInput, 5190);
  assert.equal(totals.unpricedOutput, 800);
  assert.equal(totals.unknownAttempts, 0);
});

test('the priced provider with the wrong model is unpriced, not priced by provider alone', () => {
  const totals = addProviderUsage(EMPTY_USAGE_TOTALS, 'deepseek', 'deepseek-other', {
    input_tokens: 100,
    output_tokens: 10,
  });
  assert.equal(peakCostUsd(totals), 0);
  assert.equal(totals.unpricedStages, 1);
});

test('a startup analysis that reported no usage is unknown spend, never free', () => {
  for (const missing of [null, undefined, {}]) {
    const totals = addProviderUsage(EMPTY_USAGE_TOTALS, 'deepseek', 'deepseek-flash', missing);
    assert.equal(totals.unknownAttempts, 1);
    assert.equal(peakCostUsd(totals), 0);
  }
  // An unpriced provider that reported nothing is also an unknown attempt.
  const unpriced = addProviderUsage(EMPTY_USAGE_TOTALS, 'cerebras', 'qwen-3.8-27b', null);
  assert.equal(unpriced.unpricedStages, 1);
  assert.equal(unpriced.unknownAttempts, 1);
});

test('guide stages are priced per provider: DeepSeek at peak rates, the local follower with no API charge', () => {
  let totals = addGuideCall(EMPTY_GUIDE_USAGE, 'plan', 'deepseek', 'deepseek-flash', {
    input_tokens: 1879,
    cached_input_tokens: 0,
    output_tokens: 440,
  });
  totals = addGuideCall(totals, 'follow', 'local', 'Qwen3.5-0.8B-Q4_K_M', { input_tokens: 938, output_tokens: 85 });
  totals = addGuideCall(totals, 'follow', 'local', 'Qwen3.5-0.8B-Q4_K_M', null);
  // 1879 miss * $0.30 + 440 out * $1.20 per million.
  assert.ok(Math.abs(guideStageCostUsd(totals, 'plan') - 0.0010917) < SIX_DECIMALS, String(guideStageCostUsd(totals, 'plan')));
  assert.equal(guideStageCostUsd(totals, 'follow'), 0);
  assert.equal(totals.follow.localStages, 2);
  assert.equal(totals.follow.localInput, 938);
  assert.equal(totals.follow.unknownAttempts, 0, 'a local call without usage is not unknown spend');
  assert.equal(totals.confirm, EMPTY_GUIDE_USAGE.confirm, 'other stages are untouched');
  assert.equal(describeGuideStageCost(totals, 'follow'), `로컬 2회 · ${GUIDE_LOCAL_COST_TEXT}`);
  assert.equal(GUIDE_LOCAL_COST_TEXT, 'API 요금 없음');
  assert.match(describeGuideStageCost(totals, 'plan'), /^\$0\.001092\(피크 단가 추정\)$/);
  assert.equal(describeGuideStageCost(totals, 'confirm'), '없음');
});

test('the Clef follower is local compute too, not unpriced provider spend', () => {
  const totals = addGuideCall(EMPTY_GUIDE_USAGE, 'follow', 'clef', 'clef', { input_tokens: 812, output_tokens: 0 });
  assert.equal(totals.follow.localStages, 1);
  assert.equal(totals.follow.localInput, 812);
  assert.equal(guideStageCostUsd(totals, 'follow'), 0);
  assert.equal(describeGuideStageCost(totals, 'follow'), `로컬 1회 · ${GUIDE_LOCAL_COST_TEXT}`);
});

test('a paid guide call without usage is unknown spend; a local one is not counted', () => {
  let totals = addGuideUnknownAttempt(EMPTY_GUIDE_USAGE, 'confirm', 'remote');
  totals = addGuideUnknownAttempt(totals, 'follow', 'local');
  assert.equal(totals.confirm.unknownAttempts, 1);
  assert.equal(totals.follow, EMPTY_GUIDE_USAGE.follow);
  assert.match(describeGuideStageCost(totals, 'confirm'), /사용량 미수신 1회/);
});

test('a guide call from an unpriced model is reported but never priced as DeepSeek', () => {
  const totals = addGuideCall(EMPTY_GUIDE_USAGE, 'follow', 'deepseek', 'deepseek-other', { input_tokens: 100, output_tokens: 10 });
  assert.equal(guideStageCostUsd(totals, 'follow'), 0);
  assert.equal(totals.follow.unpricedStages, 1);
});

test('talk is its own guide stage, priced like the other DeepSeek calls', () => {
  assert.deepEqual(GUIDE_STAGES, ['plan', 'follow', 'confirm', 'talk']);
  const totals = addGuideCall(EMPTY_GUIDE_USAGE, 'talk', 'deepseek', 'deepseek-flash', { input_tokens: 1000, output_tokens: 100 });
  assert.ok(guideStageCostUsd(totals, 'talk') > 0);
  assert.equal(totals.confirm, EMPTY_GUIDE_USAGE.confirm);
  assert.match(describeGuideStageCost(addGuideUnknownAttempt(EMPTY_GUIDE_USAGE, 'talk', 'remote'), 'talk'), /사용량 미수신 1회/);
});
