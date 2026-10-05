/**
 * Plan model choice: the closed set is named for the UI, and a model's readiness is read strictly from
 * `plan_models` — a missing or non-boolean entry never counts as availability, so a key-less model can
 * never enable a run (and never silently becomes the other model).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  planModelBlockedReason,
  planModelState,
  readPlanModelAvailability,
} from './modelChoice.ts';

test('availability is only what the server sent as true: a missing or non-boolean entry is unavailable', () => {
  const reported = readPlanModelAvailability({ 'deepseek:high': true, 'astra:high': false });
  assert.equal(planModelState(reported, 'deepseek:high'), 'ready');
  assert.equal(planModelState(reported, 'astra:high'), 'unavailable');

  const partial = readPlanModelAvailability({ 'deepseek:high': true });
  assert.equal(planModelState(partial, 'deepseek:high'), 'ready');
  assert.equal(planModelState(partial, 'astra:high'), 'unavailable');

  const truthy = readPlanModelAvailability({ 'astra:high': 1 });
  assert.equal(planModelState(truthy, 'astra:high'), 'unavailable');
});

test('an absent plan_models payload is unknown, never a silent enable', () => {
  for (const raw of [null, undefined, 'plan_models', 42]) {
    const availability = readPlanModelAvailability(raw);
    assert.equal(planModelState(availability, 'deepseek:high'), 'unknown');
    assert.equal(planModelState(availability, 'astra:high'), 'unknown');
  }
});

test('the chosen model runs only when it is ready; an unavailable model is not replaced', () => {
  const both = readPlanModelAvailability({ 'deepseek:high': true, 'astra:high': true });
  assert.equal(planModelBlockedReason(both, 'deepseek:high'), null);
  assert.equal(planModelBlockedReason(both, 'astra:high'), null);

  const deepseekOnly = readPlanModelAvailability({ 'deepseek:high': true, 'astra:high': false });
  assert.equal(planModelBlockedReason(deepseekOnly, 'deepseek:high'), null);
  assert.notEqual(planModelBlockedReason(deepseekOnly, 'astra:high'), null);

  assert.notEqual(planModelBlockedReason(null, 'deepseek:high'), null);
});
