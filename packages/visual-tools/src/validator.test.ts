import test from 'node:test';
import assert from 'node:assert/strict';
import {
  validateContract,
  validateCommands,
  validateCommandsSchema,
  validateAdviceRelational,
} from './validator.js';
import type { GuidanceAdvice, VisualCommand } from './index.js';

const focus: VisualCommand = { id: 'f1', kind: 'focus', box: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 }, label: '대상' };
const arrow: VisualCommand = { id: 'a1', kind: 'arrow', from: { x: 0.1, y: 0.1 }, to: { x: 0.3, y: 0.3 }, label: '방향' };
const gesture: VisualCommand = { id: 'g1', kind: 'gesture', points: [{ x: 0.1, y: 0.1 }, { x: 0.4, y: 0.4 }], duration_ms: 2000, label: '동작' };
const hint: VisualCommand = { id: 'h1', kind: 'hint', text: '표시 방향을 맞추세요', reference: 'line' };

function advice(overrides: Partial<GuidanceAdvice> = {}): GuidanceAdvice {
  return {
    explanation: '현재 상태를 확인했습니다.',
    observation: '왼쪽에 빈 슬롯이 보입니다.',
    steps: [],
    evidence_kind: 'observed_scene',
    references: [],
    commands: [],
    needs_clarification: false,
    clarification_prompt: null,
    ready_to_advance: false,
    warnings: [],
    ...overrides,
  };
}

// NOTE: diagnostic wording (which kind is named in the reason string) is intentionally NOT pinned
// here. It is a human-facing string that changes freely; the observable contract is
// valid/invalid plus the atomic render outcome, which the browser harness covers for real.
test('visual contract validation', async (t) => {
  await t.test('accepts a well formed advice payload', () => {
    assert.equal(validateContract('GuidanceAdvice', advice({ commands: [focus, arrow] })).valid, true);
  });

  await t.test('reports an unknown contract name instead of throwing', () => {
    assert.equal(validateContract('NotAContract' as never, {}).valid, false);
  });

  await t.test('rejects non-finite coordinates without JSON serialization', () => {
    assert.equal(validateCommands([{ ...focus, box: { x: Number.NaN, y: 0.1, width: 0.2, height: 0.2 } }]).valid, false);
    assert.equal(validateCommands([{ ...focus, box: { x: 0.1, y: Number.POSITIVE_INFINITY, width: 0.2, height: 0.2 } }]).valid, false);
  });

  await t.test('rejects a zero-width focus box', () => {
    assert.equal(validateCommands([{ ...focus, box: { x: 0.1, y: 0.1, width: 0, height: 0.2 } }]).valid, false);
  });

  await t.test('rejects a box escaping the unit square', () => {
    assert.equal(validateAdviceRelational(advice({ commands: [{ ...focus, box: { x: 0.9, y: 0.1, width: 0.2, height: 0.2 } }] })).valid, false);
  });
});

test('rejected payloads are reported invalid', async (t) => {
  await t.test('a bad gesture duration is refused', () => {
    assert.equal(validateCommands([{ ...gesture, duration_ms: 200 }]).valid, false);
  });

  await t.test('more than three commands are refused', () => {
    assert.equal(validateCommands([focus, arrow, gesture, hint]).valid, false);
  });

  await t.test('duplicate command ids are refused', () => {
    assert.equal(validateCommands([focus, { ...arrow, id: 'f1' }]).valid, false);
  });

  await t.test('an unknown kind is refused', () => {
    assert.equal(validateCommands([{ id: 'x1', kind: 'spotlight', label: 'x' }]).valid, false);
  });

  await t.test('a non-array payload is refused', () => {
    assert.equal(validateCommands('not-commands').valid, false);
  });

  await t.test('a valid bundle passes both the schema and the curated path', () => {
    assert.equal(validateCommandsSchema([focus, hint]), true);
    assert.equal(validateCommands([focus, hint]).valid, true);
  });

  await t.test('the low-level path also enforces the advice-level command rules', () => {
    // A focus box that escapes the unit square is refused by the low-level gate too, so a bare
    // renderCommands caller cannot disagree with the framed entrypoint.
    assert.equal(validateCommands([{ ...focus, box: { x: 0.8, y: 0.2, width: 0.4, height: 0.2 } }]).valid, false);
    assert.equal(validateCommands([gesture, { ...gesture, id: 'g2' }]).valid, false);
    assert.equal(validateCommands([{ ...focus, box: { x: 0.1, y: 0.8, width: 0.2, height: 0.4 } }]).valid, false);
    assert.equal(validateCommands([gesture]).valid, true);
  });

  await t.test('a hostile kind never resolves through the prototype', () => {
    for (const kind of ['__proto__', 'constructor', 'toString', 'hasOwnProperty']) {
      assert.equal(validateCommands([{ id: 'x1', kind }]).valid, false, `kind ${kind} must be rejected`);
    }
    assert.equal(validateCommands([{ id: 'x1' }]).valid, false);
    assert.equal(validateCommands([{ id: 'x1', kind: 42 }]).valid, false);
  });

  await t.test('a whitespace-only clarification prompt is treated like Python', () => {
    assert.equal(
      validateAdviceRelational(advice({ needs_clarification: true, clarification_prompt: '   ' })).valid,
      false
    );
    assert.equal(
      validateAdviceRelational(advice({ clarification_prompt: '   ' })).valid,
      false
    );
  });
});

test('advice relational rules', async (t) => {
  await t.test('clarification cannot carry spatial commands', () => {
    assert.equal(
      validateAdviceRelational(advice({ needs_clarification: true, clarification_prompt: '각도를 바꿔주세요', commands: [focus] })).valid,
      false
    );
  });

  await t.test('clarification alone with a hint is allowed', () => {
    assert.equal(
      validateAdviceRelational(advice({ needs_clarification: true, clarification_prompt: '각도를 바꿔주세요', commands: [hint] })).valid,
      true
    );
  });

  await t.test('uncertain_view requires clarification', () => {
    assert.equal(validateAdviceRelational(advice({ evidence_kind: 'uncertain_view' })).valid, false);
  });

  await t.test('reference evidence requires a cited reference', () => {
    assert.equal(validateAdviceRelational(advice({ evidence_kind: 'reference' })).valid, false);
    assert.equal(validateAdviceRelational(advice({ evidence_kind: 'reference', references: ['manual-3'] })).valid, true);
  });

  await t.test('at most one gesture command is allowed', () => {
    assert.equal(validateAdviceRelational(advice({ commands: [gesture, { ...gesture, id: 'g2' }] })).valid, false);
  });

  await t.test('text guidance cannot carry visual commands', () => {
    assert.equal(validateAdviceRelational(advice({ commands: [focus] }), { guidanceMode: 'text' }).valid, false);
    assert.equal(validateAdviceRelational(advice({ commands: [focus] }), { guidanceMode: 'visual' }).valid, true);
  });
});
