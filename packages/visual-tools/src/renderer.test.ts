import test from 'node:test';
import assert from 'node:assert/strict';
import { renderGuidance, type FramedRenderContext } from './renderer.js';
import type { GuidanceAdvice } from './index.js';

/** A host that has no usable DOM: any attempt to touch it throws, which is exactly what we assert. */
function unusableHost(currentFrameId: string): FramedRenderContext {
  return { currentFrameId } as unknown as FramedRenderContext;
}

function advice(overrides: Partial<GuidanceAdvice> = {}): GuidanceAdvice {
  return {
    explanation: '설명',
    observation: '관찰',
    steps: [],
    evidence_kind: 'observed_scene',
    references: [],
    commands: [{ id: 'f1', kind: 'focus', box: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 }, label: '대상' }],
    needs_clarification: false,
    clarification_prompt: null,
    ready_to_advance: false,
    warnings: [],
    ...overrides,
  };
}

test('framed guidance entrypoint', async (t) => {
  await t.test('a malformed payload is rejected without a guidance id', async () => {
    const report = await renderGuidance({ guidance_id: 'g1', advice: { nonsense: true } }, unusableHost('frame-1'));
    assert.equal(report.status, 'rejected');
    assert.equal(report.guidance_id, null);
    assert.deepEqual(report.command_ids, []);
    assert.ok(report.reason);
  });

  await t.test('a frame mismatch is stale before any DOM work', async () => {
    const report = await renderGuidance(
      { guidance_id: 'g1', frame_id: 'frame-old', advice: advice() },
      unusableHost('frame-new')
    );
    assert.equal(report.status, 'stale');
    assert.equal(report.guidance_id, 'g1');
    assert.equal(report.frame_id, 'frame-old');
    assert.deepEqual(report.command_ids, []);
  });

  await t.test('an ungrounded reference is rejected before any DOM work', async () => {
    const report = await renderGuidance(
      { guidance_id: 'g2', frame_id: 'frame-1', advice: advice({ evidence_kind: 'reference', references: [] }) },
      { ...unusableHost('frame-1'), referenceIds: ['manual-1'] }
    );
    assert.equal(report.status, 'rejected');
    assert.equal(report.guidance_id, 'g2');
  });

  await t.test('a valid payload against an unusable host reports a rejected render', async () => {
    const report = await renderGuidance(
      { guidance_id: 'g3', frame_id: 'frame-1', advice: advice() },
      unusableHost('frame-1')
    );
    assert.equal(report.status, 'rejected');
    assert.equal(report.guidance_id, 'g3');
    assert.ok(report.reason);
  });
});
