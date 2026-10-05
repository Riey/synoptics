/**
 * Guide-call tap for bug reports: images come out of the recorded request, the call's result passes
 * through untouched, and nothing is reported while no tap is set.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { imageFileTag, setGuideTap, stripImages, tapGuideCall, type GuideCallEvent } from './guideTap.ts';

test('image fields are replaced by references and collected', () => {
  const { request, images } = stripImages({
    trigger: 'target_left',
    scene: { frame_id: 'f2', image_base64: 'data:image/jpeg;base64,QUJD' },
    before_scene: { frame_id: 'f1', image_base64: 'REVG' },
  });
  assert.deepEqual(request, {
    trigger: 'target_left',
    scene: { frame_id: 'f2', image_base64: '<image:scene.image_base64>' },
    before_scene: { frame_id: 'f1', image_base64: '<image:before_scene.image_base64>' },
  });
  assert.deepEqual(images, [
    { field: 'scene.image_base64', base64: 'QUJD', mime: 'image/jpeg' },
    { field: 'before_scene.image_base64', base64: 'REVG', mime: 'image/jpeg' },
  ]);
  assert.equal(imageFileTag('before_scene.image_base64'), 'before_scene');
  assert.equal(imageFileTag('frames[0].image_base64'), 'frames_0');
});

test('the tap sees answers and errors; the caller gets the same result', async () => {
  const seen: GuideCallEvent[] = [];
  setGuideTap((event) => seen.push(event));
  try {
    assert.deepEqual(await tapGuideCall('/api/guide/follow', { a: 1 }, async () => ({ step_status: 'done' })), { step_status: 'done' });
    await assert.rejects(tapGuideCall('/api/guide/confirm', { b: 2 }, async () => { throw new Error('409 stale_plan'); }), /stale_plan/);
  } finally {
    setGuideTap(null);
  }
  assert.deepEqual(seen.map((event) => [event.path, event.response ?? null, event.error ?? null]), [
    ['/api/guide/follow', { step_status: 'done' }, null],
    ['/api/guide/confirm', null, 'Error: 409 stale_plan'],
  ]);
  assert.equal(await tapGuideCall('/api/guide/talk', {}, async () => 'untapped'), 'untapped');
  assert.equal(seen.length, 2);
});
