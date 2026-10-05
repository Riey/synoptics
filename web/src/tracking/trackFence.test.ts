import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  decideError,
  decideUpdate,
  isBoxDrawable,
  isTerminalTrackState,
  runFenceKey,
  type FenceState,
} from './trackFence.ts';

const base: FenceState = { activeRunId: 'run-1', pairedFrameId: 'f-9', pairedFrameSeq: 4, lastVersion: 7 };

test('decideUpdate accepts the paired frame of the active run with a newer version', () => {
  const decision = decideUpdate({ runId: 'run-1', frameId: 'f-9', frameSeq: 4, version: 8 }, base);
  assert.deepEqual(decision, { accept: true, reason: 'accepted' });
});

test('decideUpdate rejects everything when no run is active', () => {
  const decision = decideUpdate(
    { runId: 'run-1', frameId: 'f-9', frameSeq: 4, version: 8 },
    { ...base, activeRunId: null }
  );
  assert.deepEqual(decision, { accept: false, reason: 'no_run' });
});

test('decideUpdate drops a stopped run\u2019s late response regardless of version', () => {
  const decision = decideUpdate({ runId: 'run-0', frameId: 'f-9', frameSeq: 4, version: 999 }, base);
  assert.deepEqual(decision, { accept: false, reason: 'stale_run' });
});

test('decideUpdate drops an answer for a frame we are not pairing', () => {
  const otherFrame = decideUpdate({ runId: 'run-1', frameId: 'f-10', frameSeq: 5, version: 8 }, base);
  assert.deepEqual(otherFrame, { accept: false, reason: 'frame_mismatch' });
  const otherSeq = decideUpdate({ runId: 'run-1', frameId: 'f-9', frameSeq: 5, version: 8 }, base);
  assert.deepEqual(otherSeq, { accept: false, reason: 'frame_mismatch' });
});

test('decideUpdate drops a duplicate or older version within the run', () => {
  assert.deepEqual(decideUpdate({ runId: 'run-1', frameId: 'f-9', frameSeq: 4, version: 7 }, base), {
    accept: false,
    reason: 'stale_version',
  });
  assert.deepEqual(decideUpdate({ runId: 'run-1', frameId: 'f-9', frameSeq: 4, version: 6 }, base), {
    accept: false,
    reason: 'stale_version',
  });
});

test('isBoxDrawable follows (box != null) == (state == tracking)', () => {
  assert.equal(isBoxDrawable({ x: 0, y: 0, width: 1, height: 1 }, 'tracking'), true);
  assert.equal(isBoxDrawable(null, 'tracking'), false);
  assert.equal(isBoxDrawable({ x: 0, y: 0, width: 1, height: 1 }, 'acquiring'), false);
  assert.equal(isBoxDrawable({ x: 0, y: 0, width: 1, height: 1 }, 'occluded'), false);
  assert.equal(isBoxDrawable(null, 'lost'), false);
  assert.equal(isBoxDrawable(undefined, 'unavailable'), false);
});

test('lost and unavailable end the run; acquiring and occluded keep it alive', () => {
  assert.equal(isTerminalTrackState('lost'), true);
  assert.equal(isTerminalTrackState('unavailable'), true);
  assert.equal(isTerminalTrackState('acquiring'), false);
  assert.equal(isTerminalTrackState('occluded'), false);
  assert.equal(isTerminalTrackState('tracking'), false);
});

test('the v4 error taxonomy maps onto run/frame/panel actions', () => {
  assert.equal(decideError(409, 'stale_run'), 'retire_run');
  assert.equal(decideError(409, 'not_active_run'), 'retire_run');
  assert.equal(decideError(409, 'stale_frame'), 'drop_frame');
  assert.equal(decideError(409, 'stale_start'), 'lost_start_race');
  assert.equal(decideError(409, 'seed_not_first_frame'), 'retire_run');
  assert.equal(decideError(503, 'tracker_unavailable'), 'unavailable');
  assert.equal(decideError(504, 'timeout'), 'unavailable');
  assert.equal(decideError(500, 'boom'), 'unknown');
});

test('the run fence key changes with source, geometry, mirror, mode and task', () => {
  const key = runFenceKey({ sourceEpoch: 1, cameraFrameSize: '640x480', mirrorView: false, sceneMode: 'camera', taskKey: 'a' });
  const same = runFenceKey({ sourceEpoch: 1, cameraFrameSize: '640x480', mirrorView: false, sceneMode: 'camera', taskKey: 'a' });
  assert.equal(key, same);
  for (const change of [
    { sourceEpoch: 2, cameraFrameSize: '640x480', mirrorView: false, sceneMode: 'camera', taskKey: 'a' },
    { sourceEpoch: 1, cameraFrameSize: '1280x720', mirrorView: false, sceneMode: 'camera', taskKey: 'a' },
    { sourceEpoch: 1, cameraFrameSize: '640x480', mirrorView: true, sceneMode: 'camera', taskKey: 'a' },
    { sourceEpoch: 1, cameraFrameSize: '640x480', mirrorView: false, sceneMode: 'photo', taskKey: 'a' },
    { sourceEpoch: 1, cameraFrameSize: '640x480', mirrorView: false, sceneMode: 'camera', taskKey: 'b' },
  ]) {
    assert.notEqual(runFenceKey(change), key);
  }
});
