/**
 * Session recording for bug reports: every closed segment is handed out as a playable file, recording
 * continues across a flush, and a recorder that cannot start is counted, not fatal.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { SessionRecorder, pickRecorderMime, type RecordedSegment, type RecorderLike } from './sessionRecorder.ts';

class FakeRecorder implements RecorderLike {
  state = 'inactive';
  mimeType = 'video/webm;codecs=vp8';
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: (() => void) | null = null;
  start(): void {
    this.state = 'recording';
  }
  emit(bytes: number): void {
    this.ondataavailable?.({ data: new Blob([new Uint8Array(bytes)]) });
  }
  stop(): void {
    this.state = 'inactive';
    this.emit(1);
    queueMicrotask(() => this.onstop?.());
  }
}

function setup(maxBytes = 1_000_000) {
  const made: FakeRecorder[] = [];
  const closed: RecordedSegment[] = [];
  let clock = 1000;
  const recorder = new SessionRecorder({
    create: () => {
      const fake = new FakeRecorder();
      made.push(fake);
      return fake;
    },
    mime: 'video/webm',
    segmentMs: 60_000,
    maxBytes,
    now: () => clock,
    stopTimeoutMs: 50,
  });
  recorder.onSegmentClosed = (segment) => closed.push(segment);
  return { recorder, made, closed, tick: (ms: number) => (clock += ms) };
}

const stream = {} as MediaStream;

test('flush hands out the running segment and keeps recording in a new one', async () => {
  const { recorder, made, closed, tick } = setup();
  recorder.attach(stream);
  made[0].emit(100);
  tick(5000);
  await recorder.flush();
  assert.equal(closed.length, 1);
  assert.equal(closed[0].bytes, 101);
  assert.equal(closed[0].endedAt - closed[0].startedAt, 5000);
  assert.equal(closed[0].blob.size, 101);
  assert.equal(made[1].state, 'recording');
  assert.equal(recorder.stats().recording, true);
  await recorder.detach();
});

test('a replaced stream closes the old segment; detach closes the last one', async () => {
  const { recorder, made, closed } = setup();
  recorder.attach(stream);
  made[0].emit(10);
  recorder.attach({} as MediaStream);
  made[1].emit(20);
  await recorder.detach();
  assert.equal(recorder.stats().recording, false);
  assert.deepEqual(closed.map((segment) => [segment.index, segment.bytes]), [[0, 11], [1, 21]]);
});

test('a segment past maxBytes is closed early', async () => {
  const { recorder, made, closed } = setup(50);
  recorder.attach(stream);
  made[0].emit(60);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(closed.length, 1);
  assert.equal(made.length, 2);
  await recorder.detach();
});

test('a recorder that throws is counted, not fatal', () => {
  const recorder = new SessionRecorder({
    create: () => {
      throw new Error('NotSupportedError');
    },
    mime: 'video/webm',
    segmentMs: 1000,
    maxBytes: 1000,
  });
  recorder.attach(stream);
  assert.deepEqual(recorder.stats(), { recording: false, bytes: 0, segments: 0, failures: 1 });
});

test('mime choice prefers webm', () => {
  assert.equal(pickRecorderMime((mime) => mime.startsWith('video/mp4')), 'video/mp4;codecs=avc1');
  assert.equal(pickRecorderMime(() => true), 'video/webm;codecs=vp9');
  assert.equal(pickRecorderMime(() => false), null);
});
