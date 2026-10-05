/**
 * Streaming bug-report uploads: one report per run, segments in order with a meta refresh after each,
 * retry after a failure, bounded pending memory, and a fresh report after finish.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { AutoReporter, extensionForMime, type ReportApi } from './autoReporter.ts';
import type { RecordedSegment } from './sessionRecorder.ts';

function segment(index: number, bytes = 10, mime = 'video/webm'): RecordedSegment {
  return { index, startedAt: index * 1000, endedAt: index * 1000 + 900, mime, bytes, blob: new Blob([new Uint8Array(bytes)]) };
}

function fakeApi() {
  const calls: string[] = [];
  const metas: Record<string, unknown>[] = [];
  let reports = 0;
  let failNextPut = false;
  const api: ReportApi = {
    async create() {
      reports += 1;
      calls.push(`create r${reports}`);
      return `r${reports}`;
    },
    async putSegment(reportId, position) {
      if (failNextPut) {
        failNextPut = false;
        throw new Error('영상 업로드 실패 (502)');
      }
      calls.push(`put ${reportId}/${position}`);
    },
    async putFile(reportId, name) {
      calls.push(`file ${reportId}/${name}`);
    },
    async putMeta(reportId, meta) {
      metas.push(meta);
      calls.push(`meta ${reportId}`);
    },
    async finish(reportId, keepalive) {
      calls.push(`done ${reportId}${keepalive ? ' keepalive' : ''}`);
    },
  };
  return { api, calls, metas, failOnce: () => (failNextPut = true) };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test('segments go into one report in order, meta refreshed once per upload pass', async () => {
  const { api, calls, metas } = fakeApi();
  const reporter = new AutoReporter({ api, snapshot: () => ({ goal: '안경 벗기' }), maxPendingBytes: 1000 });
  reporter.push(segment(0));
  reporter.push(segment(1));
  await settle();
  await reporter.finish();
  assert.deepEqual(calls, ['create r1', 'put r1/0', 'put r1/1', 'meta r1', 'done r1']);
  const last = metas.at(-1) as { goal: string; segments: { file: string }[] };
  assert.equal(last.goal, '안경 벗기');
  assert.deepEqual(last.segments.map((entry) => entry.file), ['video-0000.webm', 'video-0001.webm']);
});

test('a failed upload is kept and retried', async () => {
  const { api, calls, failOnce } = fakeApi();
  const statuses: (string | null)[] = [];
  const reporter = new AutoReporter({
    api,
    snapshot: () => ({}),
    maxPendingBytes: 1000,
    retryMs: 5,
    onStatus: (status) => statuses.push(status.error),
  });
  failOnce();
  reporter.push(segment(0));
  await settle();
  assert.equal(reporter.status().pending, 1);
  assert.match(reporter.status().error ?? '', /502/);
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(reporter.status().pending, 0);
  assert.equal(reporter.status().error, null);
  assert.deepEqual(calls, ['create r1', 'put r1/0', 'meta r1']);
});

test('pending segments past maxPendingBytes drop the oldest', () => {
  const { api } = fakeApi();
  const reporter = new AutoReporter({ api, snapshot: () => ({}), maxPendingBytes: 25 });
  reporter.push(segment(0));
  reporter.push(segment(1));
  reporter.push(segment(2));
  assert.equal(reporter.status().dropped, 1);
});

test('after finish the next segment starts a new report; abandon marks done with keepalive', async () => {
  const { api, calls } = fakeApi();
  const reporter = new AutoReporter({ api, snapshot: () => ({}), maxPendingBytes: 1000 });
  reporter.push(segment(0));
  await reporter.finish();
  reporter.push(segment(1, 10, 'video/mp4'));
  await settle();
  reporter.abandon();
  assert.deepEqual(calls, ['create r1', 'put r1/0', 'meta r1', 'done r1', 'create r2', 'put r2/0', 'meta r2', 'done r2 keepalive']);
  assert.equal(reporter.status().reportId, null);
});

test('file extension follows the recorder mime', () => {
  assert.equal(extensionForMime('video/mp4;codecs=avc1'), 'mp4');
  assert.equal(extensionForMime('video/webm;codecs=vp8'), 'webm');
});

test('a comment rides in report.json; with no report yet it opens one', async () => {
  const { api, calls, metas } = fakeApi();
  const reporter = new AutoReporter({ api, snapshot: () => ({}), maxPendingBytes: 1000 });
  await reporter.comment('상자가 컵 대신 손을 따라감', new Date('2026-10-03T10:00:00Z'));
  reporter.push(segment(0));
  await settle();
  await reporter.comment('두 번째');
  assert.deepEqual(calls, ['create r1', 'meta r1', 'put r1/0', 'meta r1', 'meta r1']);
  const last = metas.at(-1) as { comments: { at: string; text: string }[] };
  assert.deepEqual(last.comments.map((entry) => entry.text), ['상자가 컵 대신 손을 따라감', '두 번째']);
  assert.equal(last.comments[0].at, '2026-10-03T10:00:00.000Z');
  assert.equal(reporter.status().comments, 2);
  await reporter.finish();
  assert.equal(reporter.status().comments, 0);
});

test('a logged guide call lands in report.json and its frames go up as files', async () => {
  const { api, calls, metas } = fakeApi();
  const reporter = new AutoReporter({ api, snapshot: () => ({}), maxPendingBytes: 1000 });
  reporter.logCall({ seq: 1, stage: 'confirm', response: { inferred_done: 'unsure' } }, [
    { name: 'call-0001-confirm-scene.jpg', blob: new Blob([new Uint8Array(5)]) },
    { name: 'call-0001-confirm-before_scene.jpg', blob: new Blob([new Uint8Array(5)]) },
  ]);
  await settle();
  assert.deepEqual(calls, ['create r1', 'file r1/call-0001-confirm-scene.jpg', 'file r1/call-0001-confirm-before_scene.jpg', 'meta r1']);
  const last = metas.at(-1) as { calls: { stage: string; response: { inferred_done: string } }[] };
  assert.equal(last.calls[0].response.inferred_done, 'unsure');
  assert.equal(reporter.status().calls, 1);
});
