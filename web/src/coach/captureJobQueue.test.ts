/** The capture worker's job order (`captureJobQueue.ts`): one at a time, tracking frames first. */
import test from 'node:test';
import assert from 'node:assert/strict';

import { CaptureJobQueue, type CapturePriority } from './captureJobQueue.ts';

interface Job {
  name: string;
  priority: CapturePriority;
}

function recordingQueue() {
  const started: string[] = [];
  let running = 0;
  let maxRunning = 0;
  const finishers: Array<() => void> = [];
  const queue = new CaptureJobQueue<Job>(async (job) => {
    started.push(job.name);
    running += 1;
    maxRunning = Math.max(maxRunning, running);
    await new Promise<void>((resolve) => finishers.push(resolve));
    running -= 1;
  });
  const finishNext = async () => {
    finishers.shift()?.();
    await new Promise<void>((resolve) => setImmediate(resolve));
  };
  return { queue, started, finishNext, maxRunning: () => maxRunning };
}

test('a tracking frame goes ahead of queued guide jobs, never ahead of the job already running', async () => {
  const { queue, started, finishNext, maxRunning } = recordingQueue();
  queue.push({ name: 'guide-1', priority: 'guide' });
  queue.push({ name: 'guide-2', priority: 'guide' });
  queue.push({ name: 'track-1', priority: 'track' });
  queue.push({ name: 'guide-3', priority: 'guide' });
  queue.push({ name: 'track-2', priority: 'track' });
  for (let i = 0; i < 5; i += 1) await finishNext();
  assert.deepEqual(started, ['guide-1', 'track-1', 'track-2', 'guide-2', 'guide-3']);
  assert.equal(maxRunning(), 1, 'jobs never overlap');
});

test('a runner that throws does not wedge the queue', async () => {
  const done: string[] = [];
  const queue = new CaptureJobQueue<Job>(async (job) => {
    if (job.name === 'bad') throw new Error('runner bug');
    done.push(job.name);
  });
  queue.push({ name: 'bad', priority: 'track' });
  queue.push({ name: 'after', priority: 'guide' });
  await new Promise<void>((resolve) => setImmediate(resolve));
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(done, ['after']);
});
