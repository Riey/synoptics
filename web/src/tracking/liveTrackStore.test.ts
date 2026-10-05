import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  LIVE_BOX_FRESHNESS_MS,
  createLiveTrackStore,
  isLiveBoxDrawable,
  liveBoxAgeMs,
  type LiveTrackSnapshot,
} from './liveTrackStore.ts';

const BOX = { x: 0.2, y: 0.3, width: 0.25, height: 0.2 };

function snap(overrides: Partial<LiveTrackSnapshot> = {}): LiveTrackSnapshot {
  return {
    runId: 'run-1',
    trackId: 'trk-1',
    generation: 1,
    state: 'tracking',
    box: BOX,
    capturedAt: 10_000,
    version: 3,
    target: 'mug',
    ...overrides,
  };
}

test('the freshness limit is 300 ms', () => {
  assert.equal(LIVE_BOX_FRESHNESS_MS, 300);
});

test('isLiveBoxDrawable: tracking + box + age within the limit (inclusive)', () => {
  const s = snap();
  assert.equal(isLiveBoxDrawable(s, 10_000), true);
  assert.equal(isLiveBoxDrawable(s, 10_300), true);
  assert.equal(isLiveBoxDrawable(s, 10_301), false);
});

test('isLiveBoxDrawable hides every non-tracking state even with a box', () => {
  for (const state of ['acquiring', 'occluded', 'lost', 'unavailable'] as const) {
    assert.equal(isLiveBoxDrawable(snap({ state }), 10_000), false, state);
  }
});

test('isLiveBoxDrawable hides a missing snapshot or box', () => {
  assert.equal(isLiveBoxDrawable(null, 10_000), false);
  assert.equal(isLiveBoxDrawable(snap({ box: null }), 10_000), false);
});

test('isLiveBoxDrawable honours a custom limit and treats a future capture as age 0', () => {
  assert.equal(isLiveBoxDrawable(snap(), 10_050, 40), false);
  assert.equal(isLiveBoxDrawable(snap(), 10_040, 40), true);
  assert.equal(isLiveBoxDrawable(snap({ capturedAt: 12_000 }), 10_000), true);
  assert.equal(liveBoxAgeMs(snap({ capturedAt: 12_000 }), 10_000), 0);
});

test('isLiveBoxDrawable fails closed on a non-numeric capture time', () => {
  assert.equal(isLiveBoxDrawable(snap({ capturedAt: Number.NaN }), 10_000), false);
});

test('liveBoxAgeMs is capture-to-now and null without a snapshot', () => {
  assert.equal(liveBoxAgeMs(snap(), 10_120), 120);
  assert.equal(liveBoxAgeMs(null, 10_120), null);
});

test('store: publish notifies subscribers and both getters return the same snapshot', () => {
  const store = createLiveTrackStore();
  let calls = 0;
  const unsubscribe = store.subscribe(() => {
    calls += 1;
  });
  assert.equal(store.getSnapshot(), null);
  store.publish(snap());
  assert.equal(calls, 1);
  const first = store.getSnapshot();
  assert.ok(first);
  assert.equal(store.current(), first);
  assert.deepEqual(first.box, BOX);
  // Stable between publishes (useSyncExternalStore requirement), new identity after one.
  assert.equal(store.getSnapshot(), first);
  store.publish(snap({ version: 4 }));
  assert.notEqual(store.getSnapshot(), first);
  assert.equal(store.getSnapshot()?.version, 4);
  unsubscribe();
  store.publish(snap({ version: 5 }));
  assert.equal(calls, 2);
});

test('store: the published snapshot is frozen and detached from the caller box', () => {
  const store = createLiveTrackStore();
  const box = { ...BOX };
  store.publish(snap({ box }));
  box.x = 0.9;
  const current = store.current();
  assert.equal(current?.box?.x, BOX.x);
  assert.ok(Object.isFrozen(current));
});

test('store: a box outside tracking is never stored', () => {
  const store = createLiveTrackStore();
  store.publish(snap({ state: 'occluded', box: BOX }));
  assert.equal(store.current()?.box, null);
  assert.equal(store.current()?.state, 'occluded');
});

test('store: clear empties and notifies once; clearing an empty store is silent', () => {
  const store = createLiveTrackStore();
  let calls = 0;
  store.subscribe(() => {
    calls += 1;
  });
  store.clear();
  assert.equal(calls, 0);
  store.publish(snap());
  store.clear();
  assert.equal(calls, 2);
  assert.equal(store.getSnapshot(), null);
  store.clear();
  assert.equal(calls, 2);
});

test('store: a listener may unsubscribe while being notified without skipping others', () => {
  const store = createLiveTrackStore();
  const seen: string[] = [];
  const offA = store.subscribe(() => {
    seen.push('a');
    offA();
  });
  store.subscribe(() => seen.push('b'));
  store.publish(snap());
  store.publish(snap({ version: 9 }));
  assert.deepEqual(seen, ['a', 'b', 'b']);
});
