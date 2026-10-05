/**
 * Triggers (acquired / target_moved / anchor_lost / heartbeat) and the dispatch gate (pure).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  INITIAL_GATE_STATE,
  INITIAL_TRIGGER_STATE,
  decideDispatch,
  dropPending,
  enqueue,
  hasMoved,
  markDispatched,
  markPlan,
  markSettled,
  markTalk,
  noteDispatch,
  noteStageDispatch,
  remoteCallsInWindow,
  stepTriggers,
} from './triggers.ts';

const BOX = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 };

function obs(overrides: Record<string, unknown> = {}) {
  return { runId: 'run-1', trackId: 'trk-1', generation: 1, state: 'tracking', box: BOX, ...overrides } as never;
}

/** Feed (t, observation, accepted?) samples; return every fired trigger with its time. */
function run(
  samples: { t: number; o: unknown; accepted?: unknown; fence?: string; provider?: 'local' | 'deepseek' | null }[]
) {
  let state = INITIAL_TRIGGER_STATE;
  const fired: string[] = [];
  for (const sample of samples) {
    const step = stepTriggers(state, {
      observation: sample.o as never,
      nowMs: sample.t,
      fenceKey: sample.fence ?? 'f1',
      accepted: (sample.accepted ?? null) as never,
      followProvider: sample.provider === undefined ? 'local' : sample.provider,
    });
    state = step.state;
    for (const name of step.fired) fired.push(`${name}@${sample.t}`);
  }
  return { fired, state };
}

test('acquired fires once per (run, generation), again for a new generation or run', () => {
  const { fired } = run([
    { t: 0, o: obs({ state: 'acquiring', box: null }) },
    { t: 100, o: obs() },
    { t: 200, o: obs() },
    { t: 300, o: obs({ state: 'occluded', box: null }) },
    { t: 400, o: obs() },
    { t: 500, o: obs({ generation: 2 }) },
    { t: 600, o: obs({ runId: 'run-2' }) },
  ]);
  assert.deepEqual(fired, ['acquired@100', 'acquired@500', 'acquired@600']);
});

test('target_moved needs > 20% of frame width or > 1.5x area, sustained 300 ms, once per excursion', () => {
  const accepted = { runId: 'run-1', generation: 1, box: BOX };
  const moved = { ...BOX, x: 0.65 }; // centre shift 0.25 > 0.2
  const { fired } = run([
    { t: 0, o: obs(), accepted },
    { t: 100, o: obs({ box: moved }), accepted },
    { t: 350, o: obs({ box: moved }), accepted },
    { t: 400, o: obs({ box: moved }), accepted },
    { t: 500, o: obs({ box: moved }), accepted },
    { t: 900, o: obs({ box: moved }), accepted },
  ]);
  assert.deepEqual(
    fired.filter((name) => name.startsWith('target_moved')),
    ['target_moved@400']
  );
});

test('a brief move that comes back before 300 ms does not fire', () => {
  const accepted = { runId: 'run-1', generation: 1, box: BOX };
  const moved = { ...BOX, x: 0.7 };
  const { fired } = run([
    { t: 0, o: obs(), accepted },
    { t: 100, o: obs({ box: moved }), accepted },
    { t: 300, o: obs(), accepted },
    { t: 350, o: obs({ box: moved }), accepted },
    { t: 600, o: obs({ box: moved }), accepted },
  ]);
  assert.deepEqual(fired.filter((name) => name.startsWith('target_moved')), []);
});

test('movement thresholds: centre shift, area ratio, and the width-relative vertical shift', () => {
  assert.equal(hasMoved({ ...BOX, x: 0.59 }, BOX), false); // 0.19
  assert.equal(hasMoved({ ...BOX, x: 0.61 }, BOX), true); // 0.21
  const grown = { x: 0.35, y: 0.35, width: 0.3, height: 0.3 }; // area 2.25x, same centre
  assert.equal(hasMoved(grown, BOX), true);
  const slightly = { x: 0.38, y: 0.38, width: 0.24, height: 0.24 }; // 1.44x
  assert.equal(hasMoved(slightly, BOX), false);
  // 0.3 of frame HEIGHT on a 16:9 frame is 0.169 of frame width: not moved.
  assert.equal(hasMoved({ ...BOX, y: 0.7 }, BOX, undefined, 9 / 16), false);
  assert.equal(hasMoved({ ...BOX, y: 0.7 }, BOX, undefined, 1), true);
});

test('a new accepted box re-arms target_moved; another generation is never compared', () => {
  const first = { runId: 'run-1', generation: 1, box: BOX };
  const moved = { ...BOX, x: 0.7 };
  const second = { runId: 'run-1', generation: 1, box: moved };
  const { fired } = run([
    { t: 0, o: obs({ box: moved }), accepted: first },
    { t: 300, o: obs({ box: moved }), accepted: first },
    { t: 400, o: obs({ box: BOX }), accepted: second },
    { t: 700, o: obs({ box: BOX }), accepted: second },
    { t: 800, o: obs({ generation: 5, box: moved }), accepted: second },
    { t: 1200, o: obs({ generation: 5, box: moved }), accepted: second },
  ]);
  assert.deepEqual(fired.filter((name) => name.startsWith('target_moved')), ['target_moved@300', 'target_moved@700']);
});

test('anchor_lost fires after 1 s of lost, is cancelled by tracking and re-armed on re-entry', () => {
  const lost = obs({ state: 'lost', box: null });
  const { fired } = run([
    { t: 0, o: obs() },
    { t: 100, o: lost },
    { t: 900, o: lost },
    { t: 1000, o: obs() }, // cancelled at 900 ms of lost
    { t: 1100, o: lost },
    { t: 2050, o: lost },
    { t: 2100, o: lost },
    { t: 5000, o: lost },
  ]);
  assert.deepEqual(fired.filter((name) => name.startsWith('anchor_lost')), ['anchor_lost@2100']);
});

test('anchor_lost timer is cancelled by a new run and by a fence change', () => {
  const lost = obs({ state: 'lost', box: null });
  const byRun = run([
    { t: 0, o: lost },
    { t: 900, o: obs({ runId: 'run-2', state: 'lost', box: null }) },
    { t: 1500, o: obs({ runId: 'run-2', state: 'lost', box: null }) },
  ]);
  assert.deepEqual(byRun.fired, []);
  const byFence = run([
    { t: 0, o: lost, fence: 'f1' },
    { t: 900, o: lost, fence: 'f2' },
    { t: 1500, o: lost, fence: 'f2' },
    { t: 1900, o: lost, fence: 'f2' },
  ]);
  assert.deepEqual(byFence.fired, ['anchor_lost@1900']);
});

test('heartbeat: every 4 s while tracking with the local follower, 8 s with DeepSeek, never when not tracking', () => {
  const samples = [];
  for (let t = 0; t <= 9_000; t += 500) samples.push({ t, o: obs() });
  const local = run(samples);
  assert.deepEqual(local.fired, ['acquired@0', 'heartbeat@4000', 'heartbeat@8000']);
  const remote = run(samples.map((sample) => ({ ...sample, provider: 'deepseek' as const })));
  assert.deepEqual(remote.fired, ['acquired@0', 'heartbeat@8000']);
  const occluded = [];
  for (let t = 0; t <= 11_000; t += 500) occluded.push({ t, o: obs({ state: 'occluded', box: null }) });
  assert.deepEqual(run(occluded).fired, []);
});

test('another trigger does not restart the heartbeat; only a follow dispatch does', () => {
  const accepted = { runId: 'run-1', generation: 1, box: BOX };
  const moved = { ...BOX, x: 0.7 };
  const samples: { t: number; o: unknown; accepted?: unknown; provider?: 'local' | 'deepseek' | null }[] = [];
  for (let t = 0; t <= 8_500; t += 500) samples.push({ t, o: obs({ box: t >= 5000 && t <= 6000 ? moved : BOX }), accepted, provider: 'deepseek' });
  // target_moved fires at 5500 (no follow dispatch noted): the heartbeat is still due at 8000.
  assert.deepEqual(run(samples).fired, ['acquired@0', 'target_moved@5500', 'heartbeat@8000']);
});

test('a dispatch for another reason restarts the heartbeat interval', () => {
  let state = stepTriggers(INITIAL_TRIGGER_STATE, {
    observation: obs(),
    nowMs: 0,
    fenceKey: 'f',
    accepted: null,
    followProvider: 'local',
  }).state;
  state = noteDispatch(state, 4000);
  const at6 = stepTriggers(state, { observation: obs(), nowMs: 6000, fenceKey: 'f', accepted: null, followProvider: 'local' });
  assert.deepEqual(at6.fired, []);
  const at9 = stepTriggers(at6.state, { observation: obs(), nowMs: 9000, fenceKey: 'f', accepted: null, followProvider: 'local' });
  assert.deepEqual(at9.fired, ['heartbeat']);
});

test('the Clef follower uses the local heartbeat interval', () => {
  const tick = (state: typeof INITIAL_TRIGGER_STATE, t: number) =>
    stepTriggers(state, { observation: obs(), nowMs: t, fenceKey: 'f', accepted: null, followProvider: 'clef' });
  const state = tick(INITIAL_TRIGGER_STATE, 0).state;
  assert.deepEqual(tick(state, 3999).fired, []);
  assert.deepEqual(tick(state, 4000).fired, ['heartbeat']);
});

test('a confirm dispatch does not postpone the heartbeat; a follow dispatch does', () => {
  const tick = (state: typeof INITIAL_TRIGGER_STATE, t: number) =>
    stepTriggers(state, { observation: obs(), nowMs: t, fenceKey: 'f', accepted: null, followProvider: 'deepseek' });
  let state = tick(INITIAL_TRIGGER_STATE, 0).state; // tracking from 0 (acquired)
  state = noteStageDispatch(state, 'follow', 1000); // the acquired follow leaves: due at 9000
  const fired: string[] = [];
  for (let t = 1500; t <= 12_000; t += 500) {
    // confirm(step_done) right after an advance, and another confirm later: neither moves the heartbeat.
    if (t === 2500 || t === 7000) state = noteStageDispatch(state, 'confirm', t);
    const step = tick(state, t);
    state = step.state;
    for (const name of step.fired) fired.push(`${name}@${t}`);
  }
  assert.deepEqual(fired, ['heartbeat@9000']);
  // The same sequence with a follow at 2500 instead moves it to 10500.
  let other = noteStageDispatch(tick(INITIAL_TRIGGER_STATE, 0).state, 'follow', 1000);
  const firedFollow: string[] = [];
  for (let t = 1500; t <= 12_000; t += 500) {
    if (t === 2500) other = noteStageDispatch(other, 'follow', t);
    const step = tick(other, t);
    other = step.state;
    for (const name of step.fired) firedFollow.push(`${name}@${t}`);
  }
  assert.deepEqual(firedFollow, ['heartbeat@10500']);
});

const GATE = {
  followLocalFloorMs: 1000,
  followRemoteFloorMs: 2200,
  confirmFloorMs: 2200,
  remoteSpacingMs: 1200,
  remoteBudget: 3,
  remoteWindowMs: 120_000,
  remotePerMinute: 20,
};
const follow = (trigger: string, lane: 'local' | 'remote' = 'local') => ({ kind: 'follow' as const, trigger: trigger as never, lane });
const confirm = (trigger: string, meta?: Record<string, unknown>) => ({ kind: 'confirm' as const, trigger: trigger as never, lane: 'remote' as const, meta });

test('gate: per stage one in flight + one pending (newest wins, never displaced by a less important call)', () => {
  let gate = enqueue(INITIAL_GATE_STATE, follow('acquired'));
  const first = decideDispatch(gate, 'follow', 0, GATE);
  assert.equal(first.kind, 'dispatch');
  gate = markDispatched(gate, (first as { call: never }).call, 0, GATE);
  assert.equal(gate.follow.pending, null);
  gate = enqueue(gate, follow('heartbeat'));
  gate = enqueue(gate, follow('target_changed'));
  assert.equal(gate.follow.pending?.trigger, 'target_changed');
  gate = enqueue(gate, follow('heartbeat'));
  assert.equal(gate.follow.pending?.trigger, 'target_changed', 'a heartbeat never displaces another follow');
  assert.equal(decideDispatch(gate, 'follow', 10, GATE).kind, 'busy');
  // Confirms have their own slot: a step re-check outranks a goal check; the completion recheck outranks both.
  gate = enqueue(gate, confirm('step_done'));
  gate = enqueue(gate, confirm('goal_check'));
  assert.equal(gate.confirm.pending?.trigger, 'step_done');
  gate = enqueue(gate, confirm('goal_check', { recheck: true }));
  assert.equal(gate.confirm.pending?.trigger, 'goal_check');
  gate = enqueue(gate, confirm('step_done'));
  assert.equal((gate.confirm.pending?.meta as { recheck?: boolean })?.recheck, true);
  assert.equal(gate.follow.pending?.trigger, 'target_changed');
});

test('gate: a confirm in flight never blocks the next follow (separate slots and floors)', () => {
  // follow (DeepSeek) at 0, its answer advances a step and a confirm(step_done) leaves at 1300.
  let gate = markSettled(markDispatched(INITIAL_GATE_STATE, follow('acquired', 'remote'), 0, GATE), 'follow');
  gate = enqueue(gate, confirm('step_done'));
  assert.equal(decideDispatch(gate, 'confirm', 1300, GATE).kind, 'dispatch');
  gate = markDispatched(gate, gate.confirm.pending!, 1300, GATE);
  // A change is seen at 1600: the follow floor (2.2 s from 0) and the paid spacing (1.2 s from 1300) apply,
  // the confirm in flight does not.
  gate = enqueue(gate, follow('target_changed', 'remote'));
  assert.deepEqual(decideDispatch(gate, 'follow', 1600, GATE), { kind: 'wait', waitMs: 900 });
  assert.equal(decideDispatch(gate, 'follow', 2500, GATE).kind, 'dispatch');
  // Local follows: 1.0 s floor, no paid spacing.
  let local = markSettled(markDispatched(INITIAL_GATE_STATE, follow('acquired'), 0, GATE), 'follow');
  local = markDispatched(enqueue(local, confirm('step_done')), confirm('step_done'), 100, GATE);
  local = enqueue(local, follow('target_changed'));
  assert.deepEqual(decideDispatch(local, 'follow', 400, GATE), { kind: 'wait', waitMs: 600 });
  assert.equal(decideDispatch(local, 'follow', 1000, GATE).kind, 'dispatch');
  // Confirm floor 2.2 s between confirms.
  local = enqueue(markSettled(local, 'confirm'), confirm('goal_check'));
  assert.deepEqual(decideDispatch(local, 'confirm', 1300, GATE), { kind: 'wait', waitMs: 1000 });
});

test('gate: the plan is exclusive', () => {
  let gate = markPlan(INITIAL_GATE_STATE, true, 0, GATE);
  gate = enqueue(gate, follow('acquired'));
  assert.equal(decideDispatch(gate, 'follow', 5000, GATE).kind, 'busy');
  gate = markPlan(gate, false, 3000, GATE);
  assert.equal(decideDispatch(gate, 'follow', 5000, GATE).kind, 'dispatch');
});

test('gate: DeepSeek budget is rolling (N per window, plan included) and keeps the pending call; local calls are free', () => {
  let gate = markPlan(INITIAL_GATE_STATE, true, 0, GATE);
  gate = markPlan(gate, false, 2000, GATE);
  for (const at of [10_000, 20_000]) gate = markSettled(markDispatched(gate, confirm('goal_check'), at, GATE), 'confirm');
  gate = enqueue(gate, confirm('goal_check'));
  assert.deepEqual(decideDispatch(gate, 'confirm', 30_000, GATE), { kind: 'capped', waitMs: 90_000, limit: 'budget' });
  assert.equal(gate.confirm.pending?.trigger, 'goal_check', 'the pending call is kept for when the window frees');
  assert.equal(decideDispatch(gate, 'confirm', 120_001, GATE).kind, 'dispatch');
  assert.equal(remoteCallsInWindow(gate, 30_000, GATE), 3);
  assert.equal(remoteCallsInWindow(gate, 130_001, GATE), 1);
  gate = enqueue(gate, follow('heartbeat'));
  assert.equal(decideDispatch(gate, 'follow', 30_000, GATE).kind, 'dispatch');
  gate = dropPending(gate, 'confirm');
  assert.equal(gate.confirm.pending, null);
});

test('gate: the server per-minute guide ceiling is mirrored', () => {
  const config = { ...GATE, remoteBudget: 30, remotePerMinute: 2 };
  let gate = INITIAL_GATE_STATE;
  for (const at of [0, 5_000]) gate = markSettled(markDispatched(gate, confirm('goal_check'), at, config), 'confirm');
  gate = enqueue(gate, follow('heartbeat', 'remote'));
  assert.deepEqual(decideDispatch(gate, 'follow', 10_000, config), { kind: 'capped', waitMs: 50_000, limit: 'per_minute' });
});

test('a talk shares the confirm floor and counts as a DeepSeek-backed call', () => {
  const config = {
    followLocalFloorMs: 1000,
    followRemoteFloorMs: 2200,
    confirmFloorMs: 2200,
    remoteSpacingMs: 1200,
    remoteBudget: 30,
    remoteWindowMs: 120_000,
    remotePerMinute: 20,
  };
  let state = markTalk(INITIAL_GATE_STATE, 1_000, config);
  assert.equal(state.confirm.lastAt, 1_000);
  assert.equal(state.confirm.inFlight, null);
  assert.equal(state.lastRemoteAt, 1_000);
  assert.deepEqual(state.remoteTimes, [1_000]);
  assert.equal(remoteCallsInWindow(state, 2_000, config), 1);
  state = enqueue(state, { kind: 'confirm', trigger: 'goal_check', lane: 'remote' });
  assert.deepEqual(decideDispatch(state, 'confirm', 2_000, config), { kind: 'wait', waitMs: 1_200 });
});

test('while a talk waits, the remote lane is held (no starvation); local follows still go', () => {
  const config = {
    followLocalFloorMs: 1000,
    followRemoteFloorMs: 2200,
    confirmFloorMs: 2200,
    remoteSpacingMs: 1200,
    remoteBudget: 30,
    remoteWindowMs: 120_000,
    remotePerMinute: 20,
  };
  const confirm = enqueue(INITIAL_GATE_STATE, { kind: 'confirm', trigger: 'goal_check', lane: 'remote' });
  assert.deepEqual(decideDispatch(confirm, 'confirm', 10_000, config, true), { kind: 'busy' });
  assert.equal(decideDispatch(confirm, 'confirm', 10_000, config, false).kind, 'dispatch');
  const local = enqueue(INITIAL_GATE_STATE, { kind: 'follow', trigger: 'heartbeat', lane: 'local' });
  assert.equal(decideDispatch(local, 'follow', 10_000, config, true).kind, 'dispatch');
});
