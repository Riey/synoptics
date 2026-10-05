/**
 * target_changed (pure part): crop geometry, the area-averaging ladder sizes, the shift-removed residual and
 * the sustain/once-per-excursion rule.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  APPEARANCE_THRESHOLD,
  DEFAULT_APPEARANCE_CONFIG,
  INITIAL_APPEARANCE_STATE,
  appearanceCrop,
  appearanceSampleDue,
  compareAppearance,
  ladderSizes,
  lumaFromRgba,
  stepAppearance,
} from './anchorAppearance.ts';

const G = DEFAULT_APPEARANCE_CONFIG.grid;
const FRAME = { width: 1280, height: 720 };

function flat(value: number, n = G * G) {
  return new Float32Array(n).fill(value);
}

test('crop: the box grown by pad on every side, clamped to the frame, whole pixels', () => {
  assert.deepEqual(appearanceCrop({ x: 0.5, y: 0.5, width: 0.1, height: 0.2 }, FRAME, 0.15), {
    x: 620,
    y: 338,
    width: 168, // floor(620.8) .. ceil(787.2)
    height: 188,
  });
  // Touching the corner: clamped, never negative.
  const corner = appearanceCrop({ x: 0, y: 0, width: 0.2, height: 0.2 }, FRAME, 0.15)!;
  assert.equal(corner.x, 0);
  assert.equal(corner.y, 0);
  assert.equal(corner.width, Math.ceil(0.23 * 1280));
  assert.equal(appearanceCrop({ x: 0.5, y: 0.5, width: 0, height: 0.2 }, FRAME, 0.15), null);
  assert.equal(appearanceCrop({ x: Number.NaN, y: 0.5, width: 0.1, height: 0.2 }, FRAME, 0.15), null);
  assert.equal(appearanceCrop({ x: 0.5, y: 0.5, width: 0.0005, height: 0.0005 }, FRAME, 0), null);
});

test('ladder: every draw reduces by at most 2x per axis, axes independently, final draw < 2x', () => {
  for (const [w, h] of [
    [166, 188],
    [215, 720],
    [1280, 720],
    [30, 900],
  ]) {
    let from = { width: w, height: h };
    for (const next of ladderSizes(w, h, G)) {
      assert.ok(from.width / next.width <= 2.01 && from.height / next.height <= 2.01, `${w}x${h}`);
      from = next;
    }
    assert.ok(from.width <= G * 2 && from.height <= G * 2, `${w}x${h} ends within 2x of the grid`);
  }
  assert.deepEqual(ladderSizes(40, 30, G), []);
});

test('luma from RGBA is BT.601 in 0..1', () => {
  const grid = lumaFromRgba([255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255], 3);
  assert.ok(Math.abs(grid[0] - 1) < 1e-6);
  assert.equal(grid[1], 0);
  assert.ok(Math.abs(grid[2] - 0.299) < 1e-6);
});

test('compare: identical and uniformly re-exposed grids score 0; a local change scores its share', () => {
  const base = new Float32Array(G * G).map((_, i) => (i % 7) / 10);
  assert.equal(compareAppearance(base, base).mean, 0);
  const brighter = base.map((v) => v + 0.2);
  const shifted = compareAppearance(base, brighter);
  assert.ok(shifted.mean < 1e-6, 'a uniform brightness shift is removed');
  assert.ok(Math.abs(shifted.shift - 0.2) < 1e-6);
  // One 6x6 block (1/16 of the grid) changes by 0.48: mean ≈ 0.48/16 after the shift is removed.
  const local = Float32Array.from(base);
  for (let y = 0; y < 6; y += 1) for (let x = 0; x < 6; x += 1) local[y * G + x] += 0.48;
  const score = compareAppearance(base, local);
  const shift = 0.48 / 16;
  const expected = (36 * (0.48 - shift) + (G * G - 36) * shift) / (G * G);
  assert.ok(Math.abs(score.mean - expected) < 1e-5, `${score.mean} vs ${expected}`);
  assert.ok(Math.abs(score.patch - (0.48 - shift)) < 1e-5);
  assert.deepEqual(compareAppearance(base, new Float32Array(10)), { mean: 1, patch: 1, shift: 0 });
});

const anchor = { runId: 'run-1', trackId: 'trk-1', generation: 1 };
const reference = { ...anchor, grid: flat(0.5) };
function changed(amount: number) {
  const grid = flat(0.5);
  for (let i = 0; i < grid.length / 2; i += 1) grid[i] += amount; // half the cells: mean residual = amount/2
  return grid;
}

function feed(samples: { t: number; grid: Float32Array | null; ref?: typeof reference | null; a?: typeof anchor | null }[]) {
  let state = INITIAL_APPEARANCE_STATE;
  const fired: number[] = [];
  for (const s of samples) {
    const step = stepAppearance(state, {
      nowMs: s.t,
      anchor: s.a === undefined ? anchor : s.a,
      grid: s.grid,
      reference: s.ref === undefined ? reference : s.ref,
    });
    state = step.state;
    if (step.fired) fired.push(s.t);
  }
  return { fired, state };
}

test('fires after 2 consecutive changed samples, once per excursion; re-arms below threshold or on a new reference', () => {
  const big = changed(APPEARANCE_THRESHOLD * 4); // mean 2x threshold
  const small = changed(APPEARANCE_THRESHOLD); // mean 0.5x threshold
  assert.deepEqual(feed([{ t: 0, grid: big }, { t: 250, grid: small }, { t: 500, grid: big }]).fired, []);
  assert.deepEqual(
    feed([
      { t: 0, grid: big },
      { t: 250, grid: big },
      { t: 500, grid: big },
      { t: 750, grid: small },
      { t: 1000, grid: big },
      { t: 1250, grid: big },
    ]).fired,
    [250, 1250]
  );
  const newer = { ...anchor, grid: flat(0.5) };
  assert.deepEqual(
    feed([
      { t: 0, grid: big },
      { t: 250, grid: big },
      { t: 500, grid: big, ref: newer },
      { t: 750, grid: big, ref: newer },
    ]).fired,
    [250, 750]
  );
});

test('no reference, another anchor identity or no sample: never fires, streak resets', () => {
  const big = changed(APPEARANCE_THRESHOLD * 4);
  assert.deepEqual(feed([{ t: 0, grid: big, ref: null }, { t: 250, grid: big, ref: null }]).fired, []);
  const other = { ...anchor, generation: 2 };
  assert.deepEqual(feed([{ t: 0, grid: big, a: other }, { t: 250, grid: big, a: other }]).fired, []);
  assert.deepEqual(feed([{ t: 0, grid: big }, { t: 250, grid: null }, { t: 500, grid: big }]).fired, []);
  assert.deepEqual(feed([{ t: 0, grid: big }, { t: 250, grid: big, a: null }, { t: 500, grid: big }]).fired, []);
});

test('sampling cadence is sampleMs', () => {
  assert.equal(appearanceSampleDue(INITIAL_APPEARANCE_STATE, 0), true);
  const state = { ...INITIAL_APPEARANCE_STATE, lastSampleAt: 1000 };
  assert.equal(appearanceSampleDue(state, 1249), false);
  assert.equal(appearanceSampleDue(state, 1250), true);
});
