/**
 * Object-centric change trigger `target_changed` (pure part): has the tracked object's APPEARANCE changed
 * since the frame of the last accepted guide answer, even though its box did not move?
 *
 * `target_moved` only sees the box (centre shift, area ratio). Changes that happen inside a box that stays
 * put — digits on a display, a lid opening, something coming off the object but staying near it — never
 * move the box, so they need a look at the pixels inside it:
 *
 * - Every `sampleMs` while tracking, the live frame is cropped to the tracked box grown by `pad` (a
 *   fraction of the box on every side, clamped to the frame) and reduced to a `grid`×`grid` luma grid by an
 *   area-averaging ladder (the DOM part is `anchorSampler.ts`). The crop FOLLOWS the box and is resampled to
 *   a fixed grid, so the grid is normalised by the box: a translation or a scale change of the whole object
 *   alone reproduces the same grid.
 * - The reference is the grid sampled with the frame of the last ACCEPTED (bound) guide answer, for the
 *   same anchor identity (run, track, generation). Without one there is no trigger.
 * - The score removes the mean brightness shift (exposure) and averages what is left:
 *     shift = mean(cur) − mean(ref);  residual_i = |cur_i − ref_i − shift|;  mean = Σ residual_i / N
 *   `patch` (the strongest `patchCells`×`patchCells` block mean of the residual) is reported for the trace.
 * - The trigger fires when `mean` stays above `threshold` for `sustain` consecutive samples, once per
 *   excursion: it re-arms when a newer reference arrives (a newer answer was accepted) or the score falls
 *   back to or below the threshold.
 *
 * No task knowledge: the module knows boxes and luma only. The threshold is a measured policy value (see
 * `APPEARANCE_THRESHOLD`). Nothing here imports a value from another module, so `node --test` runs it.
 */
import type { Box } from '@visual-coach/visual-tools';

export interface AppearanceConfig {
  /** Cells per grid side. */
  grid: number;
  /** Box growth on every side, as a fraction of the box's own width/height (the focus ring's default). */
  pad: number;
  /** Sampling period while tracking. */
  sampleMs: number;
  /** Mean residual (0..1 luma) above which a sample counts as changed. */
  threshold: number;
  /** Consecutive changed samples needed to fire. */
  sustain: number;
  /** Block size (cells) of the reported `patch` score. */
  patchCells: number;
}

/**
 * Mean-residual threshold, measured 2026-10-01 on the three fixtures (dev host, headless Chromium 153, this
 * module's sampler on 4 fps frames of the pillarboxed 1280×720 clips, boxes = the production tracker's
 * boxes from two earlier E2E runs; pairs = reference sample vs a sample ≤ 3 s later):
 *   - nothing changes, only the box (same frame, another time's box; remote held still) n=360:
 *     p95 0.026, max 0.029 — the stationary floor;
 *   - hands moving on/around an unchanged object: remote thumb n=52 max 0.042; glasses on face n=123
 *     p50 0.144; case in hands n=330 p50 0.147 — these DO fire (a hand in the box is a real change);
 *   - object state changed: glasses off the face n=97 min 0.154; lid closed n=102 min 0.122;
 *   - display digits changed (26→27, 27→28) n=128 min 0.011, p50 0.021, max 0.045 — BELOW the stationary
 *     floor: a digit change alone is not visible to this trigger (the box jitter is larger).
 * 0.06 is 2.1× the stationary max and ≤ 0.5× the smallest measured state change.
 */
export const APPEARANCE_THRESHOLD = 0.06;

export const DEFAULT_APPEARANCE_CONFIG: AppearanceConfig = {
  grid: 24,
  pad: 0.15,
  sampleMs: 250,
  threshold: APPEARANCE_THRESHOLD,
  sustain: 2,
  patchCells: 6,
};

export interface PixelRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

/**
 * The crop, in source pixels, for a normalized box: grown by `pad`×(width, height) on every side, clamped
 * to the frame, snapped outward to whole pixels. Null when the box is degenerate or the crop is < 2 px.
 */
export function appearanceCrop(box: Box, frame: { width: number; height: number }, pad: number): PixelRect | null {
  if (![box.x, box.y, box.width, box.height].every(Number.isFinite)) return null;
  if (box.width <= 0 || box.height <= 0 || frame.width <= 0 || frame.height <= 0) return null;
  const p = Number.isFinite(pad) && pad > 0 ? pad : 0;
  const left = clamp01(box.x - box.width * p) * frame.width;
  const top = clamp01(box.y - box.height * p) * frame.height;
  const right = clamp01(box.x + box.width * (1 + p)) * frame.width;
  const bottom = clamp01(box.y + box.height * (1 + p)) * frame.height;
  const x = Math.floor(left);
  const y = Math.floor(top);
  const width = Math.min(frame.width, Math.ceil(right)) - x;
  const height = Math.min(frame.height, Math.ceil(bottom)) - y;
  if (width < 2 || height < 2) return null;
  return { x, y, width, height };
}

/**
 * The sizes an area-averaging ladder passes through from a crop to the grid: each axis is halved
 * independently while it is more than twice the grid, so every draw reduces by at most 2× per axis and the
 * final draw (to grid×grid) also reduces by less than 2×. Empty when the crop is already ≤ 2× the grid.
 */
export function ladderSizes(width: number, height: number, grid: number): Array<{ width: number; height: number }> {
  const out: Array<{ width: number; height: number }> = [];
  let w = Math.max(1, Math.round(width));
  let h = Math.max(1, Math.round(height));
  while (w > grid * 2 || h > grid * 2) {
    w = w > grid * 2 ? Math.max(grid, Math.round(w / 2)) : w;
    h = h > grid * 2 ? Math.max(grid, Math.round(h / 2)) : h;
    out.push({ width: w, height: h });
  }
  return out;
}

/** RGBA bytes → luma in 0..1 (BT.601), one value per pixel. */
export function lumaFromRgba(data: ArrayLike<number>, cells: number): Float32Array {
  const grid = new Float32Array(cells);
  for (let i = 0; i < cells; i += 1) {
    const o = i * 4;
    grid[i] = (0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2]) / 255;
  }
  return grid;
}

export interface AppearanceScore {
  /** Mean residual after removing the mean brightness shift (0..1). */
  mean: number;
  /** Strongest block mean of the residual (0..1). */
  patch: number;
  /** The removed brightness shift (cur − ref), signed. */
  shift: number;
}

/** Compare two grids of the same size. A size mismatch is "everything changed" (fails toward one follow). */
export function compareAppearance(
  reference: Float32Array,
  current: Float32Array,
  patchCells: number = DEFAULT_APPEARANCE_CONFIG.patchCells
): AppearanceScore {
  const n = reference.length;
  if (n === 0 || n !== current.length) return { mean: 1, patch: 1, shift: 0 };
  let sumR = 0;
  let sumC = 0;
  for (let i = 0; i < n; i += 1) {
    sumR += reference[i];
    sumC += current[i];
  }
  const shift = (sumC - sumR) / n;
  const residual = new Float32Array(n);
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const r = Math.abs(current[i] - reference[i] - shift);
    residual[i] = r;
    total += r;
  }
  const side = Math.round(Math.sqrt(n));
  let patch = 0;
  if (side * side === n && patchCells > 0 && patchCells <= side) {
    const blocks = Math.floor(side / patchCells);
    for (let by = 0; by < blocks; by += 1) {
      for (let bx = 0; bx < blocks; bx += 1) {
        let s = 0;
        for (let y = 0; y < patchCells; y += 1) {
          const row = (by * patchCells + y) * side + bx * patchCells;
          for (let x = 0; x < patchCells; x += 1) s += residual[row + x];
        }
        const m = s / (patchCells * patchCells);
        if (m > patch) patch = m;
      }
    }
  }
  return { mean: total / n, patch, shift };
}

/** The anchor identity a grid was sampled on. */
export interface AppearanceAnchor {
  runId: string;
  trackId: string;
  generation: number;
}

export interface AppearanceReference extends AppearanceAnchor {
  grid: Float32Array;
}

export interface AppearanceState {
  reference: AppearanceReference | null;
  streak: number;
  fired: boolean;
  lastSampleAt: number | null;
  /** The newest score (for the trace), null before a comparable sample. */
  lastScore: AppearanceScore | null;
}

export const INITIAL_APPEARANCE_STATE: AppearanceState = {
  reference: null,
  streak: 0,
  fired: false,
  lastSampleAt: null,
  lastScore: null,
};

/** Whether a sample is due (`sampleMs` since the previous one). */
export function appearanceSampleDue(state: AppearanceState, nowMs: number, config: AppearanceConfig = DEFAULT_APPEARANCE_CONFIG): boolean {
  return state.lastSampleAt === null || nowMs - state.lastSampleAt >= config.sampleMs;
}

export interface AppearanceInput {
  nowMs: number;
  /** The live anchor now (tracking only), or null. */
  anchor: AppearanceAnchor | null;
  /** The grid sampled now for that anchor, or null when no sample could be taken. */
  grid: Float32Array | null;
  /** The grid of the last accepted answer, or null. */
  reference: AppearanceReference | null;
}

export interface AppearanceStep {
  state: AppearanceState;
  fired: boolean;
}

function sameAnchor(a: AppearanceAnchor, b: AppearanceAnchor): boolean {
  return a.runId === b.runId && a.trackId === b.trackId && a.generation === b.generation;
}

/** One sample. Call when `appearanceSampleDue`; a null grid/anchor resets the streak (no evidence). */
export function stepAppearance(
  previous: AppearanceState,
  input: AppearanceInput,
  config: AppearanceConfig = DEFAULT_APPEARANCE_CONFIG
): AppearanceStep {
  let state: AppearanceState = { ...previous, lastSampleAt: input.nowMs };
  if (input.reference !== state.reference) {
    // A newer accepted answer (or none): the excursion is measured from it, freshly armed.
    state = { ...state, reference: input.reference, streak: 0, fired: false, lastScore: null };
  }
  const reference = state.reference;
  if (!reference || !input.anchor || !input.grid || !sameAnchor(reference, input.anchor)) {
    return { state: { ...state, streak: 0, lastScore: null }, fired: false };
  }
  const score = compareAppearance(reference.grid, input.grid, config.patchCells);
  if (score.mean <= config.threshold) {
    return { state: { ...state, streak: 0, fired: false, lastScore: score }, fired: false };
  }
  const streak = state.streak + 1;
  if (!state.fired && streak >= config.sustain) {
    return { state: { ...state, streak, fired: true, lastScore: score }, fired: true };
  }
  return { state: { ...state, streak, lastScore: score }, fired: false };
}
