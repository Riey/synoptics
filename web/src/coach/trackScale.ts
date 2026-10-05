/**
 * Adaptive tracking upload size (`?trackscale=auto`, v2 only, off by default — the operator decides).
 *
 * The tracking loop is pipelined (`framePrefetch.ts`): its period approaches max(capture, round trip), and
 * the round trip measured on both demo devices is ~55 ms p50 (2026-10-04, Mac Safari and Windows Edge,
 * docs/e2e/2026-10-04-client-perf-devices.md on the latency branch). The capture's own CPU work (draw +
 * encode, the worker's stage times) is what a slow device can shrink: at 960×540 it touches 56% of the
 * pixels of 1280×720, and the replay check (latency 82051da) found tracking robustness unchanged at
 * 960×540 q0.88 with ~0.5 px less box precision.
 *
 * - Down: once the work's EWMA exceeds `SCALE_DOWN_ABOVE_MS` (40 ms). With the wait for a new camera frame
 *   (up to 33 ms) and the reply hop on top, such a capture no longer fits under the 55 ms round trip and
 *   starts to set the period. The Mac's whole capture measured 16 ms p50, so it never scales down.
 * - Up: only when the reduced work predicts a full-size cost under 25 ms (EWMA < 25 × 0.5625 ≈ 14 ms). The
 *   gap between the two thresholds and `MIN_DWELL_MS` keep it from flapping.
 * - At least `MIN_SAMPLES` captures at the current size before any change, so one slow outlier (a GC, a
 *   tab switch) never flips it.
 *
 * The tracker resizes every frame to its model input and boxes are normalised, so a size change between
 * two frames of a run moves no box. Pure, so `node --test` runs it.
 */
import { DurationEstimate } from '../tracking/framePrefetch';
import type { TrackScaleMode } from './captureMode';

/** Long side of the reduced tracking upload (1280×720 → 960×540). */
export const REDUCED_LONG_SIDE = 960;
export const SCALE_DOWN_ABOVE_MS = 40;
/** (960×540)/(1280×720): the reduced upload's share of the pixel work. */
const PIXEL_SHARE = (960 * 540) / (1280 * 720);
export const SCALE_UP_BELOW_MS = 25 * PIXEL_SHARE;
export const MIN_SAMPLES = 8;
export const MIN_DWELL_MS = 3000;

export class TrackScaleController {
  private readonly mode: TrackScaleMode;
  private reduced: boolean;
  private estimate = new DurationEstimate();
  private samples = 0;
  private changedAt = Number.NEGATIVE_INFINITY;

  constructor(mode: TrackScaleMode) {
    this.mode = mode;
    this.reduced = mode === '960';
  }

  /** The long side to upload tracking frames at, or null for the camera's own size. */
  longSide(): number | null {
    return this.reduced ? REDUCED_LONG_SIDE : null;
  }

  /**
   * One tracking capture's CPU work (draw + encode + base64, ms) at the size `longSide()` gave it. A sample
   * taken at the other size (the decision changed while it was in flight) is ignored.
   */
  observe(workMs: number, atMs: number, reducedSample: boolean): void {
    if (this.mode !== 'auto' || reducedSample !== this.reduced || !Number.isFinite(workMs)) return;
    this.estimate.add(workMs);
    this.samples += 1;
    const value = this.estimate.get();
    if (value === null || this.samples < MIN_SAMPLES || atMs - this.changedAt < MIN_DWELL_MS) return;
    const flip = this.reduced ? value < SCALE_UP_BELOW_MS : value > SCALE_DOWN_ABOVE_MS;
    if (!flip) return;
    this.reduced = !this.reduced;
    this.estimate = new DurationEstimate();
    this.samples = 0;
    this.changedAt = atMs;
  }
}
