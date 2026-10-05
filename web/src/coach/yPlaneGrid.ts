/**
 * The appearance grid straight from a camera frame's luma (Y) plane: an exact area average of the crop's Y
 * samples into grid×grid cells, mirrored by reversing columns. The v2 capture worker uses it for stream
 * frames (`VideoFrame.copyTo` of the crop only), so an appearance sample never converts the whole frame to
 * RGB and never touches a canvas.
 *
 * Scale: limited-range Y (16..235) is mapped to 0..1 like the canvas path's BT.601 luma of the converted
 * RGB, so the measured `APPEARANCE_THRESHOLD` keeps its meaning (the two methods are compared in headless
 * Chromium by `capture-lab.html`; numbers in the commit/report). Every reference and every sample of one v2
 * page that comes from the camera stream uses this method.
 *
 * Pure, so `node --test` runs it.
 */
export interface PlaneRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Grow `rect` outward to even coordinates and sizes (4:2:0 chroma alignment, which `VideoFrame.copyTo`
 * requires for a sub-rectangle), clamped to the frame. Null when nothing is left.
 */
export function evenRect(rect: PlaneRect, frame: { width: number; height: number }): PlaneRect | null {
  const x0 = Math.max(0, Math.floor(rect.x / 2) * 2);
  const y0 = Math.max(0, Math.floor(rect.y / 2) * 2);
  const x1 = Math.min(frame.width - (frame.width % 2), Math.ceil((rect.x + rect.width) / 2) * 2);
  const y1 = Math.min(frame.height - (frame.height % 2), Math.ceil((rect.y + rect.height) / 2) * 2);
  if (x1 - x0 < 2 || y1 - y0 < 2) return null;
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

/**
 * Area-average a Y plane (`width`×`height` samples, `stride` bytes per row) into a grid×grid luma grid in
 * 0..1. `fullRange` false maps 16..235 to 0..1 (clamped). `mirror` reverses the columns (the crop as it
 * appears in the mirrored frame).
 */
export function yPlaneToGrid(
  plane: Uint8Array,
  width: number,
  height: number,
  stride: number,
  grid: number,
  fullRange: boolean,
  mirror: boolean
): Float32Array {
  const out = new Float32Array(grid * grid);
  const colStart = new Int32Array(grid + 1);
  for (let c = 0; c <= grid; c += 1) colStart[c] = Math.floor((c * width) / grid);
  const sums = new Float64Array(grid);
  for (let r = 0; r < grid; r += 1) {
    const rowFrom = Math.floor((r * height) / grid);
    const rowTo = Math.max(rowFrom + 1, Math.floor(((r + 1) * height) / grid));
    sums.fill(0);
    for (let y = rowFrom; y < rowTo; y += 1) {
      const base = y * stride;
      for (let c = 0; c < grid; c += 1) {
        const to = Math.max(colStart[c] + 1, colStart[c + 1]);
        let s = 0;
        for (let x = colStart[c]; x < to; x += 1) s += plane[base + x];
        sums[c] += s;
      }
    }
    const rows = rowTo - rowFrom;
    for (let c = 0; c < grid; c += 1) {
      const count = rows * Math.max(1, colStart[c + 1] - colStart[c]);
      const mean = sums[c] / count;
      const value = fullRange ? mean / 255 : (mean - 16) / 219;
      out[r * grid + (mirror ? grid - 1 - c : c)] = Math.min(1, Math.max(0, value));
    }
  }
  return out;
}

const PLANAR = new Set(['I420', 'I420A', 'NV12']);

/**
 * The appearance grid of `crop` (in the frame's displayed pixels, unmirrored) from a `VideoFrame`'s Y plane:
 * only the crop is copied (`copyTo` with a rect). Null when the frame has no readable planar layout (a
 * texture-backed or RGB frame, or non-square pixels) — the caller then uses the canvas ladder.
 */
export async function frameYGrid(frame: VideoFrame, crop: PlaneRect, grid: number, mirror: boolean): Promise<Float32Array | null> {
  const visible = frame.visibleRect;
  if (!frame.format || !PLANAR.has(frame.format) || !visible) return null;
  if (visible.width !== frame.displayWidth || visible.height !== frame.displayHeight) return null;
  const rect = evenRect(crop, { width: visible.width, height: visible.height });
  if (!rect) return null;
  const copyRect = { x: visible.x + rect.x, y: visible.y + rect.y, width: rect.width, height: rect.height };
  try {
    const buffer = new Uint8Array(frame.allocationSize({ rect: copyRect }));
    const [luma] = await frame.copyTo(buffer, { rect: copyRect });
    return yPlaneToGrid(buffer.subarray(luma.offset), rect.width, rect.height, luma.stride, grid, frame.colorSpace?.fullRange === true, mirror);
  } catch {
    return null;
  }
}
