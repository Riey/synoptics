/**
 * The DOM half of `target_changed`: crop a frame to the (padded) tracked box and reduce it to the
 * appearance grid with an area-averaging canvas ladder (each draw reduces by at most 2× per axis, the way
 * `camera/luma.ts` reduces whole frames). Only the final grid×grid pixels are read back.
 *
 * Live frames go through `drawVideoFrame` (the one place a video element is drawn; it also applies the
 * mirror once, so the crop is taken in the same mirrored space the tracker's boxes are in).
 */
import type { Box } from '@visual-coach/visual-tools';

import { configureLadderContexts, drawVideoFrame, type Context2D } from '../camera/luma';
import {
  DEFAULT_APPEARANCE_CONFIG,
  appearanceCrop,
  ladderSizes,
  lumaFromRgba,
  type AppearanceConfig,
  type PixelRect,
} from './anchorAppearance';

let ladderContexts: [CanvasRenderingContext2D, CanvasRenderingContext2D] | null = null;
let gridContext: CanvasRenderingContext2D | null = null;

function ensureContexts(): boolean {
  if (!ladderContexts) {
    const ca = document.createElement('canvas').getContext('2d');
    const cb = document.createElement('canvas').getContext('2d');
    if (!ca || !cb) return false;
    ladderContexts = [ca, cb];
    configureLadderContexts(ladderContexts);
  }
  if (!gridContext) {
    gridContext = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
    if (!gridContext) return false;
    gridContext.imageSmoothingEnabled = true;
    gridContext.imageSmoothingQuality = 'low';
  }
  return true;
}

/**
 * Grid of `source` (a drawable of `size` pixels) inside `box` grown by `config.pad`. Null when the box is
 * degenerate or the source cannot be drawn.
 */
export function sampleAnchorGrid(
  source: CanvasImageSource,
  size: { width: number; height: number },
  box: Box,
  config: AppearanceConfig = DEFAULT_APPEARANCE_CONFIG
): Float32Array | null {
  if (!ensureContexts()) return null;
  return reduceAnchorGrid(source, size, box, config, ladderContexts!, gridContext!);
}

/**
 * The crop-and-ladder on contexts the caller owns: `ladder` configured with `configureLadderContexts`,
 * `grid` created with `willReadFrequently` and low-quality smoothing (the only one read back). Shared by
 * `sampleAnchorGrid` and the capture workers, so a grid never depends on which thread produced it.
 */
export function reduceAnchorGrid(
  source: CanvasImageSource,
  size: { width: number; height: number },
  box: Box,
  config: AppearanceConfig,
  ladder: [Context2D, Context2D],
  grid: Context2D
): Float32Array | null {
  const crop = appearanceCrop(box, size, config.pad);
  if (!crop) return null;
  return reduceCropToGrid(source, crop, config, ladder, grid, false);
}

/**
 * The ladder from one source rectangle (`crop`, in the SOURCE's own pixels) to the grid. With `mirror` the
 * grid is that of the horizontally flipped crop: the flip rides on the first draw (a mirrored, ≤2× reducing
 * draw), so an unmirrored camera frame never has to be copied whole just to be flipped. Mirroring the whole
 * frame 1:1 first and cropping the mirrored rectangle (`reduceAnchorGrid` on a mirrored snapshot) reads the
 * very same source pixels into the very same cells.
 */
export function reduceCropToGrid(
  source: CanvasImageSource,
  crop: PixelRect,
  config: AppearanceConfig,
  ladder: [Context2D, Context2D],
  grid: Context2D,
  mirror: boolean
): Float32Array | null {
  if (grid.canvas.width !== config.grid || grid.canvas.height !== config.grid) {
    grid.canvas.width = config.grid;
    grid.canvas.height = config.grid;
    // Resizing a canvas resets its context state, the smoothing settings included.
    grid.imageSmoothingEnabled = true;
    grid.imageSmoothingQuality = 'low';
  }
  try {
    let current: CanvasImageSource = source;
    let from = crop;
    let index = 0;
    let flip = mirror;
    for (const next of ladderSizes(crop.width, crop.height, config.grid)) {
      const context = ladder[index];
      const canvas = context.canvas;
      if (canvas.width !== next.width || canvas.height !== next.height) {
        canvas.width = next.width;
        canvas.height = next.height;
      }
      context.setTransform(flip ? -1 : 1, 0, 0, 1, flip ? next.width : 0, 0);
      context.drawImage(current, from.x, from.y, from.width, from.height, 0, 0, next.width, next.height);
      context.setTransform(1, 0, 0, 1, 0, 0);
      flip = false;
      current = canvas;
      from = { x: 0, y: 0, width: next.width, height: next.height };
      index = 1 - index;
    }
    grid.setTransform(flip ? -1 : 1, 0, 0, 1, flip ? config.grid : 0, 0);
    grid.drawImage(current, from.x, from.y, from.width, from.height, 0, 0, config.grid, config.grid);
    grid.setTransform(1, 0, 0, 1, 0, 0);
    const { data } = grid.getImageData(0, 0, config.grid, config.grid);
    return lumaFromRgba(data, config.grid * config.grid);
  } catch {
    return null;
  } finally {
    // A failed draw must not leave a flip behind for the next sample on these shared contexts.
    for (const context of [...ladder, grid]) context.setTransform(1, 0, 0, 1, 0, 0);
  }
}

/**
 * The crop of a normalized box (in the DISPLAYED, possibly mirrored frame) in the raw source's pixels: the
 * same `appearanceCrop` rectangle, reflected across the vertical centre line when the display is mirrored.
 */
export function rawAppearanceCrop(
  box: Box,
  size: { width: number; height: number },
  config: AppearanceConfig,
  mirror: boolean
): PixelRect | null {
  const crop = appearanceCrop(box, size, config.pad);
  if (!crop || !mirror) return crop;
  return { ...crop, x: size.width - crop.x - crop.width };
}

/** One live sample of the playing video inside `box` (mirror applied once, as for the upload). */
export function sampleVideoAnchorGrid(
  video: HTMLVideoElement,
  box: Box,
  mirror: boolean,
  config: AppearanceConfig = DEFAULT_APPEARANCE_CONFIG
): Float32Array | null {
  const frame = drawVideoFrame(video, { mirror });
  if (!frame) return null;
  return sampleAnchorGrid(frame, { width: frame.width, height: frame.height }, box, config);
}
