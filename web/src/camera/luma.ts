/**
 * Frame snapshots and the coarse 48x36 luma grid of a capture.
 *
 * `drawVideoFrame` is the one place a video element is drawn to a canvas (it applies the mirror flip), so
 * every capture and every anchor sample reads the same pixels the user sees. `sampleLumaGrid` reduces a
 * snapshot to the grid that rides along with a capture (`CapturedImage.anchorGrid`).
 *
 * The scene-change comparison that used to live here served the live guidance lane, removed 2026-10-01.
 */

export const LUMA_GRID_WIDTH = 48;
export const LUMA_GRID_HEIGHT = 36;

export interface LumaSampleOptions {
  /**
   * Sample the source as its horizontally flipped self. The live preview can be mirrored, but
   * `drawImage` reads the element's raw pixels and ignores CSS transforms, so a mirrored preview needs
   * an explicit flip here or its grid would describe the opposite orientation from the encoded frame.
   */
  mirror?: boolean;
}

let samplerCanvas: HTMLCanvasElement | null = null;
let samplerContext: CanvasRenderingContext2D | null = null;

/** One ladder step: halve the size, never going below the grid. */
function nextLadderSize(width: number, height: number): { width: number; height: number } {
  return {
    width: Math.max(LUMA_GRID_WIDTH, Math.round(width / 2)),
    height: Math.max(LUMA_GRID_HEIGHT, Math.round(height / 2)),
  };
}

/**
 * Every size the anti-alias ladder passes through on its way from a source to the grid, excluding the
 * source itself. Halving stops as soon as either axis reaches the grid, so the last entry can still be
 * larger than the grid on both axes (e.g. 60x34) — the sampler's final draw then reduces by less than
 * 2x, which is exactly what keeps the whole chain alias-free.
 */
function ladderSizes(width: number, height: number): Array<{ width: number; height: number }> {
  const sizes: Array<{ width: number; height: number }> = [];
  let current = { width, height };
  while (current.width > LUMA_GRID_WIDTH && current.height > LUMA_GRID_HEIGHT) {
    current = nextLadderSize(current.width, current.height);
    sizes.push(current);
  }
  return sizes;
}

/** Intrinsic pixel size of a drawable, without drawing it. */
function drawableSize(source: CanvasImageSource): { width: number; height: number } | null {
  const drawable = source as {
    videoWidth?: number;
    videoHeight?: number;
    naturalWidth?: number;
    naturalHeight?: number;
    width?: number;
    height?: number;
  };
  const pairs: Array<[number | undefined, number | undefined]> = [
    [drawable.videoWidth, drawable.videoHeight],
    [drawable.naturalWidth, drawable.naturalHeight],
    [drawable.width, drawable.height],
  ];
  for (const [width, height] of pairs) {
    if (typeof width === 'number' && width > 0 && typeof height === 'number' && height > 0) {
      return { width, height };
    }
  }
  return null;
}

let ladderContexts: [CanvasRenderingContext2D, CanvasRenderingContext2D] | null = null;

/**
 * A 2D context of either canvas kind. The main thread reduces DOM canvases; the capture worker
 * (`coach/jpeg-encoder.worker.ts`) reduces `OffscreenCanvas`es with the very same ladder, so a grid never
 * depends on which thread produced it.
 */
export type Context2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

/** Two ladder contexts with the smoothing the reduction relies on, set once. */
export function configureLadderContexts(contexts: [Context2D, Context2D]): void {
  for (const context of contexts) {
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'low';
  }
}

/**
 * Sample any drawable source (the frozen canvas a frame was encoded from, or a full-size canvas a video
 * frame was drawn into) into the fixed luma grid. Callers validating a live element are responsible for
 * its readiness; a source that cannot be drawn yields null instead of a black grid.
 *
 * The reduction is a progressive half-size ladder, NOT one step. One `drawImage(source, 0, 0, 48, 36)`
 * from a 1280x720 frame is a ~26.7x reduction, and with the context default (`imageSmoothingQuality`
 * 'low') that is close to a point sample: a *stationary* textured scene then reads as a scene change
 * when the frame moves by a fraction of a cell. Measured in Chromium on a 1280x720 textured scene with
 * these very thresholds, one step reported `patch` 0.0578 at 1px of drift, 0.0663 at 2px, 0.0586 at a
 * 1.002 rescale and 0.0622 at 1.005, i.e. above LOCAL_PATCH_THRESHOLD on a still scene, while the same
 * root replay through this ladder reported 0.0023 / 0.0113 / 0.0050 / 0.0103 and still raised
 * `structure` 0.0709 for a real 5% pan, 0.1587 patch for a small real object and 0.0875 for a hand
 * passing through. Each step's reduction is at most 2x, so even with a filter that ignores the quality
 * hint every output pixel averages its full 2x2 source footprint and the ladder as a whole behaves as
 * an area average. It also costs nothing per call: the scratch canvases are reused and only the final
 * 48x36 grid is ever read back (1728 pixels, no full-frame `getImageData`).
 *
 * Callers must pass a *canvas* snapshot rather than a live `<video>`: Chromium resamples a video element
 * into a small canvas with a different filter than it resamples a canvas, which by itself reads as a
 * permanent scene change (see `drawVideoFrame`). Both call sites already do.
 */
export function sampleLumaGrid(source: CanvasImageSource): Float32Array | null {
  const size = drawableSize(source);
  if (!size) return null;
  if (!samplerContext) {
    samplerCanvas = document.createElement('canvas');
    samplerCanvas.width = LUMA_GRID_WIDTH;
    samplerCanvas.height = LUMA_GRID_HEIGHT;
    samplerContext = samplerCanvas.getContext('2d', { willReadFrequently: true });
    if (!samplerContext) return null;
  }
  if (!ladderContexts) {
    const firstContext = document.createElement('canvas').getContext('2d');
    const secondContext = document.createElement('canvas').getContext('2d');
    if (!firstContext || !secondContext) return null;
    ladderContexts = [firstContext, secondContext];
    configureLadderContexts(ladderContexts);
  }
  return reduceToLumaGrid(source, size, ladderContexts, samplerContext);
}

/**
 * The ladder itself, on contexts the caller owns: `ladder` are two scratch contexts configured with
 * `configureLadderContexts`, `sampler` is a `LUMA_GRID_WIDTH`x`LUMA_GRID_HEIGHT` context created with
 * `willReadFrequently` (the only one ever read back). Shared by `sampleLumaGrid` and the capture worker.
 */
export function reduceToLumaGrid(
  source: CanvasImageSource,
  size: { width: number; height: number },
  ladder: [Context2D, Context2D],
  sampler: Context2D
): Float32Array | null {
  try {
    let current: CanvasImageSource = source;
    let from = size;
    let index = 0;
    for (const next of ladderSizes(size.width, size.height)) {
      const context = ladder[index];
      const canvas = context.canvas;
      if (canvas.width !== next.width || canvas.height !== next.height) {
        canvas.width = next.width;
        canvas.height = next.height;
      }
      // The transform is reset on every call — never inherited — because the ladder canvases are shared
      // by every caller and a failed draw must not be able to poison the next sample.
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.drawImage(current, 0, 0, from.width, from.height, 0, 0, next.width, next.height);
      current = canvas;
      from = next;
      index = 1 - index;
    }
    sampler.setTransform(1, 0, 0, 1, 0, 0);
    sampler.drawImage(current, 0, 0, from.width, from.height, 0, 0, LUMA_GRID_WIDTH, LUMA_GRID_HEIGHT);
    const { data } = sampler.getImageData(0, 0, LUMA_GRID_WIDTH, LUMA_GRID_HEIGHT);
    const grid = new Float32Array(LUMA_GRID_WIDTH * LUMA_GRID_HEIGHT);
    for (let i = 0; i < grid.length; i += 1) {
      const offset = i * 4;
      grid[i] = (0.299 * data[offset] + 0.587 * data[offset + 1] + 0.114 * data[offset + 2]) / 255;
    }
    return grid;
  } catch {
    // A source can become undrawable mid-flight (track ended, element resized); treat it as "no sample".
    return null;
  } finally {
    sampler.setTransform(1, 0, 0, 1, 0, 0);
  }
}

let videoCanvas: HTMLCanvasElement | null = null;
let videoContext: CanvasRenderingContext2D | null = null;

/**
 * Read one live video frame into a full-size canvas, exactly once.
 *
 * This is the ONLY place a video element is drawn. Everything that needs a frame's pixels — the live
 * change samples and the frame that is uploaded and measured — starts from this snapshot, because a
 * video element can advance between two `drawImage` calls: reading it twice would silently pair one
 * frame's JPEG with another frame's grid. The read is synchronous from `drawImage` through the caller's
 * `getImageData`, so the shared canvas cannot be reused halfway through a read.
 *
 * Drawing through a full-size canvas is also what makes two grids comparable at all. Chromium resamples
 * a video element into a small canvas with a different filter than it resamples a canvas into the same
 * small canvas, and the difference is not uniform: measured on a high-contrast synthetic scene it
 * reaches a patch score of 0.061, above this module's own change threshold, so comparing a
 * directly-sampled live frame against a canvas-sampled anchor reports "the scene changed" forever.
 * Canvas→canvas sampling measured exactly 0 in the same experiment.
 *
 * `mirror` is applied here and only here: callers always reduce the returned canvas to the grid
 * unflipped, so a mirrored frame can never be flipped twice.
 */
export function drawVideoFrame(
  video: HTMLVideoElement,
  options: LumaSampleOptions = {}
): HTMLCanvasElement | null {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (width === 0 || height === 0) return null;

  if (!videoContext) {
    videoCanvas = document.createElement('canvas');
    videoContext = videoCanvas.getContext('2d', { willReadFrequently: true });
    if (!videoContext) return null;
  }
  if (videoCanvas!.width !== width || videoCanvas!.height !== height) {
    videoCanvas!.width = width;
    videoCanvas!.height = height;
  }

  try {
    try {
      videoContext.setTransform(options.mirror ? -1 : 1, 0, 0, 1, options.mirror ? width : 0, 0);
      videoContext.drawImage(video, 0, 0, width, height);
    } finally {
      // Restored even when the draw threw, so a failed mirrored read cannot poison the next one.
      videoContext.setTransform(1, 0, 0, 1, 0, 0);
    }
  } catch {
    return null;
  }
  return videoCanvas;
}
