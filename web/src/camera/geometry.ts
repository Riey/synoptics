/**
 * Pure geometry for placing a live guidance overlay exactly over a video's content rectangle.
 *
 * The overlay must cover the *video content* (the letterbox-fitted frame inside the media box), never
 * the media box itself: the renderer's normalized coordinates are frame pixels, so any difference
 * between the two rectangles silently shifts every drawn command.
 *
 * Nothing here touches the DOM, so the mapping can be reasoned about (and evaluated) independently of
 * the browser layout engine.
 */

export interface Size {
  width: number;
  height: number;
}

export interface ContentRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Letterbox-fitted rect of an intrinsic frame inside a box, centered on both axes.
 *
 * Returns null when either the box or the intrinsic size is unusable, so callers keep the overlay
 * hidden instead of drawing at a guessed scale. The returned aspect ratio equals `intrinsic`, which is
 * what lets the SVG use `preserveAspectRatio="none"` without distorting anything.
 */
export function computeContentRect(box: Size, intrinsic: Size): ContentRect | null {
  const boxUsable =
    Number.isFinite(box.width) && Number.isFinite(box.height) && box.width > 0 && box.height > 0;
  const intrinsicUsable =
    Number.isFinite(intrinsic.width) &&
    Number.isFinite(intrinsic.height) &&
    intrinsic.width > 0 &&
    intrinsic.height > 0;
  if (!boxUsable || !intrinsicUsable) {
    return null;
  }

  const scale = Math.min(box.width / intrinsic.width, box.height / intrinsic.height);
  const width = intrinsic.width * scale;
  const height = intrinsic.height * scale;

  return {
    left: (box.width - width) / 2,
    top: (box.height - height) / 2,
    width,
    height,
  };
}
