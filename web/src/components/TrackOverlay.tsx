/**
 * The track box, drawn over the paired captured still.
 *
 * It takes normalized 0..1 coordinates and a `viewBox` of `0 0 1 1` with `preserveAspectRatio="none"`,
 * so the caller only has to make the wrapper exactly the still's aspect ratio: there is no letterbox
 * math here and, because the still is the already-mirrored encoded frame and so is the drawn image,
 * there is no mirror transform either. `vector-effect="non-scaling-stroke"` keeps the stroke constant
 * regardless of the viewBox scale.
 *
 * Only ever mounted with a non-null box: `state != "tracking"` draws nothing (v4 invariant 1).
 */
import type { Box } from '@visual-coach/visual-tools';

export function TrackOverlay({ box }: { box: Box }) {
  return (
    <svg
      className="track-overlay"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      aria-hidden="true"
      data-testid="track-overlay"
    >
      <rect
        x={box.x}
        y={box.y}
        width={box.width}
        height={box.height}
        className="track-box-outer"
        vectorEffect="non-scaling-stroke"
      />
      <rect
        x={box.x}
        y={box.y}
        width={box.width}
        height={box.height}
        className="track-box-inner"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
