/**
 * Manual target selection: the user drags a rectangle over the paired captured still.
 *
 * The still is the already-mirrored encoded frame, and the user draws on that same displayed image,
 * so a pointer position maps straight onto the uploaded image's coordinates — a plain scale by the
 * element rect, with NO horizontal inversion. (The mirror lives in the pixels, once, in
 * `captureVideoFrame`; inverting again here would select the opposite side.)
 *
 * The emitted box is normalized 0..1 in the displayed = uploaded image, exactly what `seed_box`
 * accepts.
 */
import { useRef, useState } from 'react';
import type { Box } from '@visual-coach/visual-tools';

interface LocalRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Reject a stray click/tap that would make a degenerate seed box. */
const MIN_DRAG_PX = 12;

export function TargetSelectOverlay({
  onSelect,
  onCancel,
}: {
  onSelect: (box: Box) => void;
  onCancel: () => void;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const rectRef = useRef<DOMRect | null>(null);
  const [drag, setDrag] = useState<LocalRect | null>(null);

  const localPoint = (event: React.PointerEvent<HTMLDivElement>) => {
    const rect = rectRef.current;
    if (!rect) return { x: 0, y: 0 };
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const host = hostRef.current;
    if (!host || event.button !== 0) return;
    rectRef.current = host.getBoundingClientRect();
    const point = localPoint(event);
    host.setPointerCapture(event.pointerId);
    setDrag({ x0: point.x, y0: point.y, x1: point.x, y1: point.y });
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!drag) return;
    const point = localPoint(event);
    setDrag({ ...drag, x1: point.x, y1: point.y });
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const rect = rectRef.current;
    if (!drag || !rect) return;
    const point = localPoint(event);
    setDrag(null);
    const left = Math.max(0, Math.min(drag.x0, point.x));
    const top = Math.max(0, Math.min(drag.y0, point.y));
    const right = Math.min(rect.width, Math.max(drag.x0, point.x));
    const bottom = Math.min(rect.height, Math.max(drag.y0, point.y));
    if (right - left < MIN_DRAG_PX || bottom - top < MIN_DRAG_PX) {
      onCancel();
      return;
    }
    onSelect({
      x: left / rect.width,
      y: top / rect.height,
      width: (right - left) / rect.width,
      height: (bottom - top) / rect.height,
    });
  };

  const preview = drag
    ? {
        left: Math.min(drag.x0, drag.x1),
        top: Math.min(drag.y0, drag.y1),
        width: Math.abs(drag.x1 - drag.x0),
        height: Math.abs(drag.y1 - drag.y0),
      }
    : null;

  return (
    <div
      ref={hostRef}
      className="target-select"
      data-testid="target-select"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={() => {
        setDrag(null);
        onCancel();
      }}
    >
      {preview && (
        <div
          className="target-select-marquee"
          style={{ left: preview.left, top: preview.top, width: preview.width, height: preview.height }}
        />
      )}
      {!preview && <p className="target-select-hint">화면에서 대상을 네모로 그려주세요</p>}
    </div>
  );
}
