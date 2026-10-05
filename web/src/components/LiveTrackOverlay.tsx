/**
 * The live track box over the playing video, plus a minimal tracking status chip.
 *
 * `LiveTrackOverlay` is mounted inside CameraStage's overlay div, which is already placed on the video's
 * letterboxed content rect, so a `viewBox="0 0 1 1"` + `preserveAspectRatio="none"` svg maps normalized
 * box coordinates straight onto the painted video pixels. It never re-renders per frame: it subscribes to
 * the `liveTrack` store and, in one requestAnimationFrame per published frame, mutates only the rect
 * attributes and the svg's visibility. While a box is shown, every rAF tick re-checks freshness, so a box
 * disappears once its frame is older than `LIVE_BOX_FRESHNESS_MS` even when no new frame arrives.
 *
 * Mirror: the box is in the uploaded image's space, and the upload is already mirrored at capture when the
 * preview is mirrored — that is exactly the orientation the user sees. The overlay is never flipped again.
 */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

import {
  isLiveBoxDrawable,
  liveBoxAgeMs,
  type LiveTrackSnapshot,
  type LiveTrackStore,
} from '../tracking/liveTrackStore';

/** React read of the live snapshot (re-renders only the reading component, once per published frame). */
export function useLiveTrackSnapshot(store: LiveTrackStore): LiveTrackSnapshot | null {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
}

export function LiveTrackOverlay({ store }: { store: LiveTrackStore }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const outerRef = useRef<SVGRectElement>(null);
  const innerRef = useRef<SVGRectElement>(null);

  useEffect(() => {
    const svg = svgRef.current;
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!svg || !outer || !inner) return undefined;

    let raf = 0;
    let disposed = false;
    /** The snapshot whose box the rects currently hold, so an unchanged box is not rewritten. */
    let placed: LiveTrackSnapshot | null = null;

    const setAttr = (name: string, value: string | null) => {
      if (value === null) {
        if (svg.hasAttribute(name)) svg.removeAttribute(name);
      } else if (svg.getAttribute(name) !== value) {
        svg.setAttribute(name, value);
      }
    };

    const tick = () => {
      raf = 0;
      if (disposed) return;
      const snapshot = store.current();
      const now = Date.now();
      const drawable = isLiveBoxDrawable(snapshot, now);
      if (drawable && snapshot?.box && snapshot !== placed) {
        const { x, y, width, height } = snapshot.box;
        for (const rect of [outer, inner]) {
          rect.setAttribute('x', String(x));
          rect.setAttribute('y', String(y));
          rect.setAttribute('width', String(width));
          rect.setAttribute('height', String(height));
        }
        placed = snapshot;
      }
      const visibility = drawable ? 'visible' : 'hidden';
      if (svg.style.visibility !== visibility) svg.style.visibility = visibility;
      setAttr('data-live-visible', drawable ? 'true' : 'false');
      setAttr('data-live-state', snapshot?.state ?? 'none');
      const age = liveBoxAgeMs(snapshot, now);
      setAttr('data-live-age-ms', age === null || !Number.isFinite(age) ? null : String(Math.round(age)));
      // Keep re-checking while something is drawn: the freshness limit must hide a box with no new frame.
      if (drawable) schedule();
    };

    function schedule() {
      if (raf !== 0 || disposed) return;
      raf = window.requestAnimationFrame(tick);
    }

    const unsubscribe = store.subscribe(schedule);
    schedule();
    return () => {
      disposed = true;
      unsubscribe();
      if (raf !== 0) window.cancelAnimationFrame(raf);
      raf = 0;
      svg.style.visibility = 'hidden';
    };
  }, [store]);

  return (
    <svg
      ref={svgRef}
      className="live-track-overlay"
      data-testid="live-track-box"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <rect ref={outerRef} className="track-box-outer" vectorEffect="non-scaling-stroke" />
      <rect ref={innerRef} className="track-box-inner" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** The hook's run phase, used only while no tracker answer has been accepted yet. */
export type LiveTrackPhase = 'idle' | 'analyzing' | 'starting' | 'running' | 'error';

const LIVE_STATE_LABEL: Record<LiveTrackSnapshot['state'], string> = {
  acquiring: '획득 중',
  tracking: '추적 중',
  occluded: '가림',
  lost: '놓침',
  unavailable: '추적기 사용 불가',
};

const PHASE_LABEL: Record<LiveTrackPhase, string> = {
  idle: '추적 꺼짐',
  analyzing: '장면 분석 중',
  starting: '추적 시작 중',
  running: '획득 중',
  error: '추적 오류',
};

const CHIP_TICK_MS = 100;

/** Age as the chip prints it: whole ms under a second, then tenths of a second. */
export function formatLiveAge(ageMs: number): string {
  return ageMs < 1000 ? `${Math.round(ageMs)}ms` : `${(ageMs / 1000).toFixed(1)}초`;
}

/**
 * A minimal, always-visible tracking status over the video: state and the shown frame's capture age.
 * It is not the narration bar — no instruction text lives here.
 */
export function LiveTrackChip({ store, phase }: { store: LiveTrackStore; phase: LiveTrackPhase }) {
  const snapshot = useLiveTrackSnapshot(store);
  // Re-render for the age readout while frames may have stopped arriving; publishes re-render anyway.
  // 100 ms keeps the chip's "hidden as stale" note within one tick of the overlay's own 300 ms cut.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!snapshot) return undefined;
    const timer = window.setInterval(() => setTick((n) => n + 1), CHIP_TICK_MS);
    return () => window.clearInterval(timer);
  }, [snapshot]);

  const now = Date.now();
  const drawable = isLiveBoxDrawable(snapshot, now);
  const age = liveBoxAgeMs(snapshot, now);
  // An analysis or a run start in progress outranks a snapshot kept from a finished run.
  const busyPhase = phase === 'analyzing' || phase === 'starting';
  const label = snapshot && !busyPhase ? LIVE_STATE_LABEL[snapshot.state] ?? snapshot.state : PHASE_LABEL[phase];
  const stateKey = snapshot && !busyPhase ? snapshot.state : `phase-${phase}`;
  const parts = [label];
  if (snapshot && !busyPhase && age !== null && Number.isFinite(age)) parts.push(`${formatLiveAge(age)} 전 프레임`);
  if (snapshot && !busyPhase && snapshot.state === 'tracking' && !drawable) parts.push('오래되어 상자 숨김');
  const tone = drawable ? 'ok' : snapshot && !busyPhase && snapshot.state !== 'acquiring' ? 'warn' : 'neutral';

  return (
    <div
      className={`camera-stage-track-chip camera-stage-track-chip-${tone}`}
      data-testid="live-track-chip"
      data-live-state={stateKey}
      data-live-age-ms={snapshot && age !== null && Number.isFinite(age) ? Math.round(age) : undefined}
    >
      {parts.join(' · ')}
    </div>
  );
}
