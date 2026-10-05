/**
 * The fast loop's hand-off point: the newest accepted tracker answer, published once per frame by
 * `useObjectTrack` right after its `decideUpdate` fence passes, and read outside React by the live overlay.
 *
 * Two read paths over one value:
 * - `subscribe` / `getSnapshot` follow the `useSyncExternalStore` contract (a stable snapshot object per
 *   publish, so React components such as the status chip re-render only when a frame actually lands);
 * - `current()` is the same value as a plain getter for hot paths (a rAF tick reads it without React).
 *
 * The snapshot carries no image. `box` is in normalized 0..1 coordinates of the UPLOADED image, which is
 * already mirrored at capture when the preview is mirrored — the same orientation the user sees — so no
 * reader may mirror it again. `capturedAt` is the wall clock (`Date.now()`) at which that image was frozen;
 * readers must compare it against `Date.now()`, never `performance.now()`.
 *
 * Fail-closed rule (design v4 principle 3): a box is drawable only while `state === 'tracking'` and the
 * frame it was computed for is at most `LIVE_BOX_FRESHNESS_MS` old. There is no dimming or interpolation.
 */
import type { Box } from '@visual-coach/visual-tools';
import type { TrackFrameResponse } from '../generated/api.generated';

/** Past this capture→now age the live box no longer describes the live picture, so it is hidden. */
export const LIVE_BOX_FRESHNESS_MS = 300;

export type LiveTrackState = TrackFrameResponse['state'];

export interface LiveTrackSnapshot {
  runId: string;
  trackId: string;
  generation: number;
  state: LiveTrackState;
  /** Present only when `state === 'tracking'`; normalized 0..1 in the uploaded (already mirrored) image. */
  box: Box | null;
  /** `Date.now()` at the moment the frame this answer describes was captured. */
  capturedAt: number;
  version: number;
  /** The run's target label (an English grounder noun, or the neutral manual-selection token). */
  target: string | null;
}

export interface LiveTrackStore {
  subscribe(listener: () => void): () => void;
  getSnapshot(): LiveTrackSnapshot | null;
  /** Hot-path getter: identical to `getSnapshot`, named for non-React readers. */
  current(): LiveTrackSnapshot | null;
  publish(snapshot: LiveTrackSnapshot): void;
  /** Drop the snapshot (run retired / pair cleared / unmount). A no-op when already empty. */
  clear(): void;
}

export function createLiveTrackStore(): LiveTrackStore {
  let snapshot: LiveTrackSnapshot | null = null;
  const listeners = new Set<() => void>();
  const emit = () => {
    // Copy first: a listener may unsubscribe (or subscribe) while being notified.
    for (const listener of [...listeners]) listener();
  };
  return {
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getSnapshot: () => snapshot,
    current: () => snapshot,
    publish(next) {
      // A box outside `tracking` is never stored: the contract pairs box != null with state == tracking.
      const box = next.state === 'tracking' && next.box ? { ...next.box } : null;
      snapshot = Object.freeze({ ...next, box });
      emit();
    },
    clear() {
      if (snapshot === null) return;
      snapshot = null;
      emit();
    },
  };
}

/** Capture→now age of the snapshot's frame in ms, never negative; null without a snapshot. */
export function liveBoxAgeMs(snapshot: LiveTrackSnapshot | null, nowMs: number): number | null {
  if (!snapshot) return null;
  return Math.max(0, nowMs - snapshot.capturedAt);
}

/**
 * True iff the live box may be drawn right now: `tracking`, a box, and a frame no older than `limitMs`.
 * A capture timestamp in the future (clock skew) is treated as age 0 rather than as stale.
 */
export function isLiveBoxDrawable(
  snapshot: LiveTrackSnapshot | null,
  nowMs: number,
  limitMs: number = LIVE_BOX_FRESHNESS_MS
): boolean {
  if (!snapshot || snapshot.state !== 'tracking' || !snapshot.box) return false;
  const age = liveBoxAgeMs(snapshot, nowMs);
  return age !== null && age <= limitMs;
}
