/**
 * What the anchored overlay draws, held outside React: the role binding, the current step's bound commands
 * and the verdict warning. The overlay reads it in the same rAF tick as the live track snapshot, so a step
 * change or a rebinding shows up on the next frame without re-rendering the page.
 */
import type { AnchorBinding, BoundCommand } from './anchorProject';

export interface GuideOverlayState {
  binding: AnchorBinding | null;
  commands: BoundCommand[];
  /** The follower said the box does NOT hold the target: keep the box, draw it as a warning. */
  warn: boolean;
  /**
   * Identity of the current action motion (`motionKey`: run, plan revision, step, anchor); a change restarts
   * the motion clock. Null without a step or a binding.
   */
  motionKey: string | null;
}

export const EMPTY_GUIDE_OVERLAY: GuideOverlayState = Object.freeze({
  binding: null,
  commands: [],
  warn: false,
  motionKey: null,
});

export interface GuideOverlayStore {
  current(): GuideOverlayState;
  set(next: GuideOverlayState): void;
  subscribe(listener: () => void): () => void;
}

export function createGuideOverlayStore(): GuideOverlayStore {
  let state: GuideOverlayState = EMPTY_GUIDE_OVERLAY;
  const listeners = new Set<() => void>();
  return {
    current: () => state,
    set(next) {
      state = Object.freeze({ ...next, commands: [...next.commands] });
      for (const listener of [...listeners]) listener();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}
