/**
 * The plan step's commands drawn ON the tracked object, every animation frame, outside React.
 *
 * Mounted next to `LiveTrackOverlay` inside CameraStage's content-rect overlay div. Per rAF it reads the
 * live track snapshot and the guide overlay store, and draws only while the live box is drawable
 * (`isLiveBoxDrawable`: tracking, a box, a frame ≤ 300 ms old) AND the snapshot is the anchor the role was
 * bound to (same run, track and generation). Otherwise everything is hidden — never dimmed, never
 * interpolated (design v4 principle 3).
 *
 * - focus: the box grown by `pad` (`focusRing`, normalized, clamped to the frame), drawn in CSS px as four
 *   rounded corner brackets (`ringPath`): a 7 px halo path under a 3 px accent path. A step with an `action`
 *   but no `focus` gets the default ring (`resolveFocus`); root `data-focus` = command | implied | none.
 * - cue colour: once per `motionKey` run, on its first drawn frame, the stage video is drawn into a 96 px wide
 *   canvas and the mean colours of a band around the box (background) and of the box's inner 60 % (object)
 *   pick the accent (`pickContrastAccent`, preferred sky `#38bdf8`); the accent and its tone are exposed as
 *   `--cue-accent`, `--cue-halo`, `--cue-glow`, `--cue-bubble-bg`, `--cue-bubble-fg` on the root (and
 *   `data-cue-accent`) and locked until the key changes. No video, no frame yet, a tainted canvas or any error
 *   → the preferred accent, also locked for the run.
 * - action motion: box-relative animated primitives (`actionMotion.ts`) in the CSS-px svg, assigned each frame
 *   to a fixed pool of svg nodes (no node is created per frame). Its clock advances only on frames that draw
 *   it (capped per frame), so it pauses while the box is hidden and resumes where it was; a new `motionKey`
 *   (run, plan revision, step, anchor) restarts it. Reduced motion shows the end pose only.
 * - label: an absolutely positioned HTML element next to the box in CSS px of the overlay (`placeLabel`):
 *   above it, or below/beside it when the top is taken by the frame edge or by the narration band, the
 *   camera status pill or the live-track chip (their rects are read each frame). It avoids the box (or the
 *   focus ring) and, when there is room, the motion's extent; the leader line points at the ring/box. Text is
 *   positioned, never scaled.
 * - warn: the follower said the box does not hold the target → the box is outlined in the warning style
 *   (an orange dashed box in the 0..1 svg, independent of the cue colour; kept, never dropped) and the label
 *   says so. No motion.
 *
 * The overlay is never mirrored: box coordinates are already in the mirrored upload's space.
 */
import { useEffect, useRef } from 'react';
import type { Box } from '@visual-coach/visual-tools';
import './AnchoredGuidanceOverlay.css';

import { isLiveBoxDrawable, type LiveTrackStore } from '../tracking/liveTrackStore';
import {
  actionShortLabel,
  computeLeaderLine,
  focusRing,
  placeLabel,
  resolveFocus,
  snapshotMatchesBinding,
  LABEL_OFFSET_PX,
  type PlacedLabel,
  type PxRect,
} from '../intent/anchorProject';
import {
  MOTION_MAX_TICK_MS,
  MOTION_POOL,
  cachedMotionExtent,
  isSchematicAction,
  motionFrame,
  motionShapes,
  ringPath,
  type MotionShape,
} from '../intent/actionMotion';
import {
  PREFERRED_ACCENT,
  accentTone,
  mirrorUnitBox,
  pickContrastAccent,
  sampleSize,
  sceneColorsFromPixels,
  type CueTone,
  type SceneColors,
} from '../intent/cueColor';
import type { GuideOverlayStore } from '../intent/guideOverlayStore';

const WARN_LABEL = '대상이 맞지 않아 보임 · 대상 직접 지정';

/** On-video elements the label must not cover (looked up in the camera stage each frame; absent ones skipped). */
const LABEL_OBSTACLES =
  '[data-testid="narration-bar"], [data-testid="camera-status"], [data-testid="live-track-chip"], [data-testid="capture-advice"]';
/** The obstacles' rects in CSS px relative to `root` (visible, non-empty elements only). */
function obstacleRects(root: HTMLElement): PxRect[] {
  const stage = root.closest('.camera-stage-media') ?? root.parentElement;
  if (!stage) return [];
  const origin = root.getBoundingClientRect();
  const out: PxRect[] = [];
  for (const element of stage.querySelectorAll<HTMLElement>(LABEL_OBSTACLES)) {
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;
    if (getComputedStyle(element).visibility === 'hidden') continue;
    out.push({ left: rect.left - origin.left, top: rect.top - origin.top, width: rect.width, height: rect.height });
  }
  return out;
}

function unionBox(a: Box, b: Box): Box {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return {
    x,
    y,
    width: Math.max(a.x + a.width, b.x + b.width) - x,
    height: Math.max(a.y + a.height, b.y + b.height) - y,
  };
}

/** The user's reduced-motion media query; null where `matchMedia` is unavailable. */
function prefersReducedMotionQuery(): MediaQueryList | null {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return null;
  return window.matchMedia('(prefers-reduced-motion: reduce)');
}

const fmt = (value: number) => String(Math.round(value * 100) / 100);

/** Ghost rects of the motion pool get rounded corners (designer look). */
const MOTION_RECT_RADIUS_PX = 8;

/** The cue colour custom properties on the overlay root, and the tone field each one comes from. */
const CUE_PROPERTIES: ReadonlyArray<[string, keyof CueTone]> = [
  ['--cue-accent', 'accent'],
  ['--cue-halo', 'halo'],
  ['--cue-glow', 'glow'],
  ['--cue-bubble-bg', 'bubbleBg'],
  ['--cue-bubble-fg', 'bubbleFg'],
];

/**
 * Background/object mean colours of the stage video around `box` (normalized to the video frame), from one
 * 96 px wide draw into `canvas`. Null without a playing video with a frame; throws on a tainted canvas.
 */
function sampleStageVideo(root: HTMLElement, box: Box, canvas: HTMLCanvasElement): SceneColors | null {
  const stage = root.closest('.camera-stage-media') ?? root.parentElement;
  const video = stage?.querySelector('video');
  if (!video || video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) return null;
  const { width, height } = sampleSize(video.videoWidth, video.videoHeight);
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  // The video element is shown flipped when mirrored, but drawImage reads the unflipped frame; the box is in
  // the shown (mirrored) orientation.
  const sampled = video.dataset.mirror === 'on' ? mirrorUnitBox(box) : box;
  return sceneColorsFromPixels(pixels, width, height, sampled);
}

export function AnchoredGuidanceOverlay({
  liveTrack,
  guide,
  visible = true,
  onInstructionVisibility,
  reducedMotion,
  sceneColors,
}: {
  liveTrack: LiveTrackStore;
  guide: GuideOverlayStore;
  /** “시각화 켬/끔”: false hides every anchored drawing (the narration stays). */
  visible?: boolean;
  /** Notified only on boolean changes when an action or label is actively placed and drawn (true), or hidden/warned/unbound (false). */
  onInstructionVisibility?: (visible: boolean) => void;
  /** Overrides the `prefers-reduced-motion` media query (the motion preview page uses it). */
  reducedMotion?: boolean;
  /**
   * Replaces the stage-video sample for the cue colour (the motion preview has no camera): the scene's
   * background/object colours around the normalized `box`, or null for the preferred accent.
   */
  sceneColors?: (box: Box) => SceneColors | null;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const ringSvgRef = useRef<SVGSVGElement>(null);
  const ringHaloRef = useRef<SVGPathElement>(null);
  const ringRef = useRef<SVGPathElement>(null);
  const warnRef = useRef<SVGRectElement>(null);
  const pixelSvgRef = useRef<SVGSVGElement>(null);
  const motionGroupRef = useRef<SVGGElement>(null);
  const leaderLineRef = useRef<SVGPathElement>(null);
  const leaderDotRef = useRef<SVGCircleElement>(null);
  const calloutRef = useRef<HTMLDivElement>(null);
  const labelTextRef = useRef<HTMLSpanElement>(null);
  const visibleRef = useRef(visible);
  visibleRef.current = visible;
  const onInstructionVisibilityRef = useRef(onInstructionVisibility);
  onInstructionVisibilityRef.current = onInstructionVisibility;
  const lastInstructionVisibleRef = useRef<boolean | null>(null);
  const scheduleRef = useRef<() => void>(() => {});
  // Motion clock: drawn time since the current motionKey started (pauses while the motion is not drawn).
  const motionKeyRef = useRef<string | null>(null);
  const elapsedRef = useRef(0);
  const lastTickRef = useRef<number | null>(null);
  /** The callout's last placement: kept while it still fits (no side-to-side jumps between frames). */
  const placementRef = useRef<PlacedLabel['placement'] | null>(null);
  const reducedMotionRef = useRef(reducedMotion ?? prefersReducedMotionQuery()?.matches ?? false);
  const sceneColorsRef = useRef(sceneColors);
  sceneColorsRef.current = sceneColors;
  /** The run's cue colour is chosen (once per motionKey); false until its first drawn frame. */
  const cueLockedRef = useRef(false);

  // Reduced motion: the prop wins; otherwise follow the media query, re-read on change.
  useEffect(() => {
    if (reducedMotion !== undefined) {
      reducedMotionRef.current = reducedMotion;
      scheduleRef.current();
      return undefined;
    }
    const query = prefersReducedMotionQuery();
    reducedMotionRef.current = query?.matches ?? false;
    scheduleRef.current();
    if (!query) return undefined;
    const onChange = () => {
      reducedMotionRef.current = query.matches;
      scheduleRef.current();
    };
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, [reducedMotion]);

  useEffect(() => {
    const root = rootRef.current;
    const svg = svgRef.current;
    const ringSvg = ringSvgRef.current;
    const ringHalo = ringHaloRef.current;
    const ring = ringRef.current;
    const warn = warnRef.current;
    const pixelSvg = pixelSvgRef.current;
    const motionGroup = motionGroupRef.current;
    const leaderLine = leaderLineRef.current;
    const leaderDot = leaderDotRef.current;
    const callout = calloutRef.current;
    const labelText = labelTextRef.current;
    if (
      !root ||
      !svg ||
      !ringSvg ||
      !ringHalo ||
      !ring ||
      !warn ||
      !pixelSvg ||
      !motionGroup ||
      !leaderLine ||
      !leaderDot ||
      !callout ||
      !labelText
    )
      return undefined;

    const pool = {
      rect: [...motionGroup.querySelectorAll<SVGRectElement>('rect')],
      path: [...motionGroup.querySelectorAll<SVGPathElement>('path')],
      circle: [...motionGroup.querySelectorAll<SVGCircleElement>('circle')],
    };

    for (const rect of pool.rect) rect.setAttribute('rx', String(MOTION_RECT_RADIUS_PX));

    let raf = 0;
    let disposed = false;
    let sampleCanvas: HTMLCanvasElement | null = null;

    // Writes skip unchanged values (a settled motion on a still box touches nothing).
    const setAttr = (el: Element, name: string, value: string) => {
      if (el.getAttribute(name) !== value) el.setAttribute(name, value);
    };
    const setStyle = (el: HTMLElement | SVGElement, prop: 'display' | 'visibility' | 'left' | 'top' | 'transform', value: string) => {
      if (el.style[prop] !== value) el.style[prop] = value;
    };
    const setData = (el: HTMLElement, key: string, value: string) => {
      if (el.dataset[key] !== value) el.dataset[key] = value;
    };

    const setRect = (rect: SVGRectElement, box: { x: number; y: number; width: number; height: number } | null) => {
      if (!box) {
        setStyle(rect, 'display', 'none');
        return;
      }
      setStyle(rect, 'display', '');
      setAttr(rect, 'x', String(box.x));
      setAttr(rect, 'y', String(box.y));
      setAttr(rect, 'width', String(box.width));
      setAttr(rect, 'height', String(box.height));
    };

    const setRing = (d: string | null) => {
      for (const path of [ringHalo, ring]) {
        if (d === null) {
          setStyle(path, 'display', 'none');
        } else {
          setStyle(path, 'display', '');
          setAttr(path, 'd', d);
        }
      }
    };

    /** Write the run's cue colours as custom properties on the root (unchanged values are skipped). */
    const applyCue = (tone: CueTone) => {
      for (const [name, key] of CUE_PROPERTIES) {
        if (root.style.getPropertyValue(name) !== tone[key]) root.style.setProperty(name, tone[key]);
      }
      setData(root, 'cueAccent', tone.accent);
    };

    /** The run's accent: sampled once from the scene around `box`; the preferred accent when that fails. */
    const chooseCue = (box: Box): CueTone => {
      let accent = PREFERRED_ACCENT;
      try {
        const override = sceneColorsRef.current;
        let colors: SceneColors | null;
        if (override) {
          colors = override(box);
        } else {
          sampleCanvas ??= document.createElement('canvas');
          colors = sampleStageVideo(root, box, sampleCanvas);
        }
        if (colors) accent = pickContrastAccent(colors.bg, colors.obj, PREFERRED_ACCENT);
      } catch {
        // A tainted canvas (SecurityError) or any drawing error: keep the preferred accent for this run.
        accent = PREFERRED_ACCENT;
      }
      return accentTone(accent);
    };

    /** Assign shapes to the pool in order per kind; hide every node left over. */
    const drawMotion = (shapes: MotionShape[]) => {
      const used = { rect: 0, path: 0, circle: 0 };
      for (const shape of shapes) {
        const node = pool[shape.kind][used[shape.kind]];
        used[shape.kind] += 1;
        if (!node) continue;
        if (shape.kind === 'rect') {
          setAttr(node, 'x', fmt(shape.x));
          setAttr(node, 'y', fmt(shape.y));
          setAttr(node, 'width', fmt(shape.width));
          setAttr(node, 'height', fmt(shape.height));
        } else if (shape.kind === 'path') {
          setAttr(node, 'd', shape.d);
        } else {
          setAttr(node, 'cx', fmt(shape.cx));
          setAttr(node, 'cy', fmt(shape.cy));
          setAttr(node, 'r', fmt(shape.r));
        }
        setAttr(node, 'opacity', fmt(shape.opacity));
        setAttr(node, 'data-dashed', 'dashed' in shape && shape.dashed ? 'true' : 'false');
        setAttr(node, 'data-filled', 'filled' in shape && shape.filled ? 'true' : 'false');
        setStyle(node, 'display', shape.opacity > 0 ? '' : 'none');
      }
      for (const kind of ['rect', 'path', 'circle'] as const) {
        for (let i = used[kind]; i < pool[kind].length; i += 1) setStyle(pool[kind][i], 'display', 'none');
      }
    };

    /** No motion on screen: hide the pool, drop the attributes, pause the clock. */
    const clearMotion = () => {
      drawMotion([]);
      lastTickRef.current = null;
      if (root.dataset.motion !== undefined) delete root.dataset.motion;
      if (root.dataset.motionPhase !== undefined) delete root.dataset.motionPhase;
    };

    const reportInstructionVisibility = (vis: boolean) => {
      if (lastInstructionVisibleRef.current !== vis) {
        lastInstructionVisibleRef.current = vis;
        onInstructionVisibilityRef.current?.(vis);
      }
    };

    const hideLeader = () => {
      setStyle(leaderLine, 'display', 'none');
      setStyle(leaderDot, 'display', 'none');
    };

    const hideAll = (reason: string) => {
      setStyle(svg, 'visibility', 'hidden');
      setStyle(ringSvg, 'visibility', 'hidden');
      setStyle(pixelSvg, 'visibility', 'hidden');
      setStyle(callout, 'visibility', 'hidden');
      hideLeader();
      clearMotion();
      setData(root, 'focus', 'none');
      setData(root, 'guideVisible', 'false');
      setData(root, 'guideReason', reason);
      reportInstructionVisibility(false);
    };

    /** Where the callout fits off `avoid` (CSS px), and whether that spot is unobstructed and in bounds. */
    const placeCallout = (
      avoid: Box,
      overlay: { width: number; height: number },
      labelSize: { width: number; height: number },
      obstacles: PxRect[]
    ) => {
      const placed = placeLabel(avoid, overlay, labelSize, obstacles, LABEL_OFFSET_PX, placementRef.current);
      const fits = placed.overlap === 0 && overlay.width > 0 && overlay.height > 0
        && placed.left >= 0 && placed.top >= 0
        && placed.left + labelSize.width <= overlay.width + 0.5
        && placed.top + labelSize.height <= overlay.height + 0.5;
      return { placed, fits };
    };

    const positionCallout = (placed: PlacedLabel, visible: boolean) => {
      if (visible) placementRef.current = placed.placement;
      setStyle(callout, 'left', `${fmt(placed.left)}px`);
      setStyle(callout, 'top', `${fmt(placed.top)}px`);
      setData(callout, 'placement', placed.placement);
      setData(callout, 'overlap', String(Math.round(placed.overlap)));
      setStyle(callout, 'transform', 'none');
      setStyle(callout, 'visibility', visible ? 'visible' : 'hidden');
    };

    const tick = () => {
      raf = 0;
      if (disposed) return;
      const snapshot = liveTrack.current();
      const state = guide.current();
      const now = Date.now();
      const drawable = isLiveBoxDrawable(snapshot, now);
      if (state.motionKey !== motionKeyRef.current) {
        motionKeyRef.current = state.motionKey;
        elapsedRef.current = 0;
        lastTickRef.current = null;
        placementRef.current = null;
        cueLockedRef.current = false;
      }
      setData(root, 'guideWarn', state.warn ? 'true' : 'false');
      if (!visibleRef.current) hideAll('toggle_off');
      else if (!state.binding) hideAll('unbound');
      else if (!drawable || !snapshot?.box) hideAll('box_hidden');
      else if (!snapshotMatchesBinding(snapshot, state.binding)) hideAll('other_anchor');
      else {
        const box = snapshot.box;
        const focus = resolveFocus(state.commands);
        const ringBox = focus ? focusRing(box, focus.pad) : null;
        const actionCmd = state.warn
          ? undefined
          : state.commands.find((c): c is Extract<typeof c, { kind: 'action' }> => c.kind === 'action');
        const labelCmd = state.warn
          ? undefined
          : state.commands.find((c): c is Extract<typeof c, { kind: 'label' }> => c.kind === 'label');
        const hasCallout = state.warn || Boolean(actionCmd || labelCmd);

        // 1. Callout content first (it sizes the callout).
        if (state.warn) {
          if (labelText.textContent !== WARN_LABEL) labelText.textContent = WARN_LABEL;
          if (!callout.classList.contains('anchored-label-warn')) callout.classList.add('anchored-label-warn');
          callout.removeAttribute('data-action');
          callout.removeAttribute('data-direction');
          setData(callout, 'schematic', 'false');
          setAttr(callout, 'aria-label', WARN_LABEL);
        } else if (hasCallout) {
          if (callout.classList.contains('anchored-label-warn')) callout.classList.remove('anchored-label-warn');
          let displayText: string;
          if (actionCmd) {
            displayText = labelCmd ? labelCmd.text : actionShortLabel(actionCmd.action, actionCmd.direction);
            setData(callout, 'action', actionCmd.action);
            setData(callout, 'direction', actionCmd.direction);
          } else {
            displayText = labelCmd ? labelCmd.text : '';
            callout.removeAttribute('data-action');
            callout.removeAttribute('data-direction');
          }
          // The four schematic glyphs carry the "no real pin position" note in the callout itself.
          setData(callout, 'schematic', actionCmd && isSchematicAction(actionCmd.action) ? 'true' : 'false');
          setAttr(callout, 'aria-label', displayText);
          if (labelText.textContent !== displayText) labelText.textContent = displayText;
        }

        // 2. Every layout read of the frame, together, before any svg/position write.
        const width = root.clientWidth;
        const height = root.clientHeight;
        const overlay = { width, height };
        const obstacles = hasCallout ? obstacleRects(root) : [];
        const labelSize = hasCallout ? { width: callout.offsetWidth, height: callout.offsetHeight } : { width: 0, height: 0 };
        // The cue colour: once per run, on its first drawn frame (one small frame sample), then locked.
        const cue = cueLockedRef.current ? null : chooseCue(box);

        // 3. Writes.
        if (cue) {
          applyCue(cue);
          cueLockedRef.current = true;
        }
        setRing(
          ringBox && width > 0 && height > 0
            ? ringPath({ left: ringBox.x * width, top: ringBox.y * height, width: ringBox.width * width, height: ringBox.height * height })
            : null
        );
        setData(root, 'focus', focus?.source ?? 'none');
        setRect(warn, state.warn ? box : null);
        ring.classList.toggle('anchored-focus-warn', state.warn);
        ringHalo.classList.toggle('anchored-focus-warn', state.warn);
        setStyle(svg, 'visibility', 'visible');
        setStyle(ringSvg, 'visibility', 'visible');
        setData(root, 'guideVisible', 'true');

        if (state.warn) {
          clearMotion();
          const { placed } = placeCallout(box, overlay, labelSize, obstacles);
          positionCallout(placed, true);
          hideLeader();
          setStyle(pixelSvg, 'visibility', 'hidden');
          setData(root, 'guideReason', 'drawn');
          reportInstructionVisibility(false);
        } else if (!hasCallout) {
          clearMotion();
          setStyle(callout, 'visibility', 'hidden');
          setStyle(pixelSvg, 'visibility', 'hidden');
          hideLeader();
          setData(root, 'guideReason', 'drawn');
          reportInstructionVisibility(false);
        } else {
          const targetBox = ringBox ?? box;
          let motionAvoid: Box | null = null;

          if (actionCmd && width > 0 && height > 0) {
            const t = performance.now();
            if (lastTickRef.current !== null) {
              elapsedRef.current += Math.min(Math.max(0, t - lastTickRef.current), MOTION_MAX_TICK_MS);
            }
            lastTickRef.current = t;
            const pxBox = { left: box.x * width, top: box.y * height, width: box.width * width, height: box.height * height };
            const frame = motionFrame(elapsedRef.current, reducedMotionRef.current);
            drawMotion(motionShapes(actionCmd.action, actionCmd.direction, pxBox, frame, overlay));
            setData(root, 'motion', `${actionCmd.action}:${actionCmd.direction}`);
            setData(root, 'motionPhase', frame.settled ? 'settled' : 'playing');
            const extent = cachedMotionExtent(actionCmd.action, actionCmd.direction, pxBox, overlay);
            motionAvoid = unionBox(targetBox, {
              x: extent.left / width,
              y: extent.top / height,
              width: extent.width / width,
              height: extent.height / height,
            });
          } else {
            clearMotion();
          }

          // Off the motion when there is room; otherwise off the object only (the motion may run under it).
          let avoided = motionAvoid ? 'motion' : 'object';
          let { placed, fits } = placeCallout(motionAvoid ?? targetBox, overlay, labelSize, obstacles);
          if (!fits && motionAvoid) {
            ({ placed, fits } = placeCallout(targetBox, overlay, labelSize, obstacles));
            avoided = 'object';
          }
          positionCallout(placed, fits);
          setData(callout, 'avoid', avoided);

          // The leader line ends at the object (ring/box), not at the motion's extent.
          const linePts = computeLeaderLine(
            targetBox,
            overlay,
            { left: placed.left, top: placed.top, width: labelSize.width, height: labelSize.height },
            placed.placement
          );
          setAttr(leaderLine, 'd', `M ${fmt(linePts.from.x)} ${fmt(linePts.from.y)} L ${fmt(linePts.to.x)} ${fmt(linePts.to.y)}`);
          setAttr(leaderDot, 'cx', fmt(linePts.to.x));
          setAttr(leaderDot, 'cy', fmt(linePts.to.y));
          setStyle(leaderLine, 'display', fits ? '' : 'none');
          setStyle(leaderDot, 'display', fits ? '' : 'none');
          setStyle(pixelSvg, 'visibility', fits || actionCmd ? 'visible' : 'hidden');

          setData(root, 'guideReason', fits ? 'drawn' : 'no_callout_space');
          reportInstructionVisibility(fits);
        }
      }
      if (root.dataset.guideVisible === 'true') schedule();
    };

    function schedule() {
      if (raf !== 0 || disposed) return;
      raf = window.requestAnimationFrame(tick);
    }
    scheduleRef.current = schedule;

    const offTrack = liveTrack.subscribe(schedule);
    const offGuide = guide.subscribe(schedule);
    schedule();
    return () => {
      disposed = true;
      offTrack();
      offGuide();
      if (raf !== 0) window.cancelAnimationFrame(raf);
      raf = 0;
      hideAll('unmounted');
    };
  }, [liveTrack, guide]);

  // The toggle re-evaluates on the next frame.
  useEffect(() => {
    scheduleRef.current();
  }, [visible]);

  return (
    <div
      ref={rootRef}
      className="anchored-guidance"
      data-testid="anchored-guidance"
      data-guide-visible="false"
      data-guide-reason="unbound"
      data-guide-warn="false"
    >
      <svg ref={svgRef} className="anchored-guidance-svg" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
        <rect ref={warnRef} className="anchored-warn-box" vectorEffect="non-scaling-stroke" data-testid="anchored-warn-box" />
      </svg>
      <svg ref={ringSvgRef} className="anchored-ring-svg" aria-hidden="true">
        <path ref={ringHaloRef} className="anchored-focus-outer" style={{ display: 'none' }} />
        <path ref={ringRef} className="anchored-focus-inner" data-testid="anchored-focus" style={{ display: 'none' }} />
      </svg>
      <svg ref={pixelSvgRef} className="anchored-pixel-svg" aria-hidden="true">
        <g ref={motionGroupRef} className="anchored-motion" data-testid="anchored-motion">
          {Array.from({ length: MOTION_POOL.rect }, (_, i) => (
            <rect key={`r${i}`} style={{ display: 'none' }} />
          ))}
          {Array.from({ length: MOTION_POOL.path }, (_, i) => (
            <path key={`p${i}`} style={{ display: 'none' }} />
          ))}
          {Array.from({ length: MOTION_POOL.circle }, (_, i) => (
            <circle key={`c${i}`} style={{ display: 'none' }} />
          ))}
        </g>
        <path ref={leaderLineRef} className="anchored-leader-line" />
        <circle ref={leaderDotRef} className="anchored-leader-dot" r="2.5" />
      </svg>
      <div
        ref={calloutRef}
        className="anchored-label anchored-callout"
        data-testid="anchored-label"
        role="note"
        aria-live="polite"
      >
        <span className="anchored-callout-dot" aria-hidden="true" />
        <span ref={labelTextRef} className="anchored-callout-text" />
        <span className="anchored-callout-note">동작 예시 · 실제 핀 위치 아님</span>
      </div>
    </div>
  );
}
