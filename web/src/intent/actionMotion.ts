/**
 * The action motion layer: box-relative animated primitives that show a step's `action` happening on the
 * tracked object (grasp, remove, move, rotate, press, open, close, insert, and since 2026-10-03 attach, fold,
 * place, fit, screw, hold, flip, pull, push — the latter nine from a designer's mock; since 2026-10-05 align,
 * connect, disconnect, bend — the graph core's schematic actions). Pure geometry and timing;
 * the overlay assigns the shapes to a fixed pool of svg nodes every animation frame.
 *
 * - Geometry is in CSS px of the overlay root (not the 0..1 viewBox), so arcs, brackets and arrowheads are
 *   never distorted by the frame's non-uniform scale (design v4 principle 1: anchors give a box only, no
 *   pose; screen-space drawing, no non-uniform scale). Every shape is derived from the live box alone.
 * - Frame: move and insert keep their ghost travel inside the overlay (box side, at least 40 px, never past
 *   the frame edge minus 6 px); with under 16 px of room move draws a trail to the edge and an arrowhead just
 *   inside it instead of a ghost. pull/push share the room toward (and, for push, behind) the box between the
 *   ghost's shift, the line and the arrowhead, and pull back onto the box when even the arrowhead does not fit;
 *   across the motion their hook/bar and heads stay inside too. fold/flip draw their arc on the side with room
 *   (right first) and shrink/move it inside when neither side has enough; attach keeps its band inside. place/fit
 *   aim at a destination box: the caller's `target`, else a PROVISIONAL box beside the tracked one
 *   (`provisionalTarget`; there is no destination anchor yet) placed so the landing / end ghost fits the frame.
 *   For these actions every drawn vertex stays inside the frame for any box inside it (tested unclipped); the
 *   dashed seams are cut at the frame edge. screw and hold (like press and rotate) are not frame-aware. Small boxes get minimum sizes (arrowheads 10 px, remove growth 24 px per side,
 *   press/hold ring 28 px) so the motion stays visible. align/connect/disconnect/bend are schematic: they mark
 *   the tracked box only (no pin, polarity, voltage or measured target position) and are scaled by one factor to
 *   the room the frame leaves around the box centre or beyond its edge, so an edge or tiny box keeps a complete
 *   glyph inside the frame instead of a clipped one.
 * - Timing: `MOTION_PLAY_COUNT` plays of `MOTION_CYCLE_MS`; the motion runs in the first
 *   `MOTION_EASE_FRACTION` of a cycle and then holds; every cycle but the last fades out at its end. After
 *   the last play the end pose is held (no endless repetition). Reduced motion shows only the end pose.
 * - Paths use only `M`/`L`/`Z` (arcs are polylines), so their bounds and mirror images are exact.
 *
 * Nothing here imports a value from another module, so `node --test` runs it directly.
 */
import type { ActionDirection, ActionType } from './anchorProject';

/** An axis-aligned rectangle in CSS px of the overlay root. */
export interface PxBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export type MotionShape =
  | { kind: 'rect'; x: number; y: number; width: number; height: number; opacity: number; dashed?: boolean; filled?: boolean }
  | { kind: 'path'; d: string; opacity: number; filled?: boolean; dashed?: boolean }
  | { kind: 'circle'; cx: number; cy: number; r: number; opacity: number; filled?: boolean };

/** The overlay's size in CSS px (the frame the motion must stay inside where it can). */
export interface FrameSize {
  width: number;
  height: number;
}

export interface MotionFrame {
  /** 0..1, eased within the cycle (1 = end pose). */
  progress: number;
  /** 0..1 fade at cycle boundaries; multiplies every shape's opacity. */
  alpha: number;
  cycle: number;
  /** The plays are over (or reduced motion): the end pose is held. */
  settled: boolean;
}

/** One play of the motion. */
export const MOTION_CYCLE_MS = 1600;
/** Plays before the end pose is held. */
export const MOTION_PLAY_COUNT = 3;
/** The motion happens in this leading fraction of a cycle; the rest holds the end pose. */
export const MOTION_EASE_FRACTION = 0.7;
/** Fade in at a cycle's start and out at a non-final cycle's end. */
export const MOTION_FADE_MS = 150;
/** Largest clock step per drawn frame (a stalled tab or a hidden box never jumps the motion). */
export const MOTION_MAX_TICK_MS = 100;
/** Extent padding: the 3 px stroke's half width plus the drop-shadow halo, rounded up. */
export const MOTION_STROKE_PAD_PX = 4;
/** The most nodes of each kind any action uses: the overlay's fixed svg node pool. */
export const MOTION_POOL = Object.freeze({ rect: 1, path: 5, circle: 2 });

// Visual constants (CSS px or box fractions).
const GRASP_HEIGHT_FRACTION = 0.6;
const GRASP_ARM_FRACTION = 0.12;
const GRASP_ARM_MIN_PX = 8;
const GRASP_ARM_MAX_PX = 24;
const GRASP_START_FRACTION = 0.35;
const GRASP_START_EXTRA_PX = 24;
const GRASP_END_GAP_PX = 6;
/** remove: the ghost grows about the box centre by this fraction of the side on each side (1.0 → 1.35)… */
const REMOVE_GROW_FRACTION = 0.175;
/** …but at least / at most this many px per side, so a small box still visibly "pulls out". */
const REMOVE_GROW_MIN_PX = 24;
const REMOVE_GROW_MAX_PX = 80;
/** Gap between a ghost corner and its outward arrowhead. */
const REMOVE_CORNER_GAP_PX = 2;
/** move/insert ghost travel: the box side, at least this many px, never past the frame edge minus the inset. */
const MIN_TRAVEL_PX = 40;
const FRAME_INSET_PX = 6;
/** Below this room to the frame edge, move draws no ghost: a trail to the edge and an arrowhead just inside it. */
const EDGE_MODE_ROOM_PX = 16;
/** The edge-mode arrowhead's tip stays this far inside the frame edge. */
const EDGE_TIP_INSET_PX = 3;
const ROTATE_RADIUS_EXTRA_PX = 10;
const ROTATE_SWEEP_DEG = 270;
const ROTATE_SEGMENT_DEG = 4;
const PRESS_START_FRACTION = 0.55;
const PRESS_START_MIN_PX = 28;
const PRESS_END_FRACTION = 0.12;
const PRESS_END_MIN_PX = 6;
const PRESS_DOT_R_PX = 3;
const PRESS_DOT_FROM = 0.8;
const LID_LIFT_FRACTION = 0.35;
const LID_LIFT_MIN_PX = 16;
const LID_WIDTH_SCALE = 0.9;
const SEAM_OVERHANG_PX = 6;
/** insert: the ghost's centre starts this many box heights above the box centre (0.6·h above the top edge). */
const INSERT_TRAVEL_FRACTION = 1.1;
const INSERT_START_SCALE = 0.8;
const INSERT_END_SCALE = 0.45;
const INSERT_TRAIL_FRACTION = 0.35;
const INSERT_TRAIL_MIN_PX = 10;
const INSERT_TRAIL_MAX_PX = 36;
/** attach: a tape band sweeps across the box's middle, from 12 px left of it to 12 px right of it. */
const ATTACH_BAND_FRACTION = 0.14;
const ATTACH_BAND_MIN_PX = 8;
const ATTACH_BAND_MAX_PX = 18;
const ATTACH_OVERHANG_PX = 2 * SEAM_OVERHANG_PX;
const ATTACH_BAND_OPACITY = 0.45;
/** fold: a flap standing on the box's top edge (half its height, ≥12 px) folds down over it. */
const FOLD_FLAP_FRACTION = 0.5;
const FOLD_FLAP_MIN_PX = 12;
const FOLD_FLAP_INSET_FRACTION = 0.06;
const FOLD_ARC_SEGMENTS = 36;
/** Smallest fold arc radii when the frame leaves little room (vertical, horizontal). */
const FOLD_ARC_MIN_RY_PX = 12;
const FOLD_ARC_MIN_RX_PX = 6;
/** place / fit: the provisional destination beside the box (designer's preview `targetOf`). */
const TARGET_GAP_FRACTION = 0.4;
const TARGET_GAP_MIN_PX = 60;
const TARGET_FRAME_INSET_PX = 12;
const PLACE_TARGET_WIDTH_SCALE = 1.3;
const PLACE_TARGET_MIN_WIDTH_PX = 80;
const PLACE_TARGET_HEIGHT_SCALE = 0.45;
const PLACE_TARGET_MIN_HEIGHT_PX = 24;
const PLACE_TARGET_TOP_FRACTION = 0.9;
const PLACE_LIFT_FRACTION = 0.5;
const PLACE_LIFT_MIN_PX = 20;
const PLACE_LIFT_MAX_PX = 80;
const PLACE_PATH_SEGMENTS = 24;
const PLACE_MARK_TICK_PX = 6;
/** screw: a head circle with a turning cross and a 108° arc that runs 1.5 turns around it. */
const SCREW_HEAD_FRACTION = 0.22;
const SCREW_HEAD_MIN_PX = 8;
const SCREW_HEAD_MAX_PX = 26;
const SCREW_ARC_GAP_PX = 10;
const SCREW_TURNS = 1.5;
const SCREW_ARC_RAD = 0.6 * Math.PI;
const SCREW_ARC_SEGMENTS = 16;
/** hold: the press ring contracts in the first 35 % of the motion, then a progress ring closes around it. */
const HOLD_PRESS_FRACTION = 0.35;
const HOLD_RING_GAP_PX = 10;
const HOLD_RING_SEGMENTS = 72;
/** flip: the box squashes to a line and back while an arc runs down its side. */
const FLIP_ARC_GAP_PX = 8;
const FLIP_ARC_RX_FRACTION = 0.18;
const FLIP_ARC_RX_MIN_PX = 16;
const FLIP_ARC_SEGMENTS = 36;
const FLIP_MIN_HEIGHT_PX = 2;
/** Smallest flip arc horizontal radius when the frame leaves little room. */
const FLIP_ARC_MIN_RX_PX = 8;
/** pull / push: the ghost shifts 0.3·side (16..48 px) the given way. */
const PUSH_PULL_SHIFT_FRACTION = 0.3;
const PUSH_PULL_SHIFT_MIN_PX = 16;
const PUSH_PULL_SHIFT_MAX_PX = 48;
/** Half the width of the pull hook / push bar across the motion: 0.4 of that side, at least 8 px. */
const PUSH_PULL_HALF_FRACTION = 0.4;
const PUSH_PULL_HALF_MIN_PX = 8;
const PUSH_PULL_GAP_PX = 4;
const PULL_ARM_FRACTION = 0.12;
const PULL_ARM_MIN_PX = 8;
const PULL_ARM_MAX_PX = 16;
/** The push shaft is this many arrow lengths; the pull line two arrow lengths plus 8 px. */
const PUSH_SHAFT_ARROWS = 2.5;
const PULL_LINE_ARROWS = 2;
const PULL_LINE_EXTRA_PX = 8;
/**
 * align / connect / disconnect / bend are schematic glyphs: everything is drawn relative to the tracked box only
 * — no pin, polarity, voltage or measured target position — and each takes the room the frame leaves around the
 * box centre, so a box at the frame's edge gets a smaller but complete glyph rather than one drawn outside it.
 */
/** align: the component (a ghost of half the box) slides sideways onto the guide line through the box's middle. */
const ALIGN_GHOST_FRACTION = 0.5;
const ALIGN_TRAVEL_FRACTION = 0.35;
/** The guide's end ticks (half length) and its overhang past the component, from the component's height. */
const ALIGN_TICK_FRACTION = 0.22;
const ALIGN_TICK_MIN_PX = 4;
const ALIGN_TICK_MAX_PX = 14;
const ALIGN_GUIDE_OVER_FRACTION = 0.12;
const ALIGN_GUIDE_OVER_MIN_PX = 2;
const ALIGN_GUIDE_OVER_MAX_PX = 8;
/** connect / disconnect: two connector halves whose pins meet on the box's middle line. */
const CONNECT_BODY_FRACTION = 0.18;
const CONNECT_BODY_MIN_PX = 8;
const CONNECT_BODY_MAX_PX = 40;
const CONNECT_PIN_FRACTION = 0.25;
const CONNECT_PIN_MIN_PX = 4;
const CONNECT_PIN_MAX_PX = 12;
const CONNECT_HEIGHT_FRACTION = 0.5;
/** The widest separation between the halves' inner edges, on top of the mating separation (2 × the pin). */
const CONNECT_SEPARATION_FRACTION = 0.3;
const CONNECT_SEPARATION_MIN_PX = 18;
const CONNECT_SEPARATION_MAX_PX = 80;
/** The mating guide's overhang past a half, from the half's height. */
const CONNECT_GUIDE_OVER_FRACTION = 0.3;
const CONNECT_GUIDE_OVER_MIN_PX = 3;
const CONNECT_GUIDE_OVER_MAX_PX = 14;
/** The direction arrowhead's stand-off from a half's outer edge: behind it (connect) or ahead of it (disconnect). */
const CONNECT_ARROW_GAP_PX = 3;
/** bend: the lead leaves the box, then its leg swings about a hinge off the straight continuation. */
const BEND_STUB_FRACTION = 0.35;
const BEND_STUB_MIN_PX = 12;
const BEND_STUB_MAX_PX = 40;
const BEND_LEG_FRACTION = 0.5;
const BEND_LEG_MIN_PX = 20;
const BEND_LEG_MAX_PX = 72;
/** How far the leg swings off the straight continuation at the end pose (one quadrant of swing at most). */
const BEND_SWEEP_DEG = 72;
const BEND_ARC_FRACTION = 0.35;
const BEND_ARC_MIN_PX = 8;
const BEND_ARC_MAX_PX = 28;
const BEND_ARC_SEGMENTS = 12;
const BEND_JOINT_FRACTION = 0.12;
const BEND_JOINT_MIN_PX = 3;
const BEND_JOINT_MAX_PX = 8;
/**
 * Every vertex of an arrowhead whose tip sits ≤ 0.5 arrow lengths from a point of its curve (along the tangent)
 * is within this many arrow lengths of that point (√(1² + 0.6²) ≈ 1.17): the curve's bounds grown by it bound
 * the head at any progress.
 */
const CURVE_HEAD_REACH = 1.2;
const ARROW_FRACTION = 0.15;
const ARROW_MIN_PX = 10;
const ARROW_MAX_PX = 14;
/** A trail/arrowhead fades in over this leading part of the progress (no zero-length arrow at p=0). */
const TRAIL_FADE_PROGRESS = 0.15;
/** Cached extents are computed for the rounded box size; this covers the ≤0.5 px rounding (≤2× per axis). */
const EXTENT_ROUNDING_SLACK_PX = 1;
const EXTENT_CACHE_MAX = 512;
const GHOST_OPACITY = 0.9;
const LINE_OPACITY = 0.9;

const clamp = (value: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, value));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
/** Fixed-precision path numbers (2 decimals; `-0` written as `0`). */
const n = (value: number) => {
  const r = Math.round(value * 100) / 100;
  return String(Object.is(r, -0) ? 0 : r);
};

const SETTLED: MotionFrame = Object.freeze({ progress: 1, alpha: 1, cycle: MOTION_PLAY_COUNT, settled: true });

/** Where the motion is after `elapsedMs` of drawn time. */
export function motionFrame(elapsedMs: number, reducedMotion: boolean): MotionFrame {
  const elapsed = Number.isFinite(elapsedMs) ? Math.max(0, elapsedMs) : 0;
  if (reducedMotion || elapsed >= MOTION_CYCLE_MS * MOTION_PLAY_COUNT) return { ...SETTLED };
  const cycle = Math.floor(elapsed / MOTION_CYCLE_MS);
  const inCycle = elapsed - cycle * MOTION_CYCLE_MS;
  const t = inCycle / MOTION_CYCLE_MS;
  const progress = easeInOutCubic(Math.min(t / MOTION_EASE_FRACTION, 1));
  const fadeIn = Math.min(1, inCycle / MOTION_FADE_MS);
  const fadeOut = cycle < MOTION_PLAY_COUNT - 1 ? Math.min(1, (MOTION_CYCLE_MS - inCycle) / MOTION_FADE_MS) : 1;
  return { progress, alpha: clamp(Math.min(fadeIn, fadeOut), 0, 1), cycle, settled: false };
}

/** A filled triangle whose tip is at (x, y), pointing along (ux, uy). */
function arrowHead(x: number, y: number, ux: number, uy: number, size: number): string {
  const len = Math.hypot(ux, uy) || 1;
  const dx = ux / len;
  const dy = uy / len;
  const bx = x - dx * size;
  const by = y - dy * size;
  const half = size * 0.6;
  return `M ${n(x)} ${n(y)} L ${n(bx - dy * half)} ${n(by + dx * half)} L ${n(bx + dy * half)} ${n(by - dx * half)} Z`;
}

const arrowSize = (box: PxBox) => clamp(ARROW_FRACTION * Math.min(box.width, box.height), ARROW_MIN_PX, ARROW_MAX_PX);

/** A rect scaled by `s` about (cx, cy). */
function scaledRect(cx: number, cy: number, width: number, height: number, s: number) {
  return { x: cx - (width * s) / 2, y: cy - (height * s) / 2, width: width * s, height: height * s };
}

const DIRECTION_VECTORS: Partial<Record<ActionDirection, [number, number]>> = {
  up: [0, -1],
  down: [0, 1],
  left: [-1, 0],
  right: [1, 0],
};

/** The frame's usable edges in CSS px (0..width/height), or ±Infinity without a frame size. */
interface FrameEdges {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** The frame's usable edges; without a frame size, unbounded. */
function frameEdges(viewport: FrameSize | undefined): FrameEdges {
  const ok = viewport && viewport.width > 0 && viewport.height > 0;
  return ok
    ? { x0: 0, y0: 0, x1: viewport.width, y1: viewport.height }
    : { x0: -Infinity, y0: -Infinity, x1: Infinity, y1: Infinity };
}

/** Room (px) from the box's edge on side (dx, dy) to the frame edge minus the inset; Infinity without a frame. */
function roomToward(box: PxBox, dx: number, dy: number, edges: FrameEdges): number {
  const room =
    dx > 0 ? edges.x1 - FRAME_INSET_PX - (box.left + box.width)
    : dx < 0 ? box.left - (edges.x0 + FRAME_INSET_PX)
    : dy > 0 ? edges.y1 - FRAME_INSET_PX - (box.top + box.height)
    : box.top - (edges.y0 + FRAME_INSET_PX);
  return Number.isNaN(room) ? 0 : room;
}

/** +1 to draw a side arc right of the box, −1 to draw it left: right when it fits there or has the most room. */
function arcSide(box: PxBox, need: number, edges: FrameEdges): 1 | -1 {
  const right = roomToward(box, 1, 0, edges);
  if (right >= need) return 1;
  return roomToward(box, -1, 0, edges) > right ? -1 : 1;
}

/**
 * A span of half size `rWant` around `c` that fits [lo, hi]: first shrunk toward `rMin` around `c`, then (when
 * even that does not fit) moved inside, never larger than half the interval. Unbounded edges change nothing.
 */
function fitSpan(c: number, rWant: number, rMin: number, lo: number, hi: number): { c: number; r: number } {
  let r = Math.min(rWant, Math.max(rMin, Math.min(c - lo, hi - c)));
  if (Number.isFinite(hi - lo)) r = Math.max(0, Math.min(r, (hi - lo) / 2));
  return { c: clamp(c, lo + r, hi - r), r };
}

/**
 * A side arc's centre x and horizontal radius: `gap` px beside the box on side `sx`, its radius shrunk toward
 * `rxMin` to the room there, then moved in, so that [ax, ax + sx·rx] stays `reach` px inside the frame's inset.
 */
function sideArcX(box: PxBox, sx: 1 | -1, gap: number, rxWant: number, rxMin: number, reach: number, edges: FrameEdges) {
  const lo = edges.x0 + FRAME_INSET_PX + reach;
  const hi = edges.x1 - FRAME_INSET_PX - reach;
  const near = sx > 0 ? box.left + Math.max(0, box.width) + gap : box.left - gap;
  let rx = Math.min(rxWant, Math.max(rxMin, sx > 0 ? hi - near : near - lo));
  if (Number.isFinite(hi - lo)) rx = Math.max(0, Math.min(rx, hi - lo));
  const ax = sx > 0 ? clamp(near, lo, hi - rx) : clamp(near, lo + rx, hi);
  return { ax, rx };
}

/** A dashed seam across the box at `y`, overhanging it by 6 px each side but cut at the frame edges. */
function seamPath(box: PxBox, y: number, edges: FrameEdges): string {
  const x0 = Math.max(box.left - SEAM_OVERHANG_PX, edges.x0);
  const x1 = Math.min(box.left + Math.max(0, box.width) + SEAM_OVERHANG_PX, edges.x1);
  return `M ${n(x0)} ${n(y)} L ${n(x1)} ${n(y)}`;
}

/**
 * PROVISIONAL (2026-10-03): the destination place/fit aim at when the caller has none. The plan has no
 * destination anchor yet, so this is a box beside the tracked one (the designer's preview `targetOf`): a
 * clamp(0.4·w, ≥60 px) gap to the right — to the left when the right side lacks room — kept 12 px inside the
 * frame; place gets a low support (max(1.3·w, 80) × max(0.45·h, 24) px, top at 0.9·h), fit a box of the same
 * size at the same height. It shows the motion's shape, not where the object really goes; replace it with a
 * tracked destination when one exists.
 */
export function provisionalTarget(action: 'place' | 'fit', box: PxBox, viewport?: FrameSize): PxBox {
  const e = frameEdges(viewport);
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const inset = TARGET_FRAME_INSET_PX;
  const gapWant = Math.max(TARGET_GAP_MIN_PX, w * TARGET_GAP_FRACTION);
  if (action === 'fit') {
    // The ghost slides `gap` px toward a same-size destination and ends touching it, so the PAIR (end ghost +
    // destination) must fit: right of the box when the gap fits there, else left, else the side with more room;
    // the gap shrinks to that room (0 when even the pair does not fit: the ghost then stays and only the junction
    // marks the side).
    const capRight = e.x1 - inset - (box.left + 2 * w);
    const capLeft = box.left - w - (e.x0 + inset);
    const right = capRight >= gapWant || (capLeft < gapWant && capRight >= capLeft);
    const gap = clamp(right ? capRight : capLeft, 0, gapWant);
    const left = right ? box.left + w + gap : box.left - gap - w;
    return { left, top: clamp(box.top, e.y0 + inset, e.y1 - h - inset), width: w, height: h };
  }
  // Never wider than the frame allows (but at least the box, so the ghost still rests on it).
  const tw = Math.min(Math.max(w * PLACE_TARGET_WIDTH_SCALE, PLACE_TARGET_MIN_WIDTH_PX), Math.max(w, e.x1 - e.x0 - 2 * inset));
  const th = Math.max(h * PLACE_TARGET_HEIGHT_SCALE, PLACE_TARGET_MIN_HEIGHT_PX);
  let left = box.left + w + gapWant;
  if (left + tw > e.x1 - inset) left = box.left - gapWant - tw;
  left = clamp(left, e.x0 + inset, e.x1 - tw - inset);
  // The ghost (the box's height) must land on it inside the frame: its top is at least h below the frame's inset.
  const top = Math.max(
    clamp(box.top + h * PLACE_TARGET_TOP_FRACTION, e.y0 + inset + h, e.y1 - th - inset),
    e.y0 + FRAME_INSET_PX + h
  );
  return { left, top, width: tw, height: th };
}

/**
 * fold's flap and side arc. The hinge is the top edge and the flap stands above it (sy = −1) — or, without room
 * above and with more below, the bottom edge with the flap hanging below (sy = +1). The flap is cut to the room on
 * its side and to the room past the hinge it folds into. The arc is a half ellipse beside the box on the side with
 * room (sx), centred on the hinge; with too little room it shrinks and moves inside the frame (`fitSpan`,
 * `sideArcX`), its arrowhead included.
 */
function foldGeometry(box: PxBox, edges: FrameEdges) {
  const h = Math.max(0, box.height);
  const flapWant = Math.max(FOLD_FLAP_FRACTION * h, FOLD_FLAP_MIN_PX);
  const size = arrowSize(box);
  const reach = CURVE_HEAD_REACH * size;
  const ryWant = flapWant * 0.8 + 8;
  const up = Math.max(0, roomToward(box, 0, -1, edges));
  const down = Math.max(0, roomToward(box, 0, 1, edges));
  const sy: 1 | -1 = up >= Math.max(flapWant, ryWant + reach) || up >= down ? -1 : 1;
  const hingeY = sy < 0 ? box.top : box.top + h;
  const flapH = Math.min(flapWant, sy < 0 ? up : down, h + (sy < 0 ? down : up));
  const span = fitSpan(hingeY, ryWant, FOLD_ARC_MIN_RY_PX, edges.y0 + FRAME_INSET_PX + reach, edges.y1 - FRAME_INSET_PX - reach);
  const gap = SEAM_OVERHANG_PX + 4;
  const sx = arcSide(box, gap + 0.5 * span.r + reach, edges);
  const { ax, rx } = sideArcX(box, sx, gap, 0.5 * span.r, Math.min(0.5 * span.r, FOLD_ARC_MIN_RX_PX), reach, edges);
  return { flapH, size, rx, ry: span.r, sx, sy, ax, ay: span.c, hingeY };
}

/**
 * flip's side arc: a half ellipse around the box's middle on the side with room; its radii (h/2 + 8 vertical,
 * max(16, 0.18·w) horizontal) shrink to the room and the arc moves inside the frame when they must (`fitSpan`,
 * `sideArcX`), its arrowhead included.
 */
function flipGeometry(box: PxBox, edges: FrameEdges) {
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const size = arrowSize(box);
  const reach = CURVE_HEAD_REACH * size;
  const rxWant = Math.max(FLIP_ARC_RX_MIN_PX, FLIP_ARC_RX_FRACTION * w);
  const span = fitSpan(box.top + h / 2, h / 2 + FLIP_ARC_GAP_PX, FLIP_ARC_GAP_PX, edges.y0 + FRAME_INSET_PX + reach, edges.y1 - FRAME_INSET_PX - reach);
  const sx = arcSide(box, FLIP_ARC_GAP_PX + rxWant + reach, edges);
  const { ax, rx } = sideArcX(box, sx, FLIP_ARC_GAP_PX, rxWant, Math.min(rxWant, FLIP_ARC_MIN_RX_PX), reach, edges);
  return { size, rx, ry: span.r, sx, ax, ay: span.c };
}

function screwGeometry(box: PxBox) {
  const rHead = clamp(SCREW_HEAD_FRACTION * Math.min(Math.max(0, box.width), Math.max(0, box.height)), SCREW_HEAD_MIN_PX, SCREW_HEAD_MAX_PX);
  return { rHead, r: rHead + SCREW_ARC_GAP_PX, size: arrowSize(box) };
}

function pressRadii(w: number, h: number) {
  const rEnd = Math.max(PRESS_END_FRACTION * Math.min(w, h), PRESS_END_MIN_PX);
  const rStart = Math.max(PRESS_START_FRACTION * Math.max(w, h), PRESS_START_MIN_PX, rEnd + 4);
  return { rEnd, rStart };
}

/** place's travel: from the box centre over an arc of `lift` px to rest on the destination's top. */
function placeGeometry(box: PxBox, edges: FrameEdges, viewport: FrameSize | undefined, target: PxBox | undefined) {
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const cx = box.left + w / 2;
  const cy = box.top + h / 2;
  const t = target ?? provisionalTarget('place', box, viewport);
  const tcx = t.left + t.width / 2;
  // The landing surface: the destination's top, lowered (ghost AND landing mark together) when the ghost resting
  // on it would cross the frame's top inset. The provisional destination already leaves that room.
  const surface = Math.max(t.top, edges.y0 + FRAME_INSET_PX + h);
  const landCy = surface - h / 2;
  // The arc's peak keeps the ghost's top inside the frame (y(q) ≥ min(cy, landCy) − lift for every q).
  const headroom = Math.min(cy, landCy) - h / 2 - (edges.y0 + FRAME_INSET_PX);
  const lift = Math.min(clamp(PLACE_LIFT_FRACTION * h, PLACE_LIFT_MIN_PX, PLACE_LIFT_MAX_PX), Math.max(0, headroom));
  const at = (q: number): [number, number] => [lerp(cx, tcx, q), lerp(cy, landCy, q) - lift * Math.sin(Math.PI * q)];
  return { t, tcx, surface, landCy, lift, at, cx, cy, w, h };
}

/** A polyline through `count + 1` points of `point(i / count)`. */
function polyline(count: number, point: (u: number) => [number, number]): string {
  let d = '';
  for (let i = 0; i <= count; i += 1) {
    const [x, y] = point(i / count);
    d += `${i === 0 ? 'M' : ' L'} ${n(x)} ${n(y)}`;
  }
  return d;
}

/** Room (px) from the centre line `c` to the frame's edge on one side, minus the inset; Infinity when unbounded. */
function centreRoom(c: number, edge: number, sign: 1 | -1): number {
  const room = sign > 0 ? edge - FRAME_INSET_PX - c : c - (edge + FRAME_INSET_PX);
  return Number.isFinite(room) ? room : Infinity;
}

/** How much of a glyph that wants `want` px around the centre fits in `room` (never more than all of it). */
const centreFit = (want: number, room: number) => (want > 0 ? clamp(room / want, 0, 1) : 1);

/** How much of a glyph that reaches `reach` px past a box edge fits in the `room` there (never more than all). */
const edgeFit = (reach: number, room: number, boxHalf: number) =>
  reach > 0 ? clamp((room - boxHalf) / reach, 0, 1) : 1;

/**
 * align's guide and component: a dashed vertical guide line through the box's middle, ticked at both ends, and a
 * ghost of half the box that starts `travel` px to the right of it and settles centred on it, arrowhead leading.
 * The fit `k` keeps the ghost's travel and the guide's ticks inside the room around the box centre.
 */
function alignGeometry(box: PxBox, edges: FrameEdges) {
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const cx = box.left + w / 2;
  const cy = box.top + h / 2;
  const size = arrowSize(box);
  const gw = ALIGN_GHOST_FRACTION * w;
  const gh = ALIGN_GHOST_FRACTION * h;
  const travel = ALIGN_TRAVEL_FRACTION * w;
  const tick = clamp(ALIGN_TICK_FRACTION * gh, ALIGN_TICK_MIN_PX, ALIGN_TICK_MAX_PX);
  const over = clamp(ALIGN_GUIDE_OVER_FRACTION * gh, ALIGN_GUIDE_OVER_MIN_PX, ALIGN_GUIDE_OVER_MAX_PX);
  const wantX = Math.max(travel + gw / 2, tick, size / 2);
  const wantY = Math.max(gh / 2 + over, size / 2);
  const k = Math.min(
    centreFit(wantX, centreRoom(cx, edges.x1, 1)),
    centreFit(wantX, centreRoom(cx, edges.x0, -1)),
    centreFit(wantY, centreRoom(cy, edges.y1, 1)),
    centreFit(wantY, centreRoom(cy, edges.y0, -1))
  );
  return { cx, cy, gw: gw * k, gh: gh * k, travel: travel * k, tick: tick * k, over: over * k, size: size * k };
}

/**
 * connect / disconnect's two connector halves: bodies `bw` wide (centred on the box, `hh` tall) whose inner edges
 * are `sepMax` apart at their widest, each with a pin reaching inward and a direction arrowhead outside its outer
 * edge (behind it for connect, ahead of it for disconnect). The fit `k` keeps the widest pose inside the room.
 */
function connectGeometry(box: PxBox, edges: FrameEdges) {
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const cx = box.left + w / 2;
  const cy = box.top + h / 2;
  const size = arrowSize(box);
  const bw = clamp(CONNECT_BODY_FRACTION * w, CONNECT_BODY_MIN_PX, CONNECT_BODY_MAX_PX);
  const pin = clamp(CONNECT_PIN_FRACTION * bw, CONNECT_PIN_MIN_PX, CONNECT_PIN_MAX_PX);
  const hh = CONNECT_HEIGHT_FRACTION * h;
  const guide = clamp(CONNECT_GUIDE_OVER_FRACTION * hh, CONNECT_GUIDE_OVER_MIN_PX, CONNECT_GUIDE_OVER_MAX_PX);
  const sepMax = 2 * pin + clamp(CONNECT_SEPARATION_FRACTION * w, CONNECT_SEPARATION_MIN_PX, CONNECT_SEPARATION_MAX_PX);
  const wantX = sepMax / 2 + bw + size + CONNECT_ARROW_GAP_PX;
  const wantY = Math.max(hh / 2 + guide, size / 2);
  const k = Math.min(
    centreFit(wantX, centreRoom(cx, edges.x1, 1)),
    centreFit(wantX, centreRoom(cx, edges.x0, -1)),
    centreFit(wantY, centreRoom(cy, edges.y1, 1)),
    centreFit(wantY, centreRoom(cy, edges.y0, -1))
  );
  return {
    cx,
    cy,
    bw: bw * k,
    pin: pin * k,
    hh: hh * k,
    guide: guide * k,
    sepMax: sepMax * k,
    size: size * k,
    gap: CONNECT_ARROW_GAP_PX * k,
  };
}

/**
 * bend's lead: it leaves the box's edge on the side with room, runs `stub` px to a hinge, and its `leg` swings
 * about that hinge from the straight continuation (the dashed guide) through `BEND_SWEEP_DEG` toward the side with
 * room; the arrowhead leads at the free end and the small arc at the hinge is the angle bent. The fit `k` keeps
 * the straightest pose (the one that reaches furthest) inside the room beyond the box edge.
 */
function bendGeometry(box: PxBox, edges: FrameEdges) {
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const cx = box.left + w / 2;
  const cy = box.top + h / 2;
  const sy: 1 | -1 = roomToward(box, 0, -1, edges) > roomToward(box, 0, 1, edges) ? -1 : 1;
  const sx: 1 | -1 = roomToward(box, -1, 0, edges) > roomToward(box, 1, 0, edges) ? -1 : 1;
  const size = arrowSize(box);
  const stub = clamp(BEND_STUB_FRACTION * h, BEND_STUB_MIN_PX, BEND_STUB_MAX_PX);
  const leg = clamp(BEND_LEG_FRACTION * Math.min(w, h), BEND_LEG_MIN_PX, BEND_LEG_MAX_PX);
  const arcR = clamp(BEND_ARC_FRACTION * leg, BEND_ARC_MIN_PX, BEND_ARC_MAX_PX);
  const joint = clamp(BEND_JOINT_FRACTION * leg, BEND_JOINT_MIN_PX, BEND_JOINT_MAX_PX);
  const outRoom = sy > 0 ? centreRoom(cy, edges.y1, 1) : centreRoom(cy, edges.y0, -1);
  const sideRoom = sx > 0 ? centreRoom(cx, edges.x1, 1) : centreRoom(cx, edges.x0, -1);
  // The hinge's dot and the arrowhead's arms reach back past the centre line: fit that too, against what the
  // far side of the box leaves (nothing when the box is flush with that frame edge).
  const back = Math.max(joint, 0.6 * size);
  const backRoom = Math.max(0, w / 2 + (sx > 0 ? centreRoom(cx, edges.x0, -1) : centreRoom(cx, edges.x1, 1)));
  const k = Math.min(
    edgeFit(stub + leg + size, outRoom, h / 2),
    edgeFit(leg + size, sideRoom, w / 2),
    centreFit(back, backRoom)
  );
  return {
    cx,
    cy,
    sy,
    sx,
    hingeY: (sy > 0 ? box.top + h : box.top) + sy * stub * k,
    stub: stub * k,
    leg: leg * k,
    arcR: arcR * k,
    joint: joint * k,
    size: size * k,
  };
}

/**
 * The shapes of `action` on `box` at `frame`, kept inside `viewport` where the action has a direction to clamp
 * (move, insert, pull, push), a side to choose (fold, flip) or a destination (place, fit; `target`, else the
 * provisional one). For one box and viewport each action returns the same shape kinds in the same order for every
 * progress (stable pool assignment); a shape that should not show yet has opacity 0.
 */
export function motionShapes(
  action: ActionType,
  direction: ActionDirection,
  box: PxBox,
  frame: MotionFrame,
  viewport?: FrameSize,
  target?: PxBox
): MotionShape[] {
  const edges = frameEdges(viewport);
  const p = clamp(Number.isFinite(frame.progress) ? frame.progress : 1, 0, 1);
  const a = clamp(Number.isFinite(frame.alpha) ? frame.alpha : 1, 0, 1);
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const cx = box.left + w / 2;
  const cy = box.top + h / 2;
  const op = (value: number) => clamp(value * a, 0, 1);

  switch (action) {
    case 'grasp': {
      const arm = clamp(GRASP_ARM_FRACTION * w, GRASP_ARM_MIN_PX, GRASP_ARM_MAX_PX);
      const gap = lerp(GRASP_START_FRACTION * w + GRASP_START_EXTRA_PX, GRASP_END_GAP_PX, p);
      const y0 = cy - (GRASP_HEIGHT_FRACTION * h) / 2;
      const y1 = cy + (GRASP_HEIGHT_FRACTION * h) / 2;
      // Each bracket opens toward the box; its arm tips are `gap` px outside the edge.
      const lTip = box.left - gap;
      const lSpine = lTip - arm;
      const rTip = box.left + w + gap;
      const rSpine = rTip + arm;
      return [
        { kind: 'path', d: `M ${n(lTip)} ${n(y0)} L ${n(lSpine)} ${n(y0)} L ${n(lSpine)} ${n(y1)} L ${n(lTip)} ${n(y1)}`, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: `M ${n(rTip)} ${n(y0)} L ${n(rSpine)} ${n(y0)} L ${n(rSpine)} ${n(y1)} L ${n(rTip)} ${n(y1)}`, opacity: op(LINE_OPACITY) },
      ];
    }
    case 'remove': {
      // No direction (the plan's remove is direction "none"): the ghost grows about the box centre and fades,
      // with four arrowheads leaving its corners diagonally — "pull it off", not "lift it up".
      const gx = clamp(REMOVE_GROW_FRACTION * w, REMOVE_GROW_MIN_PX, REMOVE_GROW_MAX_PX) * p;
      const gy = clamp(REMOVE_GROW_FRACTION * h, REMOVE_GROW_MIN_PX, REMOVE_GROW_MAX_PX) * p;
      const ghost = { x: box.left - gx, y: box.top - gy, width: w + 2 * gx, height: h + 2 * gy };
      const size = arrowSize(box) * lerp(0.4, 1, p);
      const headOp = op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1));
      const shapes: MotionShape[] = [{ kind: 'rect', ...ghost, opacity: op(lerp(GHOST_OPACITY, 0.35, p)), dashed: true }];
      for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const) {
        const cornerX = sx < 0 ? ghost.x : ghost.x + ghost.width;
        const cornerY = sy < 0 ? ghost.y : ghost.y + ghost.height;
        const reach = REMOVE_CORNER_GAP_PX + size;
        shapes.push({
          kind: 'path',
          d: arrowHead(cornerX + (sx * reach) / Math.SQRT2, cornerY + (sy * reach) / Math.SQRT2, sx, sy, size),
          opacity: headOp,
          filled: true,
        });
      }
      return shapes;
    }
    case 'move': {
      const [dx, dy] = DIRECTION_VECTORS[direction] ?? [1, 0];
      const side = dx !== 0 ? w : h;
      const room = roomToward(box, dx, dy, edges);
      const size = arrowSize(box);
      const lineOp = op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1));
      if (room < EDGE_MODE_ROOM_PX) {
        // No room for a ghost: a trail from the box centre toward that frame edge, the arrowhead just inside it.
        const lateral = dx !== 0
          ? clamp(cy, edges.y0 + size, edges.y1 - size)
          : clamp(cx, edges.x0 + size, edges.x1 - size);
        const centre = dx !== 0 ? cx : cy;
        const edge = dx > 0 ? edges.x1 - EDGE_TIP_INSET_PX
          : dx < 0 ? edges.x0 + EDGE_TIP_INSET_PX
          : dy > 0 ? edges.y1 - EDGE_TIP_INSET_PX
          : edges.y0 + EDGE_TIP_INSET_PX;
        const sign = dx !== 0 ? dx : dy;
        // The tip runs from one arrow length past the centre to the edge (never behind the centre).
        const start = sign > 0 ? Math.min(centre + size, edge) : Math.max(centre - size, edge);
        const tip = lerp(start, edge, p);
        const base = sign > 0 ? Math.max(centre, tip - size) : Math.min(centre, tip + size);
        const pt = (along: number) => (dx !== 0 ? `${n(along)} ${n(lateral)}` : `${n(lateral)} ${n(along)}`);
        const tipX = dx !== 0 ? tip : lateral;
        const tipY = dx !== 0 ? lateral : tip;
        return [
          { kind: 'path', d: `M ${pt(centre)} L ${pt(base)}`, opacity: lineOp },
          { kind: 'path', d: arrowHead(tipX, tipY, dx, dy, size), opacity: lineOp, filled: true },
        ];
      }
      const travel = clamp(side, MIN_TRAVEL_PX, room);
      const ghost = { x: box.left + dx * travel * p, y: box.top + dy * travel * p, width: w, height: h };
      // The trail starts at the box's edge centre on that side; the arrowhead's tip leads at the ghost's leading
      // edge (which ends `FRAME_INSET_PX` inside the frame at most).
      const ex = cx + (dx * w) / 2;
      const ey = cy + (dy * h) / 2;
      const tx = ex + dx * travel * p;
      const ty = ey + dy * travel * p;
      const back = Math.max(0, travel * p - size);
      return [
        { kind: 'rect', ...ghost, opacity: op(lerp(GHOST_OPACITY, 0.5, p)), dashed: true },
        { kind: 'path', d: `M ${n(ex)} ${n(ey)} L ${n(ex + dx * back)} ${n(ey + dy * back)}`, opacity: lineOp },
        { kind: 'path', d: arrowHead(tx, ty, dx, dy, size), opacity: lineOp, filled: true },
      ];
    }
    case 'rotate': {
      const sign = direction === 'counterclockwise' ? -1 : 1;
      // Half the diagonal: the arc clears the box's corners, so it never crosses the object.
      const r = 0.5 * Math.hypot(w, h) + ROTATE_RADIUS_EXTRA_PX;
      const start = -Math.PI / 2;
      const sweep = (ROTATE_SWEEP_DEG * p * Math.PI) / 180;
      const segments = Math.max(1, Math.ceil((ROTATE_SWEEP_DEG * p) / ROTATE_SEGMENT_DEG));
      let d = '';
      for (let i = 0; i <= segments; i += 1) {
        const theta = start + sign * sweep * (i / segments);
        d += `${i === 0 ? 'M' : ' L'} ${n(cx + r * Math.cos(theta))} ${n(cy + r * Math.sin(theta))}`;
      }
      const tip = start + sign * sweep;
      const tipX = cx + r * Math.cos(tip);
      const tipY = cy + r * Math.sin(tip);
      // Tangent along the motion: screen y points down, so increasing theta is clockwise.
      const ux = -Math.sin(tip) * sign;
      const uy = Math.cos(tip) * sign;
      const size = arrowSize(box);
      return [
        { kind: 'path', d, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: arrowHead(tipX + ux * size * 0.5, tipY + uy * size * 0.5, ux, uy, size), opacity: op(LINE_OPACITY), filled: true },
      ];
    }
    case 'press': {
      const { rEnd, rStart } = pressRadii(w, h);
      return [
        { kind: 'circle', cx, cy, r: lerp(rStart, rEnd, p), opacity: op(lerp(0.3, 0.9, p)) },
        { kind: 'circle', cx, cy, r: PRESS_DOT_R_PX, opacity: p > PRESS_DOT_FROM ? op(LINE_OPACITY) : 0, filled: true },
      ];
    }
    case 'open':
    case 'close': {
      // open: the lid (top half) lifts and narrows; close: it starts lifted and comes down to join. Both halves
      // are open U shapes (the lid has no bottom edge, the body no top edge), so the dashed seam stays visible.
      const t = action === 'open' ? p : 1 - p;
      const half = h / 2;
      const lidWidth = w * lerp(1, LID_WIDTH_SCALE, t);
      const lx0 = cx - lidWidth / 2;
      const lx1 = cx + lidWidth / 2;
      const ly = box.top - Math.max(LID_LIFT_FRACTION * h, LID_LIFT_MIN_PX) * t;
      const seamY = box.top + half;
      const bottom = box.top + h;
      const right = box.left + w;
      const shapes: MotionShape[] = [
        { kind: 'path', d: `M ${n(lx0)} ${n(ly + half)} L ${n(lx0)} ${n(ly)} L ${n(lx1)} ${n(ly)} L ${n(lx1)} ${n(ly + half)}`, opacity: op(GHOST_OPACITY) },
        { kind: 'path', d: `M ${n(box.left)} ${n(seamY)} L ${n(box.left)} ${n(bottom)} L ${n(right)} ${n(bottom)} L ${n(right)} ${n(seamY)}`, opacity: op(GHOST_OPACITY) },
        { kind: 'path', d: `M ${n(box.left - SEAM_OVERHANG_PX)} ${n(seamY)} L ${n(right + SEAM_OVERHANG_PX)} ${n(seamY)}`, opacity: op(LINE_OPACITY), dashed: true },
      ];
      if (action === 'close') {
        // Two downward arrowheads just outside the seam's ends: the closing movement, also in the end pose.
        const size = arrowSize(box);
        const headOp = op(LINE_OPACITY * p);
        const offset = SEAM_OVERHANG_PX + 2 + size * 0.6;
        shapes.push(
          { kind: 'path', d: arrowHead(box.left - offset, seamY, 0, 1, size), opacity: headOp, filled: true },
          { kind: 'path', d: arrowHead(right + offset, seamY, 0, 1, size), opacity: headOp, filled: true }
        );
      }
      return shapes;
    }
    case 'insert': {
      const scale = lerp(INSERT_START_SCALE, INSERT_END_SCALE, p);
      // The start sits `travel` above the box centre: 1.1·h, at least MIN_TRAVEL_PX, but the starting ghost's top
      // stays FRAME_INSET_PX inside the frame (0 when there is no room: the ghost then only shrinks in place).
      const size = arrowSize(box);
      const startTopRoom = cy - (INSERT_START_SCALE * h) / 2 - (edges.y0 + FRAME_INSET_PX + size);
      const travel = Math.max(0, clamp(INSERT_TRAVEL_FRACTION * h, MIN_TRAVEL_PX, startTopRoom));
      const gcy = lerp(cy - travel, cy, p);
      const ghost = scaledRect(cx, gcy, w, h, scale);
      // A short trail behind (above) the ghost, its arrowhead touching the ghost's top edge, pointing down (pushed
      // down into the ghost when the frame top leaves no room); the trail is cut at the frame's top inset.
      const trail = clamp(INSERT_TRAIL_FRACTION * h, INSERT_TRAIL_MIN_PX, INSERT_TRAIL_MAX_PX);
      const tipY = Math.max(ghost.y, edges.y0 + FRAME_INSET_PX + size);
      const trailEnd = tipY - size;
      const trailStart = Math.min(trailEnd, Math.max(edges.y0 + FRAME_INSET_PX, trailEnd - trail));
      return [
        { kind: 'rect', ...ghost, opacity: op(lerp(GHOST_OPACITY, 0.5, p)), dashed: true },
        { kind: 'path', d: `M ${n(cx)} ${n(trailStart)} L ${n(cx)} ${n(trailEnd)}`, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: arrowHead(cx, tipY, 0, 1, size), opacity: op(LINE_OPACITY), filled: true },
      ];
    }
    case 'attach': {
      // A tape band sweeps across the box's middle from left to right, its arrowhead leading.
      // The band's ends overhang the box by 12 px, cut at the frame's inset (the leading arrowhead included).
      const band = clamp(ATTACH_BAND_FRACTION * h, ATTACH_BAND_MIN_PX, ATTACH_BAND_MAX_PX);
      const size = arrowSize(box);
      const x0 = Math.max(box.left - ATTACH_OVERHANG_PX, edges.x0 + FRAME_INSET_PX);
      const x1 = Math.max(x0, Math.min(box.left + w + ATTACH_OVERHANG_PX, edges.x1 - FRAME_INSET_PX - size));
      const xe = lerp(x0, x1, p);
      // The band's line: the box's middle, kept far enough from the frame's top/bottom for the band and the head.
      const halfLane = Math.max(band / 2, 0.6 * size);
      const y = clamp(cy, edges.y0 + halfLane, edges.y1 - halfLane);
      return [
        { kind: 'path', d: `M ${n(x0)} ${n(y)} L ${n(x1)} ${n(y)}`, opacity: op(ATTACH_BAND_OPACITY), dashed: true },
        { kind: 'rect', x: x0, y: y - band / 2, width: Math.max(0, xe - x0), height: band, opacity: op(ATTACH_BAND_OPACITY), filled: true },
        { kind: 'path', d: arrowHead(xe + size, y, 1, 0, size), opacity: op(LINE_OPACITY), filled: true },
      ];
    }
    case 'fold': {
      // A flap standing on the top edge (the hinge) folds down over the box — or, at the frame top, hangs from
      // the bottom edge and folds up; a half-ellipse arc beside the box follows the fold with its arrowhead.
      const g = foldGeometry(box, edges);
      const hingeY = g.hingeY;
      const tipY = hingeY + g.sy * g.flapH * Math.cos(Math.PI * p);
      const inset = w * FOLD_FLAP_INSET_FRACTION;
      const pt = (th: number): [number, number] => [g.ax + g.sx * g.rx * Math.cos(th), g.ay - g.sy * g.ry * Math.sin(th)];
      const th = -Math.PI / 2 + Math.PI * p;
      const [tx, ty] = pt(th);
      const ux = -g.sx * g.rx * Math.sin(th);
      const uy = -g.sy * g.ry * Math.cos(th);
      const ul = Math.hypot(ux, uy) || 1;
      const lineOp = op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1));
      return [
        { kind: 'path', d: seamPath(box, hingeY, edges), opacity: op(LINE_OPACITY), dashed: true },
        { kind: 'path', d: `M ${n(box.left + inset)} ${n(hingeY)} L ${n(box.left + inset)} ${n(tipY)} L ${n(box.left + w - inset)} ${n(tipY)} L ${n(box.left + w - inset)} ${n(hingeY)}`, opacity: op(GHOST_OPACITY), dashed: true },
        { kind: 'path', d: polyline(Math.max(1, Math.ceil(FOLD_ARC_SEGMENTS * p)), (u) => pt(-Math.PI / 2 + Math.PI * p * u)), opacity: lineOp },
        { kind: 'path', d: arrowHead(tx + (ux / ul) * g.size * 0.4, ty + (uy / ul) * g.size * 0.4, ux, uy, g.size), opacity: lineOp, filled: true },
      ];
    }
    case 'place': {
      // The ghost lifts over an arc and comes to rest on the destination; a landing mark shows where.
      const g = placeGeometry(box, edges, viewport, target);
      const [gx, gy] = g.at(p);
      const t = g.t;
      const trail = polyline(Math.max(1, Math.ceil(PLACE_PATH_SEGMENTS * p)), (u) => {
        const [x, y] = g.at(p * u);
        return [x, y - h / 2];
      });
      return [
        { kind: 'rect', x: gx - w / 2, y: gy - h / 2, width: w, height: h, opacity: op(lerp(GHOST_OPACITY, 0.6, p)), dashed: true },
        { kind: 'path', d: trail, opacity: op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1)), dashed: true },
        {
          kind: 'path',
          d: `M ${n(t.left)} ${n(g.surface - PLACE_MARK_TICK_PX)} L ${n(t.left)} ${n(g.surface - 2)} L ${n(t.left + t.width)} ${n(g.surface - 2)} L ${n(t.left + t.width)} ${n(g.surface - PLACE_MARK_TICK_PX)}`,
          opacity: op(LINE_OPACITY * clamp((p - 0.5) / 0.5, 0, 1)),
        },
      ];
    }
    case 'fit': {
      // The ghost slides up against the destination; a dashed junction line with two arrowheads meeting at it.
      const t = target ?? provisionalTarget('fit', box, viewport);
      const toRight = t.left >= box.left;
      // The end ghost touches the destination; a caller's destination too near the edge for that leaves the
      // ghost inside the frame instead (the provisional one always leaves room when the pair fits).
      const endLeft = clamp(toRight ? t.left - w : t.left + t.width, edges.x0 + FRAME_INSET_PX, edges.x1 - FRAME_INSET_PX - w);
      const endTop = clamp(t.top + t.height / 2 - h / 2, edges.y0 + FRAME_INSET_PX, edges.y1 - FRAME_INSET_PX - h);
      const jx = toRight ? endLeft + w : endLeft;
      const size = arrowSize(box);
      const topY = Math.max(Math.min(t.top, endTop, box.top) - size - 6, edges.y0 + FRAME_INSET_PX + 0.6 * size);
      const botY = Math.max(topY, Math.min(Math.max(t.top + t.height, endTop + h) + 6, edges.y1 - FRAME_INSET_PX));
      const gap = size * 0.5 + 3;
      // The two heads meet at the junction, moved in when it is within one head of the frame edge.
      const hx = clamp(jx, edges.x0 + gap + size, edges.x1 - gap - size);
      const headOp = op(LINE_OPACITY * clamp((p - 0.5) / 0.5, 0, 1));
      return [
        { kind: 'rect', x: lerp(box.left, endLeft, p), y: lerp(box.top, endTop, p), width: w, height: h, opacity: op(lerp(GHOST_OPACITY, 0.6, p)), dashed: true },
        { kind: 'path', d: `M ${n(jx)} ${n(topY)} L ${n(jx)} ${n(botY)}`, opacity: op(0.6), dashed: true },
        { kind: 'path', d: arrowHead(hx - gap, topY, 1, 0, size), opacity: headOp, filled: true },
        { kind: 'path', d: arrowHead(hx + gap, topY, -1, 0, size), opacity: headOp, filled: true },
      ];
    }
    case 'screw': {
      // A screw head with a turning cross; an arc with an arrowhead runs 1.5 turns around it (tighten =
      // clockwise on screen, loosen = counterclockwise).
      const sign = direction === 'loosen' ? -1 : 1;
      const g = screwGeometry(box);
      const spin = sign * p * SCREW_TURNS * 2 * Math.PI;
      const tipTh = spin - Math.PI / 2;
      const arc = polyline(SCREW_ARC_SEGMENTS, (u) => {
        const th = tipTh - sign * SCREW_ARC_RAD * (1 - u);
        return [cx + g.r * Math.cos(th), cy + g.r * Math.sin(th)];
      });
      const ux = -Math.sin(tipTh) * sign;
      const uy = Math.cos(tipTh) * sign;
      const sl = g.rHead * 0.6;
      const c1 = Math.cos(spin) * sl;
      const s1 = Math.sin(spin) * sl;
      const tipX = cx + g.r * Math.cos(tipTh) + ux * g.size * 0.5;
      const tipY = cy + g.r * Math.sin(tipTh) + uy * g.size * 0.5;
      return [
        { kind: 'circle', cx, cy, r: g.rHead, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: `M ${n(cx - c1)} ${n(cy - s1)} L ${n(cx + c1)} ${n(cy + s1)} M ${n(cx + s1)} ${n(cy - c1)} L ${n(cx - s1)} ${n(cy + c1)}`, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: arc, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: arrowHead(tipX, tipY, ux, uy, g.size), opacity: op(LINE_OPACITY), filled: true },
      ];
    }
    case 'hold': {
      // press, then keep pressing: the ring contracts in the first 35 %, then a progress ring closes around it.
      const { rEnd, rStart } = pressRadii(w, h);
      const q = clamp(p / HOLD_PRESS_FRACTION, 0, 1);
      const fillQ = clamp((p - HOLD_PRESS_FRACTION) / (1 - HOLD_PRESS_FRACTION), 0, 1);
      const rRing = rEnd + HOLD_RING_GAP_PX;
      const ring = polyline(Math.max(1, Math.ceil(HOLD_RING_SEGMENTS * fillQ)), (u) => {
        const th = -Math.PI / 2 + 2 * Math.PI * fillQ * u;
        return [cx + rRing * Math.cos(th), cy + rRing * Math.sin(th)];
      });
      return [
        { kind: 'circle', cx, cy, r: lerp(rStart, rEnd, q), opacity: op(lerp(0.3, 0.9, q)) },
        { kind: 'circle', cx, cy, r: PRESS_DOT_R_PX, opacity: q >= 1 ? op(LINE_OPACITY) : 0, filled: true },
        { kind: 'path', d: ring, opacity: fillQ > 0 ? op(LINE_OPACITY) : 0 },
      ];
    }
    case 'flip': {
      // The ghost squashes to a line and opens again (turned over) about the dashed middle line; an arc runs
      // down the side with room.
      const g = flipGeometry(box, edges);
      const gh = Math.max(FLIP_MIN_HEIGHT_PX, h * Math.abs(Math.cos(Math.PI * p)));
      const pt = (th: number): [number, number] => [g.ax + g.sx * g.rx * Math.cos(th), g.ay + g.ry * Math.sin(th)];
      const th = -Math.PI / 2 + Math.PI * p;
      const [tx, ty] = pt(th);
      const ux = -g.sx * g.rx * Math.sin(th);
      const uy = g.ry * Math.cos(th);
      const ul = Math.hypot(ux, uy) || 1;
      const lineOp = op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1));
      return [
        { kind: 'path', d: seamPath(box, cy, edges), opacity: op(0.6), dashed: true },
        { kind: 'rect', x: box.left, y: cy - gh / 2, width: w, height: gh, opacity: op(GHOST_OPACITY * 0.8), dashed: true },
        { kind: 'path', d: polyline(Math.max(1, Math.ceil(FLIP_ARC_SEGMENTS * p)), (u) => pt(-Math.PI / 2 + Math.PI * p * u)), opacity: lineOp },
        { kind: 'path', d: arrowHead(tx + (ux / ul) * g.size * 0.4, ty + (uy / ul) * g.size * 0.4, ux, uy, g.size), opacity: lineOp, filled: true },
      ];
    }
    case 'pull':
    case 'push': {
      const [dx, dy] = DIRECTION_VECTORS[direction] ?? (action === 'pull' ? [0, 1] : [0, -1]);
      const side = dx !== 0 ? w : h;
      const half = Math.max(PUSH_PULL_HALF_FRACTION * (dx !== 0 ? h : w), PUSH_PULL_HALF_MIN_PX);
      const px = -dy;
      const py = dx;
      const size = arrowSize(box);
      const shiftMax = clamp(PUSH_PULL_SHIFT_FRACTION * side, PUSH_PULL_SHIFT_MIN_PX, PUSH_PULL_SHIFT_MAX_PX);
      const front = Math.max(0, roomToward(box, dx, dy, edges));
      const lineOp = op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1));
      const P = (x: number, y: number) => `${n(x)} ${n(y)}`;
      // Across the motion: the hook/bar (±half) and the arrowheads (±0.6 arrow) keep inside the frame edges.
      const across = Math.max(half, 0.6 * size);
      const lateral = dx !== 0 ? clamp(cy, edges.y0 + across, edges.y1 - across) : clamp(cx, edges.x0 + across, edges.x1 - across);
      const onAxis = (along: number): [number, number] => (dx !== 0 ? [along, lateral] : [lateral, along]);
      if (action === 'push') {
        // The ghost is pushed `shift` px the given way (never past the frame); behind it, a shaft and an
        // arrowhead push on the bar at its trailing edge. Behind the box: a 4 px gap, the arrowhead, then up to
        // 2.5 arrow lengths of shaft — shortened to the room there, the head moved onto the box when even it
        // does not fit.
        const shift = Math.min(shiftMax, front) * p;
        const behind = Math.max(0, roomToward(box, -dx, -dy, edges));
        const len = clamp(behind - PUSH_PULL_GAP_PX - size, 0, PUSH_SHAFT_ARROWS * size);
        const intrude = Math.max(0, PUSH_PULL_GAP_PX + size - behind);
        const ghost: MotionShape = { kind: 'rect', x: box.left + dx * shift, y: box.top + dy * shift, width: w, height: h, opacity: op(lerp(GHOST_OPACITY, 0.5, p)), dashed: true };
        const [ex, ey] = onAxis(dx !== 0 ? cx - (dx * w) / 2 + dx * shift : cy - (dy * h) / 2 + dy * shift);
        const tipX = ex - dx * (PUSH_PULL_GAP_PX - intrude);
        const tipY = ey - dy * (PUSH_PULL_GAP_PX - intrude);
        return [
          ghost,
          { kind: 'path', d: `M ${P(tipX - dx * (size + len), tipY - dy * (size + len))} L ${P(tipX - dx * size, tipY - dy * size)}`, opacity: op(LINE_OPACITY) },
          { kind: 'path', d: arrowHead(tipX, tipY, dx, dy, size), opacity: op(LINE_OPACITY), filled: true },
          { kind: 'path', d: `M ${P(ex + px * half, ey + py * half)} L ${P(ex - px * half, ey - py * half)}`, opacity: lineOp },
        ];
      }
      // pull: a hook on the leading edge, a line and an arrowhead ahead of it. The room ahead holds, in order of
      // priority, the hook and the arrowhead (pulled back onto the box when they do not fit), then the line (up to
      // 2 arrow lengths + 8 px), then the ghost's shift.
      const arm = clamp(PULL_ARM_FRACTION * side, PULL_ARM_MIN_PX, PULL_ARM_MAX_PX);
      const spare = front - (PUSH_PULL_GAP_PX + arm + size);
      const len = clamp(spare - shiftMax, 0, PULL_LINE_ARROWS * size + PULL_LINE_EXTRA_PX);
      const shift = clamp(spare - len, 0, shiftMax) * p;
      const back = Math.max(0, -spare);
      const [lx, ly] = onAxis(dx !== 0 ? cx + (dx * w) / 2 + dx * shift : cy + (dy * h) / 2 + dy * shift);
      const bx = lx + dx * (PUSH_PULL_GAP_PX - back);
      const by = ly + dy * (PUSH_PULL_GAP_PX - back);
      const ox = bx + dx * arm;
      const oy = by + dy * arm;
      return [
        { kind: 'rect', x: box.left + dx * shift, y: box.top + dy * shift, width: w, height: h, opacity: op(lerp(GHOST_OPACITY, 0.5, p)), dashed: true },
        { kind: 'path', d: `M ${P(bx + px * half, by + py * half)} L ${P(ox + px * half, oy + py * half)} L ${P(ox - px * half, oy - py * half)} L ${P(bx - px * half, by - py * half)}`, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: `M ${P(ox, oy)} L ${P(ox + dx * len, oy + dy * len)}`, opacity: lineOp },
        { kind: 'path', d: arrowHead(ox + dx * (len + size), oy + dy * (len + size), dx, dy, size), opacity: lineOp, filled: true },
      ];
    }
    case 'align': {
      // The component (a ghost of half the box) starts `travel` px right of the dashed guide through the box's
      // middle — ticked at both ends — and settles centred on it; the arrowhead leads at its leading edge.
      const g = alignGeometry(box, edges);
      const gx = g.cx + g.travel * (1 - p);
      const gy = g.cy;
      const left = gx - g.gw / 2;
      const y0 = gy - g.gh / 2 - g.over;
      const y1 = gy + g.gh / 2 + g.over;
      return [
        { kind: 'path', d: `M ${n(g.cx - g.tick)} ${n(y0)} L ${n(g.cx + g.tick)} ${n(y0)} M ${n(g.cx)} ${n(y0)} L ${n(g.cx)} ${n(y1)} M ${n(g.cx - g.tick)} ${n(y1)} L ${n(g.cx + g.tick)} ${n(y1)}`, opacity: op(LINE_OPACITY), dashed: true },
        { kind: 'rect', x: left, y: gy - g.gh / 2, width: g.gw, height: g.gh, opacity: op(lerp(GHOST_OPACITY, 0.6, p)), dashed: true },
        { kind: 'path', d: `M ${n(left)} ${n(gy)} L ${n(g.cx)} ${n(gy)}`, opacity: op(LINE_OPACITY) },
        { kind: 'path', d: arrowHead(left, gy, -1, 0, g.size), opacity: op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1)), filled: true },
      ];
    }
    case 'connect':
    case 'disconnect': {
      // Two connector halves whose pins meet on the box's middle line: connect brings them together there,
      // disconnect pulls them apart from it. The arrowheads are behind the halves pushing in (connect) or ahead of
      // them pulling out (disconnect), so the held end pose still says which of the two happened.
      const g = connectGeometry(box, edges);
      const mated = 2 * g.pin;
      const sep = lerp(action === 'connect' ? g.sepMax : mated, action === 'connect' ? mated : g.sepMax, p);
      const y0 = g.cy - g.hh / 2;
      const y1 = g.cy + g.hh / 2;
      const leftIn = g.cx - sep / 2;
      const rightIn = g.cx + sep / 2;
      const leftOut = leftIn - g.bw;
      const rightOut = rightIn + g.bw;
      const half = (outer: number, inner: number, pinTo: number) =>
        `M ${n(outer)} ${n(y0)} L ${n(inner)} ${n(y0)} L ${n(inner)} ${n(y1)} L ${n(outer)} ${n(y1)} L ${n(outer)} ${n(y0)} M ${n(inner)} ${n(g.cy)} L ${n(pinTo)} ${n(g.cy)}`;
      const heads =
        action === 'connect'
          ? `${arrowHead(leftOut, g.cy, 1, 0, g.size)} ${arrowHead(rightOut, g.cy, -1, 0, g.size)}`
          : `${arrowHead(leftOut - g.gap - g.size, g.cy, -1, 0, g.size)} ${arrowHead(rightOut + g.gap + g.size, g.cy, 1, 0, g.size)}`;
      return [
        { kind: 'path', d: half(leftOut, leftIn, leftIn + g.pin), opacity: op(LINE_OPACITY) },
        { kind: 'path', d: half(rightOut, rightIn, rightIn - g.pin), opacity: op(LINE_OPACITY) },
        { kind: 'path', d: `M ${n(g.cx)} ${n(y0 - g.guide)} L ${n(g.cx)} ${n(y1 + g.guide)}`, opacity: op(LINE_OPACITY), dashed: true },
        { kind: 'path', d: heads, opacity: op(LINE_OPACITY * clamp(p / TRAIL_FADE_PROGRESS, 0, 1)), filled: true },
      ];
    }
    case 'bend': {
      // The lead leaves the box on the side with room, runs to a hinge and its leg swings about that hinge from the
      // straight continuation (the dashed guide) through BEND_SWEEP_DEG; the arrowhead leads at the free end and
      // the small arc at the hinge is the angle bent.
      const g = bendGeometry(box, edges);
      const entryY = g.sy > 0 ? box.top + h : box.top;
      const phi = ((90 - BEND_SWEEP_DEG * p) * Math.PI) / 180;
      const ux = g.sx * Math.cos(phi);
      const uy = g.sy * Math.sin(phi);
      const tipX = g.cx + g.leg * ux;
      const tipY = g.hingeY + g.leg * uy;
      const arc = polyline(BEND_ARC_SEGMENTS, (u) => {
        const th = lerp(Math.PI / 2, phi, u);
        return [g.cx + g.sx * g.arcR * Math.cos(th), g.hingeY + g.sy * g.arcR * Math.sin(th)];
      });
      return [
        { kind: 'path', d: `M ${n(g.cx)} ${n(entryY)} L ${n(g.cx)} ${n(g.hingeY)} L ${n(tipX)} ${n(tipY)}`, opacity: op(GHOST_OPACITY) },
        { kind: 'path', d: `M ${n(g.cx)} ${n(g.hingeY)} L ${n(g.cx)} ${n(g.hingeY + g.sy * g.leg)}`, opacity: op(LINE_OPACITY), dashed: true },
        { kind: 'path', d: arc, opacity: op(LINE_OPACITY) },
        { kind: 'circle', cx: g.cx, cy: g.hingeY, r: g.joint, opacity: op(LINE_OPACITY), filled: true },
        { kind: 'path', d: arrowHead(tipX, tipY, ux, uy, g.size), opacity: op(LINE_OPACITY), filled: true },
      ];
    }
    default:
      return [];
  }
}

const EXTENT_SAMPLES = [0, 0.25, 0.5, 0.75, 1];

type Bounds = { minX: number; minY: number; maxX: number; maxY: number };

function shapeBounds(shape: MotionShape): Bounds | null {
  if (shape.kind === 'rect') return { minX: shape.x, minY: shape.y, maxX: shape.x + shape.width, maxY: shape.y + shape.height };
  if (shape.kind === 'circle') return { minX: shape.cx - shape.r, minY: shape.cy - shape.r, maxX: shape.cx + shape.r, maxY: shape.cy + shape.r };
  const nums = (shape.d.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
  if (nums.length < 2) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i + 1 < nums.length; i += 2) {
    minX = Math.min(minX, nums[i]);
    maxX = Math.max(maxX, nums[i]);
    minY = Math.min(minY, nums[i + 1]);
    maxY = Math.max(maxY, nums[i + 1]);
  }
  return { minX, minY, maxX, maxY };
}

/**
 * The four schematic glyphs (added 2026-10-05): they mark the tracked box alone — a schematic guide, two connector
 * halves or a lead — and never a pin, polarity, voltage or measured target position. The overlay's callout says so
 * next to the label, and their geometry depends on the room the frame leaves around the box, so they are never
 * cached (they belong to `FRAME_DEPENDENT` below).
 */
export const SCHEMATIC_ACTIONS: Partial<Record<ActionType, true>> = {
  align: true,
  connect: true,
  disconnect: true,
  bend: true,
};

/** Whether `action` is one of the schematic glyphs (marks the box, not a real pin or measured position). */
export function isSchematicAction(action: ActionType): boolean {
  return SCHEMATIC_ACTIONS[action] === true;
}

/**
 * Actions whose geometry depends on the box's position in the frame (they clamp to it, pick a side by the room
 * there, or aim at a destination placed by it): never cached.
 */
const FRAME_DEPENDENT: Partial<Record<ActionType, true>> = {
  move: true,
  insert: true,
  attach: true,
  place: true,
  fit: true,
  pull: true,
  push: true,
  fold: true,
  flip: true,
  ...SCHEMATIC_ACTIONS,
};


/** A half ellipse (centre (ax, ay), radii rx/ry, on side sx) from its top through its far side to its bottom. */
function halfEllipseBounds(ax: number, ay: number, rx: number, ry: number, sx: 1 | -1, reach: number): Bounds {
  const far = ax + sx * rx;
  return { minX: Math.min(ax, far) - reach, maxX: Math.max(ax, far) + reach, minY: ay - ry - reach, maxY: ay + ry + reach };
}

/**
 * Analytic bounds of what an action draws along a curve (arc + arrowhead, a lifted path), for the actions whose
 * vertices are NOT linear in progress (sampling can miss a mid-motion extreme): the whole curve's bounds, grown
 * by `CURVE_HEAD_REACH` arrow lengths where an arrowhead rides on it. Null for the other actions.
 */
function curveBounds(action: ActionType, box: PxBox, viewport: FrameSize | undefined, target: PxBox | undefined): Bounds | null {
  const edges = frameEdges(viewport);
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  const cx = box.left + w / 2;
  const cy = box.top + h / 2;
  switch (action) {
    case 'fold': {
      const g = foldGeometry(box, edges);
      // The flap's top runs between hinge − flapH and hinge + flapH (monotonic: the p=0/p=1 samples bound it).
      return halfEllipseBounds(g.ax, g.ay, g.rx, g.ry, g.sx, CURVE_HEAD_REACH * g.size);
    }
    case 'flip': {
      const g = flipGeometry(box, edges);
      return halfEllipseBounds(g.ax, g.ay, g.rx, g.ry, g.sx, CURVE_HEAD_REACH * g.size);
    }
    case 'screw': {
      const g = screwGeometry(box);
      const half = g.r + CURVE_HEAD_REACH * g.size;
      return { minX: cx - half, maxX: cx + half, minY: cy - half, maxY: cy + half };
    }
    case 'hold': {
      const { rEnd, rStart } = pressRadii(w, h);
      const half = Math.max(rStart, rEnd + HOLD_RING_GAP_PX);
      return { minX: cx - half, maxX: cx + half, minY: cy - half, maxY: cy + half };
    }
    case 'place': {
      // y(q) = lerp(cy, landCy, q) − lift·sin(πq) ≥ min(cy, landCy) − lift; sin ≥ 0 never lowers it.
      const g = placeGeometry(box, edges, viewport, target);
      return {
        minX: Math.min(g.cx, g.tcx) - w / 2,
        maxX: Math.max(g.cx, g.tcx) + w / 2,
        minY: Math.min(g.cy, g.landCy) - g.lift - h / 2,
        maxY: Math.max(g.cy, g.landCy) + h / 2,
      };
    }
    case 'bend': {
      // The swinging leg, its turning arrowhead, the angle arc and the straight guide all stay within
      // (leg + 2 × arrow) of the hinge, whatever the swing: a disc around the hinge bounds them.
      const g = bendGeometry(box, edges);
      const reach = g.leg + 2 * g.size;
      return { minX: g.cx - reach, maxX: g.cx + reach, minY: g.hingeY - reach, maxY: g.hingeY + reach };
    }
    default:
      return null;
  }
}

function clampToFrame(rect: PxBox, viewport: FrameSize | undefined): PxBox {
  const e = frameEdges(viewport);
  const left = Math.max(rect.left, e.x0);
  const top = Math.max(rect.top, e.y0);
  const right = Math.min(rect.left + rect.width, e.x1);
  const bottom = Math.min(rect.top + rect.height, e.y1);
  return { left, top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

/**
 * The rect every shape of `action` can occupy over the whole motion, together with the box itself, in CSS px,
 * padded by `MOTION_STROKE_PAD_PX` and clamped to `viewport` when given. The callout is placed off it so it
 * does not cover the animation.
 *
 * - rotate: analytic — the square around the box centre with half side R + the arrowhead size (the arrowhead
 *   sits within one arrow length of the arc at any sweep, which sampling alone misses mid-motion).
 * - fold, flip, screw, hold, place: the 5 samples below plus analytic curve bounds (`curveBounds`): their arcs,
 *   turning arrowheads and lifted path are not linear in progress. Their straight parts (flap, squash, seam,
 *   cross, rings) are monotonic or static, so the p=0/p=1 samples bound them.
 * - every other action moves each vertex linearly in progress (easing only remaps time), so the union of the
 *   shapes at p=0 and p=1 bounds the motion; the intermediate samples are kept as a cross-check.
 */
export function motionExtent(
  action: ActionType,
  direction: ActionDirection,
  box: PxBox,
  viewport?: FrameSize,
  target?: PxBox
): PxBox {
  const w = Math.max(0, box.width);
  const h = Math.max(0, box.height);
  let minX = box.left;
  let minY = box.top;
  let maxX = box.left + w;
  let maxY = box.top + h;
  if (action === 'rotate') {
    const half = 0.5 * Math.hypot(w, h) + ROTATE_RADIUS_EXTRA_PX + arrowSize(box);
    const cx = box.left + w / 2;
    const cy = box.top + h / 2;
    minX = Math.min(minX, cx - half);
    minY = Math.min(minY, cy - half);
    maxX = Math.max(maxX, cx + half);
    maxY = Math.max(maxY, cy + half);
  } else {
    const curve = curveBounds(action, box, viewport, target);
    const bounds = curve ? [curve] : [];
    for (const progress of EXTENT_SAMPLES) {
      for (const shape of motionShapes(action, direction, box, { progress, alpha: 1, cycle: 0, settled: false }, viewport, target)) {
        const b = shapeBounds(shape);
        if (b) bounds.push(b);
      }
    }
    for (const b of bounds) {
      minX = Math.min(minX, b.minX);
      minY = Math.min(minY, b.minY);
      maxX = Math.max(maxX, b.maxX);
      maxY = Math.max(maxY, b.maxY);
    }
  }
  const pad = MOTION_STROKE_PAD_PX;
  return clampToFrame({ left: minX - pad, top: minY - pad, width: maxX - minX + 2 * pad, height: maxY - minY + 2 * pad }, viewport);
}

const extentCache = new Map<string, PxBox>();

/**
 * `motionExtent` for the per-frame path. Position-independent actions are computed once per (action, direction,
 * rounded size) at the origin, translated to the box (widened by `EXTENT_ROUNDING_SLACK_PX` for the rounding)
 * and clamped to the frame; the frame-dependent ones (`FRAME_DEPENDENT`) are computed directly.
 */
export function cachedMotionExtent(
  action: ActionType,
  direction: ActionDirection,
  box: PxBox,
  viewport?: FrameSize,
  target?: PxBox
): PxBox {
  if (FRAME_DEPENDENT[action]) return motionExtent(action, direction, box, viewport, target);
  const rw = Math.round(Math.max(0, box.width));
  const rh = Math.round(Math.max(0, box.height));
  const key = `${action}:${direction}:${rw}:${rh}`;
  let local = extentCache.get(key);
  if (!local) {
    if (extentCache.size >= EXTENT_CACHE_MAX) extentCache.clear();
    local = motionExtent(action, direction, { left: 0, top: 0, width: rw, height: rh });
    extentCache.set(key, local);
  }
  const slack = EXTENT_ROUNDING_SLACK_PX;
  return clampToFrame(
    {
      left: box.left + local.left - slack,
      top: box.top + local.top - slack,
      width: local.width + 2 * slack,
      height: local.height + 2 * slack,
    },
    viewport
  );
}

/** Focus ring corner brackets: length clamp(0.22·min(w, h), 10, 36) px, elbow radius min(10, length / 2). */
const RING_BRACKET_FRACTION = 0.22;
const RING_BRACKET_MIN_PX = 10;
const RING_BRACKET_MAX_PX = 36;
const RING_ELBOW_MAX_PX = 10;

/**
 * The focus ring as four rounded corner brackets around `box` (CSS px of the overlay), one `M … L … Q … L`
 * subpath per corner, clockwise from the top left. Only `M`/`L`/`Q`, every point on the box's edge lines. A side
 * shorter than two bracket lengths (under 20 px) lets the arms cross; the 10 px minimum keeps them visible.
 */
export function ringPath(box: PxBox): string {
  const w = Math.max(0, Number.isFinite(box.width) ? box.width : 0);
  const h = Math.max(0, Number.isFinite(box.height) ? box.height : 0);
  const x0 = Number.isFinite(box.left) ? box.left : 0;
  const y0 = Number.isFinite(box.top) ? box.top : 0;
  const x1 = x0 + w;
  const y1 = y0 + h;
  const len = clamp(RING_BRACKET_FRACTION * Math.min(w, h), RING_BRACKET_MIN_PX, RING_BRACKET_MAX_PX);
  const rc = Math.min(RING_ELBOW_MAX_PX, len / 2);
  const corner = (ax: number, ay: number, sx: number, sy: number) =>
    `M ${n(ax)} ${n(ay + sy * len)} L ${n(ax)} ${n(ay + sy * rc)} Q ${n(ax)} ${n(ay)} ${n(ax + sx * rc)} ${n(ay)} L ${n(ax + sx * len)} ${n(ay)}`;
  return [corner(x0, y0, 1, 1), corner(x1, y0, -1, 1), corner(x1, y1, -1, -1), corner(x0, y1, 1, -1)].join(' ');
}

/**
 * Identity of one motion run: a change restarts the clock — a new guide run, plan revision, step or anchor, or a
 * new binding of the same anchor id (a manual reselection starts a new tracker run / track / generation). A
 * temporary hide of the same binding keeps the key, so the motion resumes.
 */
export function motionKey(parts: {
  runId: number;
  planRevision: number | null;
  stepId: string;
  anchorId: string;
  trackRunId: string;
  trackId: string;
  generation: number;
}): string {
  return JSON.stringify([
    parts.runId,
    parts.planRevision,
    parts.stepId,
    parts.anchorId,
    parts.trackRunId,
    parts.trackId,
    parts.generation,
  ]);
}
