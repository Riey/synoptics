/**
 * Pure geometry and identity for the anchored overlay (design v4 principle 1: position from the tracker,
 * meaning from the LLM).
 *
 * - A plan's commands name the ROLE `target`; once the live store reports `tracking`, the client binds that
 *   role to a real anchor id (`a1`) together with the run/track/generation the store reported.
 * - Geometry is derived per frame from the live box only: the focus ring is the box grown by `pad` (a
 *   fraction of the box, never a position), in the same normalized 0..1 space as the box; the label is
 *   placed in CSS pixels of the overlay (content rect) next to the box — above it, or below/beside it when
 *   the top is taken by the frame edge or by the narration band and the status chips (`placeLabel`) — so
 *   text is never scaled non-uniformly and never covers the box.
 * - An `AnchorRef` for a request is built from the store snapshot read at capture time. The box goes on the
 *   wire only while `state === 'tracking'` (the contract rule).
 *
 * Nothing here imports a value from another module, so `node --test` runs it directly.
 */
import type { Box } from '@visual-coach/visual-tools';
import type {
  ActionAnchorCommand,
  AnchorEcho,
  AnchorRef,
  GuideStep,
} from '../generated/api.generated';
import type { LiveTrackSnapshot } from '../tracking/liveTrackStore';

/** The role every plan command refers to before a track exists. Never a valid anchor id. */
export const TARGET_ROLE = 'target';
/** The one anchor id this demo binds the role to (anchors <= 1). */
export const PRIMARY_ANCHOR_ID = 'a1';
/** The contract's default focus pad (`FocusAnchorCommand.pad`). */
export const DEFAULT_FOCUS_PAD = 0.15;
/** The contract's pad ceiling. */
export const MAX_FOCUS_PAD = 0.5;
/** AnchorRef.label ceiling (characters). */
export const ANCHOR_LABEL_MAX = 40;
/** Screen-space gap between the box edge and the label, in CSS px. */
export const LABEL_OFFSET_PX = 8;

/** The identity a role was bound to: the live store's run/track/generation at the moment of binding. */
export interface AnchorBinding {
  anchorId: string;
  runId: string;
  trackId: string;
  generation: number;
}

export type ActionType = ActionAnchorCommand['action'];
export type ActionDirection = ActionAnchorCommand['direction'];
export type GuideCommand = GuideStep['commands'][number];

/** A plan command after the role was replaced by the bound anchor id. */
export type BoundCommand =
  | { kind: 'focus'; anchor: string; pad: number }
  | { kind: 'label'; anchor: string; text: string }
  | { kind: 'action'; anchor: string; action: ActionType; direction: ActionDirection };


const ACTION_SHORT_LABELS: Record<ActionType, string> = {
  press: '누르기',
  grasp: '잡기',
  move: '이동',
  rotate: '회전',
  open: '열기',
  close: '닫기',
  insert: '넣기',
  remove: '빼기',
  attach: '붙이기',
  fold: '접기',
  place: '올려놓기',
  fit: '맞추기',
  screw: '조이기',
  hold: '누르고 있기',
  flip: '뒤집기',
  pull: '당기기',
  push: '밀기',
  // The graph core's circuit actions (2026-10-05). align is 정렬, not fit's 맞추기, so one label per action.
  align: '정렬',
  connect: '연결하기',
  disconnect: '분리하기',
  bend: '구부리기',
};

/** Actions whose direction is a side of the displayed image (the contract's MOVE_DIRECTIONS). */
const SIDE_DIRECTION_ACTIONS: ReadonlySet<ActionType> = new Set<ActionType>(['move', 'pull', 'push']);
const SIDE_PREFIX: Partial<Record<ActionDirection, string>> = {
  up: '위로',
  down: '아래로',
  left: '왼쪽으로',
  right: '오른쪽으로',
};

function isValidActionType(val: unknown): val is ActionType {
  return typeof val === 'string' && Object.hasOwn(ACTION_SHORT_LABELS, val);
}

/** The contract's direction rule (`ActionAnchorCommand.direction_matches_action`). */
export function isValidActionDirection(action: ActionType, direction: unknown): direction is ActionDirection {
  if (SIDE_DIRECTION_ACTIONS.has(action)) {
    return direction === 'up' || direction === 'down' || direction === 'left' || direction === 'right';
  }
  if (action === 'rotate') {
    return direction === 'clockwise' || direction === 'counterclockwise';
  }
  if (action === 'screw') {
    return direction === 'tighten' || direction === 'loosen';
  }
  return direction === 'none';
}

/** The callout's short Korean text for an action (move/pull/push carry their side, rotate and screw their sense). */
export function actionShortLabel(action: ActionType, direction: ActionDirection): string {
  if (SIDE_DIRECTION_ACTIONS.has(action)) {
    const prefix = SIDE_PREFIX[direction];
    return prefix ? `${prefix} ${ACTION_SHORT_LABELS[action]}` : ACTION_SHORT_LABELS[action];
  }
  if (action === 'rotate') {
    switch (direction) {
      case 'clockwise':
        return '시계방향 회전';
      case 'counterclockwise':
        return '반시계방향 회전';
      default:
        return '회전';
    }
  }
  if (action === 'screw') return direction === 'loosen' ? '풀기' : '조이기';
  return ACTION_SHORT_LABELS[action] ?? '';
}

/** Bind the role-anchored commands of one step to `anchorId`; commands naming anything else are dropped. */
export function bindStepCommands(step: Pick<GuideStep, 'commands'> | null | undefined, anchorId: string): BoundCommand[] {
  if (!step) return [];
  const out: BoundCommand[] = [];
  for (const command of step.commands) {
    if (command.anchor !== TARGET_ROLE) continue;
    if (command.kind === 'focus') {
      const raw = typeof command.pad === 'number' && Number.isFinite(command.pad) ? command.pad : DEFAULT_FOCUS_PAD;
      out.push({ kind: 'focus', anchor: anchorId, pad: Math.min(MAX_FOCUS_PAD, Math.max(0, raw)) });
    } else if (command.kind === 'label' && typeof command.text === 'string' && command.text.trim()) {
      out.push({ kind: 'label', anchor: anchorId, text: command.text });
    } else if (command.kind === 'action') {
      const action = command.action;
      if (isValidActionType(action) && isValidActionDirection(action, command.direction)) {
        out.push({ kind: 'action', anchor: anchorId, action, direction: command.direction });
      }
    }
  }
  return out;
}

export interface LeaderLinePoints {
  from: { x: number; y: number };
  to: { x: number; y: number };
}

/**
 * Computes leader line start (edge of callout) and end (anchor on box perimeter)
 * in overlay CSS px coordinates.
 */
export function computeLeaderLine(
  avoid: Box,
  overlay: { width: number; height: number },
  calloutRect: PxRect,
  placement: PlacedLabel['placement']
): LeaderLinePoints {
  const W = Math.max(0, overlay.width);
  const H = Math.max(0, overlay.height);
  const bx = avoid.x * W;
  const by = avoid.y * H;
  const bw = avoid.width * W;
  const bh = avoid.height * H;

  let fromX = calloutRect.left + calloutRect.width / 2;
  let fromY = calloutRect.top + calloutRect.height / 2;
  let toX = bx + bw / 2;
  let toY = by + bh / 2;

  switch (placement) {
    case 'above':
      fromX = Math.max(calloutRect.left + 8, Math.min(calloutRect.left + calloutRect.width - 8, bx + bw / 2));
      fromY = calloutRect.top + calloutRect.height;
      toX = Math.max(bx, Math.min(bx + bw, fromX));
      toY = by;
      break;
    case 'below':
      fromX = Math.max(calloutRect.left + 8, Math.min(calloutRect.left + calloutRect.width - 8, bx + bw / 2));
      fromY = calloutRect.top;
      toX = Math.max(bx, Math.min(bx + bw, fromX));
      toY = by + bh;
      break;
    case 'right':
      fromX = calloutRect.left;
      fromY = Math.max(calloutRect.top + 6, Math.min(calloutRect.top + calloutRect.height - 6, by + bh / 2));
      toX = bx + bw;
      toY = Math.max(by, Math.min(by + bh, fromY));
      break;
    case 'left':
      fromX = calloutRect.left + calloutRect.width;
      fromY = Math.max(calloutRect.top + 6, Math.min(calloutRect.top + calloutRect.height - 6, by + bh / 2));
      toX = bx;
      toY = Math.max(by, Math.min(by + bh, fromY));
      break;
  }

  return { from: { x: fromX, y: fromY }, to: { x: toX, y: toY } };
}

/** The binding a `tracking` snapshot establishes; null for any other state or a missing snapshot. */
export function bindingFromSnapshot(
  snapshot: LiveTrackSnapshot | null,
  anchorId: string = PRIMARY_ANCHOR_ID
): AnchorBinding | null {
  if (!snapshot || snapshot.state !== 'tracking' || !snapshot.box) return null;
  return { anchorId, runId: snapshot.runId, trackId: snapshot.trackId, generation: snapshot.generation };
}

/** Whether `snapshot` still describes the bound anchor (same run, track and generation). */
export function snapshotMatchesBinding(snapshot: LiveTrackSnapshot | null, binding: AnchorBinding | null): boolean {
  return Boolean(
    snapshot &&
      binding &&
      snapshot.runId === binding.runId &&
      snapshot.trackId === binding.trackId &&
      snapshot.generation === binding.generation
  );
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

/**
 * The focus ring: the box grown by `pad` × its own width/height on every side, clamped to the frame.
 * Same normalized space as the box, so it is drawn in the overlay's `viewBox="0 0 1 1"` svg.
 */
export function focusRing(box: Box, pad: number = DEFAULT_FOCUS_PAD): Box {
  const p = Number.isFinite(pad) ? Math.min(MAX_FOCUS_PAD, Math.max(0, pad)) : DEFAULT_FOCUS_PAD;
  const left = clamp01(box.x - box.width * p);
  const top = clamp01(box.y - box.height * p);
  const right = clamp01(box.x + box.width * (1 + p));
  const bottom = clamp01(box.y + box.height * (1 + p));
  return { x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

/** Where the focus ring comes from: the step's own `focus` command, or implied by its `action`. */
export type FocusSource = 'command' | 'implied';

/**
 * The step's focus ring: its `focus` command's pad; else, when the step has an `action`, `DEFAULT_FOCUS_PAD`
 * (plans mean "focus + action" — an action without a ring looked weak on the real clip); else none (a
 * label-only step draws no ring).
 */
export function resolveFocus(commands: readonly BoundCommand[]): { pad: number; source: FocusSource } | null {
  const focus = commands.find((command): command is Extract<BoundCommand, { kind: 'focus' }> => command.kind === 'focus');
  if (focus) return { pad: focus.pad, source: 'command' };
  if (commands.some((command) => command.kind === 'action')) return { pad: DEFAULT_FOCUS_PAD, source: 'implied' };
  return null;
}

export interface LabelPlacement {
  /** CSS px from the overlay's left edge (the label's left edge). */
  left: number;
  /** CSS px from the overlay's top edge: the label's BOTTOM edge when `above`, its TOP edge when `below`. */
  top: number;
  placement: 'above' | 'below';
}

/**
 * Where the label goes, in CSS px of the overlay (the video's content rect): at the box's left edge,
 * `offsetPx` above its top (`box.y*H - offset`; the element is translated up by its own height). When that
 * would leave the overlay (box touching the top), it goes `offsetPx` below the box instead. `labelWidth`,
 * when known, keeps the label inside the overlay horizontally. Text is positioned, never scaled.
 */
export function labelPlacement(
  box: Box,
  overlay: { width: number; height: number },
  offsetPx: number = LABEL_OFFSET_PX,
  labelSize: { width: number; height: number } | null = null
): LabelPlacement {
  const W = Math.max(0, overlay.width);
  const H = Math.max(0, overlay.height);
  let left = box.x * W;
  if (labelSize && labelSize.width > 0) left = Math.min(left, Math.max(0, W - labelSize.width));
  left = Math.max(0, left);
  const aboveBottom = box.y * H - offsetPx;
  const needed = labelSize?.height ?? 0;
  if (aboveBottom - needed >= 0) return { left, top: aboveBottom, placement: 'above' };
  return { left, top: (box.y + box.height) * H + offsetPx, placement: 'below' };
}

/** An axis-aligned rectangle in CSS px of the overlay (left/top edges). */
export interface PxRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface PlacedLabel {
  /** The label's LEFT edge, CSS px of the overlay. */
  left: number;
  /** The label's TOP edge, CSS px of the overlay (whatever the placement). */
  top: number;
  placement: 'above' | 'below' | 'right' | 'left';
  /** Area (px²) shared with obstacles or the target; 0 when an unobstructed spot exists. */
  overlap: number;
}

function overlapArea(a: PxRect, b: PxRect): number {
  const w = Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left);
  const h = Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top);
  return w > 0 && h > 0 ? w * h : 0;
}

/**
 * Try adjacent positions first, then the content-rect corners for narrow camera surfaces. The result
 * includes target/obstacle overlap; the renderer hides the callout if no unobstructed in-bounds position
 * exists. A leader line keeps a farther callout attached to the same target.
 *
 * `preferred` (the previous frame's placement) adds hysteresis: when an adjacent position with that placement
 * is still unobstructed and in bounds, it is kept even if another side would be tried first, so the callout
 * does not jump between sides from frame to frame. The corner fallbacks never count as the preferred side.
 */
export function placeLabel(
  avoid: Box,
  overlay: { width: number; height: number },
  labelSize: { width: number; height: number },
  obstacles: PxRect[] = [],
  offsetPx: number = LABEL_OFFSET_PX,
  preferred: PlacedLabel['placement'] | null = null
): PlacedLabel {
  const W = Math.max(0, overlay.width);
  const H = Math.max(0, overlay.height);
  const lw = Math.max(0, labelSize.width);
  const lh = Math.max(0, labelSize.height);
  const bx = avoid.x * W;
  const by = avoid.y * H;
  const bw = avoid.width * W;
  const bh = avoid.height * H;
  const clampX = (x: number) => Math.max(0, Math.min(x, W - lw));
  const clampY = (y: number) => Math.max(0, Math.min(y, H - lh));
  const candidates: Array<{ left: number; top: number; placement: PlacedLabel['placement']; corner?: boolean }> = [
    { left: clampX(bx), top: by - offsetPx - lh, placement: 'above' },
    { left: clampX(bx + bw - lw), top: by - offsetPx - lh, placement: 'above' },
    { left: clampX(bx), top: by + bh + offsetPx, placement: 'below' },
    { left: clampX(bx + bw - lw), top: by + bh + offsetPx, placement: 'below' },
  ];
  // Beside the box: level with its top, its middle, then its bottom (a tall box often has a band at the top).
  for (const side of ['right', 'left'] as const) {
    const left = side === 'right' ? bx + bw + offsetPx : bx - offsetPx - lw;
    for (const top of [by, by + (bh - lh) / 2, by + bh - lh]) candidates.push({ left, top: clampY(top), placement: side });
  }
  candidates.push(
    { left: 0, top: 0, placement: 'above', corner: true },
    { left: W - lw, top: 0, placement: 'above', corner: true },
    { left: 0, top: H - lh, placement: 'below', corner: true },
    { left: W - lw, top: H - lh, placement: 'below', corner: true },
  );
  const target = { left: bx, top: by, width: bw, height: bh };
  const eps = 0.5;
  const inside = (c: { left: number; top: number }) =>
    c.top >= -eps && c.top + lh <= H + eps && c.left >= -eps && c.left + lw <= W + eps;
  let best: PlacedLabel | null = null;
  let firstFree: PlacedLabel | null = null;
  for (const c of candidates) {
    if (!inside(c)) continue;
    const rect = { left: c.left, top: c.top, width: lw, height: lh };
    const overlap = obstacles.reduce((sum, o) => sum + overlapArea(rect, o), overlapArea(rect, target));
    const placed = { left: c.left, top: c.top, placement: c.placement, overlap };
    if (overlap === 0) {
      if (!preferred || (c.placement === preferred && !c.corner)) return placed;
      firstFree ??= placed;
      continue;
    }
    if (!best || overlap < best.overlap) best = placed;
  }
  if (firstFree) return firstFree;
  if (best) return best;
  const fallback = { left: clampX(bx), top: clampY(by - offsetPx - lh), placement: 'above' as const };
  const rect = { left: fallback.left, top: fallback.top, width: lw, height: lh };
  return { ...fallback, overlap: obstacles.reduce((sum, o) => sum + overlapArea(rect, o), overlapArea(rect, target)) };
}

/** The anchor's label on the wire: `primary` (the plan's target noun), else `fallback`, trimmed to the limit. */
export function anchorLabel(primary: string | null | undefined, fallback: string | null | undefined): string {
  const raw = (primary ?? '').trim() || (fallback ?? '').trim() || 'object';
  return raw.slice(0, ANCHOR_LABEL_MAX);
}

/**
 * The request's AnchorRef, from the store snapshot read at capture time. `box` only while tracking
 * (contract rule). Null without a snapshot: there is no run to fence the request with.
 */
export function buildAnchorRef(
  snapshot: LiveTrackSnapshot | null,
  label: string,
  anchorId: string = PRIMARY_ANCHOR_ID
): AnchorRef | null {
  if (!snapshot || anchorId === TARGET_ROLE) return null;
  const ref: AnchorRef = {
    anchor_id: anchorId,
    role: 'target',
    label: label.slice(0, ANCHOR_LABEL_MAX) || 'object',
    run_id: snapshot.runId,
    track_id: snapshot.trackId,
    generation: snapshot.generation,
    state: snapshot.state,
  };
  if (snapshot.state === 'tracking' && snapshot.box) ref.box = { ...snapshot.box };
  return ref;
}

/** The echo the server will return for `ref` (what acceptance compares against). */
export function echoOf(ref: AnchorRef | null): AnchorEcho[] {
  return ref ? [{ anchor_id: ref.anchor_id, track_id: ref.track_id, generation: ref.generation }] : [];
}
