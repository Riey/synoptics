/**
 * Action motion layer: frame timing and box-relative geometry (pure; no value imports, so node runs it directly).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  MOTION_CYCLE_MS,
  MOTION_POOL,
  MOTION_STROKE_PAD_PX,
  cachedMotionExtent,
  MOTION_EASE_FRACTION,
  MOTION_FADE_MS,
  MOTION_PLAY_COUNT,
  motionExtent,
  motionFrame,
  motionKey,
  motionShapes,
  provisionalTarget,
  ringPath,
  type MotionFrame,
  type MotionShape,
  type PxBox,
} from './actionMotion.ts';

const BOX: PxBox = { left: 200, top: 150, width: 120, height: 80 };
const CX = BOX.left + BOX.width / 2;
const CY = BOX.top + BOX.height / 2;
const at = (progress: number): MotionFrame => ({ progress, alpha: 1, cycle: 0, settled: false });
const near = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) < eps;

type Bounds = { minX: number; minY: number; maxX: number; maxY: number };

/** Every coordinate pair of a path (the module writes only M/L/Z commands). */
function pathPoints(d: string): Array<[number, number]> {
  assert.match(d, /^[MLZ0-9.\-\s,e]*$/, `path uses only M/L/Z: ${d}`);
  const nums = (d.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
  assert.equal(nums.length % 2, 0, `even number count in ${d}`);
  const out: Array<[number, number]> = [];
  for (let i = 0; i < nums.length; i += 2) out.push([nums[i], nums[i + 1]]);
  return out;
}

function boundsOf(shape: MotionShape): Bounds {
  if (shape.kind === 'rect') {
    return { minX: shape.x, minY: shape.y, maxX: shape.x + shape.width, maxY: shape.y + shape.height };
  }
  if (shape.kind === 'circle') {
    return { minX: shape.cx - shape.r, minY: shape.cy - shape.r, maxX: shape.cx + shape.r, maxY: shape.cy + shape.r };
  }
  const pts = pathPoints(shape.d);
  return {
    minX: Math.min(...pts.map((p) => p[0])),
    minY: Math.min(...pts.map((p) => p[1])),
    maxX: Math.max(...pts.map((p) => p[0])),
    maxY: Math.max(...pts.map((p) => p[1])),
  };
}

const rects = (shapes: MotionShape[]) => shapes.filter((s): s is Extract<MotionShape, { kind: 'rect' }> => s.kind === 'rect');
const paths = (shapes: MotionShape[]) => shapes.filter((s): s is Extract<MotionShape, { kind: 'path' }> => s.kind === 'path');
const circles = (shapes: MotionShape[]) =>
  shapes.filter((s): s is Extract<MotionShape, { kind: 'circle' }> => s.kind === 'circle');
const rectCenter = (r: { x: number; y: number; width: number; height: number }) => ({ x: r.x + r.width / 2, y: r.y + r.height / 2 });

const ALL: Array<[Parameters<typeof motionShapes>[0], Parameters<typeof motionShapes>[1]]> = [
  ['grasp', 'none'],
  ['remove', 'none'],
  ['move', 'up'],
  ['move', 'down'],
  ['move', 'left'],
  ['move', 'right'],
  ['rotate', 'clockwise'],
  ['rotate', 'counterclockwise'],
  ['press', 'none'],
  ['open', 'none'],
  ['close', 'none'],
  ['insert', 'none'],
  ['attach', 'none'],
  ['fold', 'none'],
  ['place', 'none'],
  ['fit', 'none'],
  ['screw', 'tighten'],
  ['screw', 'loosen'],
  ['hold', 'none'],
  ['flip', 'none'],
  ['pull', 'up'],
  ['pull', 'down'],
  ['pull', 'left'],
  ['pull', 'right'],
  ['push', 'up'],
  ['push', 'down'],
  ['push', 'left'],
  ['push', 'right'],
  // The graph core's four schematic circuit actions (2026-10-05): no direction, drawn from the box alone.
  ['align', 'none'],
  ['connect', 'none'],
  ['disconnect', 'none'],
  ['bend', 'none'],
];
/** The nine actions added on 2026-10-03 (designer mock), with every direction. */
const NEW = ALL.slice(12, 28);
/** The four schematic actions added on 2026-10-05 (graph core), after the 2026-10-03 batch. */
const SCHEMATIC = ALL.slice(28);

// ---------------------------------------------------------------- timing

test('frame at 0 ms: first cycle, progress 0, faded out', () => {
  const f = motionFrame(0, false);
  assert.equal(f.cycle, 0);
  assert.equal(f.progress, 0);
  assert.equal(f.alpha, 0);
  assert.equal(f.settled, false);
});

test('mid-motion progress is strictly between 0 and 1 and fully opaque', () => {
  const f = motionFrame(MOTION_CYCLE_MS * MOTION_EASE_FRACTION * 0.5, false);
  assert.ok(f.progress > 0 && f.progress < 1, String(f.progress));
  assert.equal(f.alpha, 1);
  assert.equal(f.cycle, 0);
});

test('progress holds at 1 after the eased part of a cycle', () => {
  const f = motionFrame(MOTION_CYCLE_MS * (MOTION_EASE_FRACTION + 0.05), false);
  assert.equal(f.progress, 1);
});

test('easing is monotonic within a cycle', () => {
  let last = -1;
  for (let ms = 0; ms < MOTION_CYCLE_MS; ms += 20) {
    const p = motionFrame(ms, false).progress;
    assert.ok(p >= last, `${ms}: ${p} < ${last}`);
    last = p;
  }
});

test('alpha fades out at the end of a non-final cycle and back in at the next', () => {
  const end = motionFrame(MOTION_CYCLE_MS - MOTION_FADE_MS / 2, false);
  assert.ok(end.alpha > 0 && end.alpha < 1, String(end.alpha));
  assert.equal(end.cycle, 0);
  const next = motionFrame(MOTION_CYCLE_MS + MOTION_FADE_MS / 2, false);
  assert.equal(next.cycle, 1);
  assert.ok(next.alpha > 0 && next.alpha < 1, String(next.alpha));
  assert.ok(next.progress < 0.05, String(next.progress));
});

test('the final cycle does not fade out: it ends on the end pose', () => {
  const lastStart = MOTION_CYCLE_MS * (MOTION_PLAY_COUNT - 1);
  const f = motionFrame(lastStart + MOTION_CYCLE_MS - MOTION_FADE_MS / 2, false);
  assert.equal(f.cycle, MOTION_PLAY_COUNT - 1);
  assert.equal(f.alpha, 1);
  assert.equal(f.progress, 1);
  assert.equal(f.settled, false);
});

test('after the play count the end pose is held, settled', () => {
  for (const ms of [MOTION_CYCLE_MS * MOTION_PLAY_COUNT, MOTION_CYCLE_MS * MOTION_PLAY_COUNT + 1, 1e9]) {
    assert.deepEqual(motionFrame(ms, false), { progress: 1, alpha: 1, cycle: MOTION_PLAY_COUNT, settled: true });
  }
});

test('reduced motion shows the settled end pose from 0 ms', () => {
  assert.deepEqual(motionFrame(0, true), { progress: 1, alpha: 1, cycle: MOTION_PLAY_COUNT, settled: true });
  assert.deepEqual(motionFrame(500, true), { progress: 1, alpha: 1, cycle: MOTION_PLAY_COUNT, settled: true });
});

test('negative or non-finite elapsed is treated as 0', () => {
  assert.equal(motionFrame(-50, false).progress, 0);
  assert.equal(motionFrame(Number.NaN, false).progress, 0);
});

// ---------------------------------------------------------------- geometry

test('alpha multiplies every opacity', () => {
  for (const [action, direction] of ALL) {
    const full = motionShapes(action, direction, BOX, { progress: 1, alpha: 1, cycle: 0, settled: false });
    const half = motionShapes(action, direction, BOX, { progress: 1, alpha: 0.5, cycle: 0, settled: false });
    assert.equal(full.length, half.length);
    full.forEach((s, i) => assert.ok(near(half[i].opacity, s.opacity * 0.5), `${action}:${direction} #${i}`));
  }
});

test('grasp: brackets close from far outside to just outside the box edges', () => {
  const [l0, r0] = paths(motionShapes('grasp', 'none', BOX, at(0)));
  const [l1, r1] = paths(motionShapes('grasp', 'none', BOX, at(1)));
  const L0 = boundsOf(l0), R0 = boundsOf(r0), L1 = boundsOf(l1), R1 = boundsOf(r1);
  // Left bracket ends 6 px left of the box, right bracket starts 6 px right of it.
  assert.ok(near(L1.maxX, BOX.left - 6), String(L1.maxX));
  assert.ok(near(R1.minX, BOX.left + BOX.width + 6), String(R1.minX));
  // At p=0 they are farther out.
  assert.ok(L0.maxX < L1.maxX - 20);
  assert.ok(R0.minX > R1.minX + 20);
  // Never on the box.
  assert.ok(L0.maxX <= BOX.left && R0.minX >= BOX.left + BOX.width);
  // Height 0.6·h centred.
  assert.ok(near(L1.maxY - L1.minY, BOX.height * 0.6));
  assert.ok(near((L1.minY + L1.maxY) / 2, CY));
});

test('remove: no direction — the ghost grows about the box centre, four corner arrowheads point outward', () => {
  const s0 = motionShapes('remove', 'none', BOX, at(0));
  const s1 = motionShapes('remove', 'none', BOX, at(1));
  const [g0] = rects(s0);
  const [g1] = rects(s1);
  assert.deepEqual([g0.x, g0.y, g0.width, g0.height], [BOX.left, BOX.top, BOX.width, BOX.height]);
  assert.equal(g0.dashed, true);
  for (const p of [0, 0.5, 1]) {
    const [g] = rects(motionShapes('remove', 'none', BOX, at(p)));
    assert.ok(near(rectCenter(g).x, CX) && near(rectCenter(g).y, CY), `centre unchanged at p=${p}`);
  }
  assert.ok(g1.width > BOX.width && g1.height > BOX.height, 'end ghost larger than the box');
  assert.ok(near(g1.width, BOX.width + 2 * 24) && near(g1.height, BOX.height + 2 * 24), '1.35× with a 24 px/side floor');
  assert.ok(g1.opacity < g0.opacity);
  const heads = paths(s1).filter((x) => x.filled);
  assert.equal(heads.length, 4);
  for (const head of heads) {
    assert.ok(head.opacity > 0, 'visible in the end pose');
    const [tip, b1, b2] = pathPoints(head.d);
    const base = [(b1[0] + b2[0]) / 2, (b1[1] + b2[1]) / 2];
    assert.ok(Math.hypot(tip[0] - CX, tip[1] - CY) > Math.hypot(base[0] - CX, base[1] - CY), 'tip farther from the centre');
    const [dx, dy] = arrowDir(head);
    assert.ok(Math.abs(Math.abs(dx) - Math.SQRT1_2) < 0.01 && Math.abs(Math.abs(dy) - Math.SQRT1_2) < 0.01, 'diagonal');
  }
  // One arrowhead per corner quadrant; none claims a single direction.
  const quadrants = new Set(heads.map((hd) => arrowDir(hd).map(Math.sign).join(',')));
  assert.equal(quadrants.size, 4);
  // Growth is capped at 80 px per side on a huge box.
  const [big] = rects(motionShapes('remove', 'none', { left: 0, top: 0, width: 900, height: 600 }, at(1)));
  assert.ok(near(big.width, 900 + 160) && near(big.height, 600 + 160), `${big.width}×${big.height}`);
});

test('move:right: ghost ends one box width to the right, with a trail and a leading arrowhead', () => {
  const s0 = motionShapes('move', 'right', BOX, at(0));
  const s1 = motionShapes('move', 'right', BOX, at(1));
  const [g0] = rects(s0);
  const [g1] = rects(s1);
  assert.ok(near(g0.x, BOX.left) && near(g0.y, BOX.top));
  assert.ok(near(g1.x, BOX.left + BOX.width) && near(g1.y, BOX.top));
  const [trail, head] = paths(s1);
  const tb = boundsOf(trail);
  assert.ok(near(tb.minX, BOX.left + BOX.width), 'trail starts at the right edge');
  assert.equal(head.filled, true);
  const [tip] = pathPoints(head.d);
  assert.ok(near(tip[0], g1.x + g1.width), 'arrowhead tip at the ghost leading edge');
  assert.ok(tb.maxX <= tip[0], 'trail ends behind the tip');
  // Trail grows with progress.
  const mid = boundsOf(paths(motionShapes('move', 'right', BOX, at(0.5)))[0]);
  assert.ok(mid.maxX - mid.minX < tb.maxX - tb.minX);
});

test('move:up / down / left put the ghost on that side', () => {
  const up = rects(motionShapes('move', 'up', BOX, at(1)))[0];
  assert.ok(near(up.y, BOX.top - BOX.height) && near(up.x, BOX.left));
  const down = rects(motionShapes('move', 'down', BOX, at(1)))[0];
  assert.ok(near(down.y, BOX.top + BOX.height));
  const left = rects(motionShapes('move', 'left', BOX, at(1)))[0];
  assert.ok(near(left.x, BOX.left - BOX.width));
});

test('rotate: clockwise and counterclockwise are mirror images about the box centre', () => {
  for (const p of [0.3, 1]) {
    const cw = paths(motionShapes('rotate', 'clockwise', BOX, at(p)));
    const ccw = paths(motionShapes('rotate', 'counterclockwise', BOX, at(p)));
    assert.equal(cw.length, ccw.length);
    cw.forEach((shape, i) => {
      // Compare as point sets: a mirrored triangle lists its base corners in the other order.
      const byYX = (u: [number, number], v: [number, number]) => u[1] - v[1] || u[0] - v[0];
      const a = pathPoints(shape.d)
        .map(([x, y]): [number, number] => [2 * CX - x, y])
        .sort(byYX);
      const b = pathPoints(ccw[i].d).sort(byYX);
      assert.equal(a.length, b.length);
      a.forEach(([x, y], j) => {
        assert.ok(near(x, b[j][0], 0.02), `x mirror p=${p} #${i}.${j}`);
        assert.ok(near(y, b[j][1], 0.02), `y equal p=${p} #${i}.${j}`);
      });
    });
  }
});

test('rotate: arc on a circle enclosing the box corners, starting at the top, 270° at p=1', () => {
  const R = 0.5 * Math.hypot(BOX.width, BOX.height) + 10;
  const [arc, head] = paths(motionShapes('rotate', 'clockwise', BOX, at(1)));
  const pts = pathPoints(arc.d);
  for (const [x, y] of pts) assert.ok(near(Math.hypot(x - CX, y - CY), R, 0.02));
  assert.ok(near(pts[0][0], CX, 0.02) && near(pts[0][1], CY - R, 0.02), 'starts at top');
  const end = pts[pts.length - 1];
  // Clockwise on screen (y down) from the top through right and bottom ends at the left.
  assert.ok(near(end[0], CX - R, 0.02) && near(end[1], CY, 0.02), `ends at left: ${end}`);
  assert.equal(head.filled, true);
  // Passes through the right side.
  assert.ok(pts.some(([x]) => near(x, CX + R, 0.5)));
});

test('press: the ring contracts and the centre dot appears late', () => {
  const c0 = circles(motionShapes('press', 'none', BOX, at(0)));
  const c1 = circles(motionShapes('press', 'none', BOX, at(1)));
  const ring0 = c0.find((c) => !c.filled)!;
  const ring1 = c1.find((c) => !c.filled)!;
  assert.ok(near(ring0.r, 0.55 * Math.max(BOX.width, BOX.height)));
  assert.ok(ring1.r < ring0.r);
  assert.ok(near(ring1.cx, CX) && near(ring1.cy, CY));
  assert.ok(ring1.opacity > ring0.opacity);
  assert.equal(c0.filter((c) => c.filled && c.opacity > 0).length, 0);
  const dot = c1.find((c) => c.filled)!;
  assert.ok(dot.opacity > 0 && near(dot.cx, CX) && near(dot.cy, CY));
  let last = Infinity;
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    const r = circles(motionShapes('press', 'none', BOX, at(p))).find((c) => !c.filled)!.r;
    assert.ok(r < last);
    last = r;
  }
});

/** The lid (top half, no bottom edge) and body (bottom half, no top edge) of open/close, as bounds. */
function halves(shapes: MotionShape[]) {
  const [lid, body, seam] = paths(shapes);
  return { lid: boundsOf(lid), body: boundsOf(body), seam };
}

test('open: halves start joined and end separated; both are open U paths', () => {
  const h0 = halves(motionShapes('open', 'none', BOX, at(0)));
  const h1 = halves(motionShapes('open', 'none', BOX, at(1)));
  assert.equal(rects(motionShapes('open', 'none', BOX, at(1))).length, 0, 'no closed rects over the seam');
  assert.ok(near(h0.lid.maxY, h0.body.minY), 'joined at p=0');
  assert.ok(near(h0.lid.minY, BOX.top) && near(h0.body.maxY, BOX.top + BOX.height));
  assert.ok(near(h1.body.minY - h1.lid.maxY, BOX.height * 0.35), 'gap of 0.35·h at p=1');
  assert.ok(near(h1.lid.maxX - h1.lid.minX, BOX.width * 0.9));
  assert.ok(near((h1.lid.minX + h1.lid.maxX) / 2, CX));
  assert.ok(near(h1.body.minY, h0.body.minY), 'body stays');
  assert.equal(h1.seam.dashed, true);
  // The lid's polyline ends are at its bottom (open side); the body's at its top.
  const lidPts = pathPoints(paths(motionShapes('open', 'none', BOX, at(1)))[0].d);
  assert.ok(near(lidPts[0][1], h1.lid.maxY) && near(lidPts[lidPts.length - 1][1], h1.lid.maxY));
  const bodyPts = pathPoints(paths(motionShapes('open', 'none', BOX, at(1)))[1].d);
  assert.ok(near(bodyPts[0][1], h1.body.minY) && near(bodyPts[bodyPts.length - 1][1], h1.body.minY));
});

test('close: halves start separated and end joined, with downward arrowheads outside the seam ends', () => {
  const h0 = halves(motionShapes('close', 'none', BOX, at(0)));
  const s1 = motionShapes('close', 'none', BOX, at(1));
  const h1 = halves(s1);
  assert.ok(near(h0.body.minY - h0.lid.maxY, BOX.height * 0.35));
  assert.ok(near(h1.lid.maxY, h1.body.minY), 'joined at p=1');
  assert.ok(near(h1.lid.maxX - h1.lid.minX, BOX.width));
  assert.equal(h1.seam.dashed, true);
  const heads = paths(s1).filter((p) => p.filled && p.opacity > 0);
  assert.equal(heads.length, 2);
  const seamY = h1.body.minY;
  const [hl, hr] = heads.map(boundsOf).sort((a, b) => a.minX - b.minX);
  assert.ok(hl.maxX <= BOX.left - 6 && hr.minX >= BOX.left + BOX.width + 6, 'outside the seam ends');
  assert.ok(near(hl.maxY, seamY) && near(hr.maxY, seamY), 'tips on the seam line');
});

test('insert: ghost starts above at 0.8 and ends small inside the box', () => {
  const [g0] = rects(motionShapes('insert', 'none', BOX, at(0)));
  const s1 = motionShapes('insert', 'none', BOX, at(1));
  const [g1] = rects(s1);
  assert.ok(near(rectCenter(g0).y, BOX.top - BOX.height * 0.6));
  assert.ok(near(g0.width, BOX.width * 0.8));
  assert.ok(g0.y + g0.height <= BOX.top + 1e-6, 'starts above the box');
  assert.ok(near(rectCenter(g1).x, CX) && near(rectCenter(g1).y, CY));
  assert.ok(near(g1.width, BOX.width * 0.45));
  assert.ok(g1.x >= BOX.left && g1.y >= BOX.top && g1.x + g1.width <= BOX.left + BOX.width && g1.y + g1.height <= BOX.top + BOX.height);
  const head = paths(s1).find((p) => p.filled)!;
  const hb = boundsOf(head);
  assert.ok(hb.maxY > BOX.top, 'arrowhead reaches into the box');
});

test('pool budget: no action uses more nodes than the overlay pool holds', () => {
  for (const [action, direction] of ALL) {
    for (const p of [0, 0.5, 1]) {
      const s = motionShapes(action, direction, BOX, at(p));
      assert.ok(
        rects(s).length <= MOTION_POOL.rect && paths(s).length <= MOTION_POOL.path && circles(s).length <= MOTION_POOL.circle,
        `${action}:${direction}`
      );
    }
  }
});

test('shape counts do not change with progress (stable pool assignment)', () => {
  for (const [action, direction] of ALL) {
    const kinds = (p: number) => motionShapes(action, direction, BOX, at(p)).map((s) => s.kind).join(',');
    assert.equal(kinds(0), kinds(1), `${action}:${direction}`);
    assert.equal(kinds(0.4), kinds(1), `${action}:${direction}`);
  }
});

test('every number is finite for tiny and huge boxes', () => {
  const boxes: PxBox[] = [
    { left: 10, top: 10, width: 4, height: 4 },
    { left: 0, top: 0, width: 900, height: 600 },
  ];
  for (const box of boxes) {
    for (const [action, direction] of ALL) {
      for (const p of [0, 0.25, 0.5, 0.75, 1]) {
        for (const s of motionShapes(action, direction, box, at(p))) {
          assert.ok(Number.isFinite(s.opacity) && s.opacity >= 0 && s.opacity <= 1);
          if (s.kind === 'rect') {
            for (const v of [s.x, s.y, s.width, s.height]) assert.ok(Number.isFinite(v));
            assert.ok(s.width >= 0 && s.height >= 0);
          } else if (s.kind === 'circle') {
            for (const v of [s.cx, s.cy, s.r]) assert.ok(Number.isFinite(v));
            assert.ok(s.r >= 0);
          } else {
            assert.doesNotMatch(s.d, /NaN|Infinity/);
            for (const [x, y] of pathPoints(s.d)) assert.ok(Number.isFinite(x) && Number.isFinite(y));
          }
        }
      }
      const e = motionExtent(action, direction, box);
      for (const v of [e.left, e.top, e.width, e.height]) assert.ok(Number.isFinite(v));
    }
  }
});

/** Dense containment, independent of the implementation's sample points: 101 progress values, stroke half width. */
function assertExtentContains(extentOf: typeof motionExtent, box: PxBox, tag: string) {
  const halfStroke = 1.5;
  for (const [action, direction] of ALL) {
    const e = extentOf(action, direction, box);
    for (let i = 0; i <= 100; i += 1) {
      const p = i / 100;
      for (const s of motionShapes(action, direction, box, at(p))) {
        const b = boundsOf(s);
        const where = `${tag} ${action}:${direction} p=${p} ${s.kind}`;
        assert.ok(b.minX - halfStroke >= e.left - 1e-6 && b.minY - halfStroke >= e.top - 1e-6, where);
        assert.ok(
          b.maxX + halfStroke <= e.left + e.width + 1e-6 && b.maxY + halfStroke <= e.top + e.height + 1e-6,
          where
        );
      }
    }
    assert.ok(e.left <= box.left - MOTION_STROKE_PAD_PX + 1e-6 && e.top <= box.top - MOTION_STROKE_PAD_PX + 1e-6, `${tag} pad`);
  }
}

const EXTENT_BOXES: PxBox[] = [
  BOX,
  { left: 10, top: 10, width: 4, height: 4 },
  { left: 33.3, top: 71.7, width: 57.4, height: 140.6 },
  { left: 5, top: 5, width: 900, height: 600 },
];

test('extent contains every shape over 101 progress values, stroke included', () => {
  for (const box of EXTENT_BOXES) assertExtentContains(motionExtent, box, 'exact');
});

test('cached extent (rounded size, translated) also contains every shape', () => {
  for (const box of EXTENT_BOXES) assertExtentContains(cachedMotionExtent, box, 'cached');
  // Same rounded size at another position: translated, not recomputed.
  const a = cachedMotionExtent('move', 'right', { left: 0, top: 0, width: 100, height: 50 });
  const b = cachedMotionExtent('move', 'right', { left: 40, top: 30, width: 100, height: 50 });
  assert.ok(near(b.left - a.left, 40) && near(b.top - a.top, 30) && near(a.width, b.width) && near(a.height, b.height));
});

// ---------------------------------------------------------------- arrowhead directions

/** Unit vector from an arrowhead's base midpoint to its tip (the module writes tip, base, base). */
function arrowDir(shape: MotionShape): [number, number] {
  assert.equal(shape.kind, 'path');
  const pts = pathPoints((shape as Extract<MotionShape, { kind: 'path' }>).d);
  assert.equal(pts.length, 3);
  const [tip, b1, b2] = pts;
  const dx = tip[0] - (b1[0] + b2[0]) / 2;
  const dy = tip[1] - (b1[1] + b2[1]) / 2;
  const len = Math.hypot(dx, dy);
  return [dx / len, dy / len];
}
const heads = (shapes: MotionShape[]) => paths(shapes).filter((p) => p.filled);
function assertPoints(shape: MotionShape, expected: [number, number], tag: string) {
  const [x, y] = arrowDir(shape);
  assert.ok(x * expected[0] + y * expected[1] > 0.99, `${tag}: got ${x.toFixed(3)},${y.toFixed(3)}`);
}

test('arrowheads point the way the action goes', () => {
  const vec = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] } as const;
  for (const dir of ['up', 'down', 'left', 'right'] as const) {
    for (const p of [0.5, 1]) assertPoints(heads(motionShapes('move', dir, BOX, at(p)))[0], [...vec[dir]], `move:${dir} p=${p}`);
  }
  // Rotate: tangent at the tip — at the start (top) cw goes right, ccw left; at 270° both go up.
  assertPoints(heads(motionShapes('rotate', 'clockwise', BOX, at(0)))[0], [1, 0], 'rotate cw p=0');
  assertPoints(heads(motionShapes('rotate', 'counterclockwise', BOX, at(0)))[0], [-1, 0], 'rotate ccw p=0');
  assertPoints(heads(motionShapes('rotate', 'clockwise', BOX, at(1)))[0], [0, -1], 'rotate cw p=1');
  assertPoints(heads(motionShapes('rotate', 'counterclockwise', BOX, at(1)))[0], [0, -1], 'rotate ccw p=1');
  for (const p of [0.5, 1]) {
    assertPoints(heads(motionShapes('insert', 'none', BOX, at(p)))[0], [0, 1], `insert p=${p}`);
    const closeHeads = heads(motionShapes('close', 'none', BOX, at(p)));
    assert.equal(closeHeads.length, 2);
    for (const h of closeHeads) assertPoints(h, [0, 1], `close p=${p}`);
  }
});

// ---------------------------------------------------------------- frame edges and minimum sizes

const STAGE = { width: 640, height: 400 };
const insideFrame = (b: Bounds, f = STAGE) => b.minX >= -1e-6 && b.minY >= -1e-6 && b.maxX <= f.width + 1e-6 && b.maxY <= f.height + 1e-6;

test('move:down on a box touching the bottom edge stays inside the frame and still points down', () => {
  const box: PxBox = { left: 280, top: 340, width: 80, height: 60 };
  for (let i = 0; i <= 20; i += 1) {
    const shapes = motionShapes('move', 'down', box, at(i / 20), STAGE);
    assert.equal(rects(shapes).length, 0, 'no ghost without room');
    for (const sh of shapes) assert.ok(insideFrame(boundsOf(sh)), `p=${i / 20} ${sh.kind}`);
    const head = heads(shapes)[0];
    const [tip, b1, b2] = pathPoints(head.d);
    assert.ok(tip[1] > (b1[1] + b2[1]) / 2, 'tip below base');
  }
  const head = heads(motionShapes('move', 'down', box, at(1), STAGE))[0];
  assert.ok(near(pathPoints(head.d)[0][1], STAGE.height - 3), 'end tip just inside the bottom edge');
  const e = motionExtent('move', 'down', box, STAGE);
  assert.ok(e.top + e.height <= STAGE.height + 1e-6, 'extent clamped to the frame');
});

test('move:right near the right edge shortens the ghost travel to the room left', () => {
  const box: PxBox = { left: 520, top: 150, width: 80, height: 60 }; // 34 px to the edge → 28 px of room
  const s1 = motionShapes('move', 'right', box, at(1), STAGE);
  const [g] = rects(s1);
  assert.ok(near(g.x + g.width, STAGE.width - 6), 'ghost stops 6 px inside the edge');
  for (const sh of s1) assert.ok(insideFrame(boundsOf(sh)), sh.kind);
  // Touching the edge: edge mode, everything inside, tip right of base.
  const flush = motionShapes('move', 'right', { left: 560, top: 150, width: 80, height: 60 }, at(1), STAGE);
  assert.equal(rects(flush).length, 0);
  for (const sh of flush) assert.ok(insideFrame(boundsOf(sh)), sh.kind);
  assertPoints(heads(flush)[0], [1, 0], 'edge move:right');
});

test('move:up / left at the top-left corner also stay inside', () => {
  const box: PxBox = { left: 0, top: 0, width: 50, height: 30 };
  for (const dir of ['up', 'left'] as const) {
    for (const sh of motionShapes('move', dir, box, at(1), STAGE)) assert.ok(insideFrame(boundsOf(sh)), `${dir} ${sh.kind}`);
  }
});

test('insert near the top edge keeps the start ghost and trail inside the frame', () => {
  const box: PxBox = { left: 300, top: 20, width: 60, height: 40 };
  for (const p of [0, 0.5, 1]) {
    for (const sh of motionShapes('insert', 'none', box, at(p), STAGE)) assert.ok(insideFrame(boundsOf(sh)), `p=${p} ${sh.kind}`);
  }
});

test('minimum sizes: on a 60×20 box every directional motion reaches ≥24 px past the box on its side', () => {
  const box: PxBox = { left: 290, top: 190, width: 60, height: 20 };
  const right = box.left + box.width;
  const bottom = box.top + box.height;
  const reach = (action: Parameters<typeof motionExtent>[0], direction: Parameters<typeof motionExtent>[1]) => {
    const e = motionExtent(action, direction, box, STAGE);
    return { left: box.left - e.left, top: box.top - e.top, right: e.left + e.width - right, bottom: e.top + e.height - bottom };
  };
  const at24 = (v: number, tag: string) => assert.ok(v >= 24 - 1e-6, `${tag}: ${v.toFixed(1)}`);
  at24(reach('move', 'up').top, 'move:up');
  at24(reach('move', 'down').bottom, 'move:down');
  at24(reach('move', 'left').left, 'move:left');
  at24(reach('move', 'right').right, 'move:right');
  at24(reach('insert', 'none').top, 'insert');
  for (const side of ['left', 'top', 'right', 'bottom'] as const) at24(reach('remove', 'none')[side], `remove ${side}`);
  at24(reach('grasp', 'none').left, 'grasp left');
  at24(reach('grasp', 'none').right, 'grasp right');
  for (const side of ['left', 'top', 'right', 'bottom'] as const) {
    at24(reach('rotate', 'clockwise')[side], `rotate cw ${side}`);
    at24(reach('rotate', 'counterclockwise')[side], `rotate ccw ${side}`);
  }
  // Arrowheads are at least 10 px long.
  for (const [action, direction] of ALL) {
    for (const head of heads(motionShapes(action, direction, box, at(1), STAGE))) {
      const [tip, b1, b2] = pathPoints(head.d);
      assert.ok(Math.hypot(tip[0] - (b1[0] + b2[0]) / 2, tip[1] - (b1[1] + b2[1]) / 2) >= 10 - 0.02, `${action}:${direction}`);
    }
  }
  // The press ring starts at ≥28 px.
  const ring = circles(motionShapes('press', 'none', box, at(0), STAGE)).find((c) => !c.filled)!;
  assert.ok(ring.r >= 28);
});

test('cached extent equals a direct, frame-clamped extent in containment terms for every action', () => {
  const boxes: PxBox[] = [
    { left: 280, top: 340, width: 80, height: 60 },
    { left: 560, top: 150, width: 80, height: 60 },
    { left: 290.4, top: 190.6, width: 60.3, height: 20.2 },
  ];
  for (const box of boxes) {
    for (const [action, direction] of ALL) {
      const e = cachedMotionExtent(action, direction, box, STAGE);
      assert.ok(e.left >= -1e-6 && e.top >= -1e-6 && e.left + e.width <= STAGE.width + 1e-6 && e.top + e.height <= STAGE.height + 1e-6);
      for (let i = 0; i <= 100; i += 1) {
        for (const sh of motionShapes(action, direction, box, at(i / 100), STAGE)) {
          const b = boundsOf(sh);
          // Inside the extent wherever the shape is inside the frame.
          const cl = { minX: Math.max(b.minX, 0), minY: Math.max(b.minY, 0), maxX: Math.min(b.maxX, STAGE.width), maxY: Math.min(b.maxY, STAGE.height) };
          if (cl.minX > cl.maxX || cl.minY > cl.maxY) continue;
          const tag = `${action}:${direction} p=${i / 100}`;
          assert.ok(cl.minX >= e.left - 1e-6 && cl.minY >= e.top - 1e-6, tag);
          assert.ok(cl.maxX <= e.left + e.width + 1e-6 && cl.maxY <= e.top + e.height + 1e-6, tag);
        }
      }
    }
  }
});

// ---------------------------------------------------------------- focus ring brackets

/** The ring's subpaths as point lists (M, L, Q control + end, L), checking the M/L/Q-only grammar. */
function ringCorners(d: string): Array<Array<[number, number]>> {
  assert.match(d, /^[MLQ0-9.\-\s]*$/, `ring uses only M/L/Q: ${d}`);
  return d
    .split('M')
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      assert.match(part, /^[0-9.\-]+ [0-9.\-]+ L [0-9.\-]+ [0-9.\-]+ Q [0-9.\-]+ [0-9.\-]+ [0-9.\-]+ [0-9.\-]+ L [0-9.\-]+ [0-9.\-]+$/, part);
      const nums = (part.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);
      const pts: Array<[number, number]> = [];
      for (let i = 0; i < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
      return pts;
    });
}

test('ringPath: four rounded corner brackets on the box edges, length clamp(0.22·min side, 10, 36)', () => {
  const cases: Array<[PxBox, number]> = [
    [{ left: 100, top: 50, width: 200, height: 150 }, 33],
    [{ left: 0, top: 0, width: 900, height: 600 }, 36],
    [{ left: 10.25, top: 20.5, width: 30, height: 20 }, 10],
  ];
  for (const [box, len] of cases) {
    const corners = ringCorners(ringPath(box));
    assert.equal(corners.length, 4);
    const x0 = box.left, y0 = box.top, x1 = box.left + box.width, y1 = box.top + box.height;
    const rc = Math.min(10, len / 2);
    const expectCorners: Array<[number, number, number, number]> = [[x0, y0, 1, 1], [x1, y0, -1, 1], [x1, y1, -1, -1], [x0, y1, 1, -1]];
    corners.forEach((pts, i) => {
      const [ax, ay, sx, sy] = expectCorners[i];
      const tag = `${box.width}×${box.height} corner ${i}`;
      // start on the vertical edge `len` from the corner, elbow control at the corner, end on the horizontal edge.
      const want: Array<[number, number]> = [[ax, ay + sy * len], [ax, ay + sy * rc], [ax, ay], [ax + sx * rc, ay], [ax + sx * len, ay]];
      pts.forEach(([x, y], j) => assert.ok(near(x, want[j][0], 0.006) && near(y, want[j][1], 0.006), `${tag} pt ${j}: ${x},${y}`));
      for (const [x, y] of pts) assert.ok(near(x, x0, 0.006) || near(x, x1, 0.006) || near(y, y0, 0.006) || near(y, y1, 0.006), tag);
    });
  }
});

test('ringPath: finite for degenerate input', () => {
  for (const box of [{ left: 0, top: 0, width: 0, height: 0 }, { left: Number.NaN, top: 5, width: Number.POSITIVE_INFINITY, height: -3 }]) {
    const d = ringPath(box);
    assert.doesNotMatch(d, /NaN|Infinity/);
    assert.equal(ringCorners(d).length, 4);
  }
});

// ---------------------------------------------------------------- the nine actions added 2026-10-03

test('new actions: pool budget and kinds per action (fixed order, all within rect 1 / path 5 / circle 2)', () => {
  const expected: Record<string, string> = {
    attach: 'path,rect,path',
    fold: 'path,path,path,path',
    place: 'rect,path,path',
    fit: 'rect,path,path,path',
    screw: 'circle,path,path,path',
    hold: 'circle,circle,path',
    flip: 'path,rect,path,path',
    pull: 'rect,path,path,path',
    push: 'rect,path,path,path',
  };
  assert.equal(NEW.length, 16);
  for (const [action, direction] of NEW) {
    for (const box of [BOX, { left: 10, top: 10, width: 4, height: 4 }, { left: 0, top: 0, width: 900, height: 600 }]) {
      for (const view of [undefined, STAGE]) {
        for (let i = 0; i <= 20; i += 1) {
          const kinds = motionShapes(action, direction, box, at(i / 20), view).map((sh) => sh.kind).join(',');
          assert.equal(kinds, expected[action], `${action}:${direction} p=${i / 20}`);
        }
      }
    }
  }
});

test('attach: the band sweeps from 12 px left of the box to 12 px right of it, arrowhead leading right', () => {
  const s0 = motionShapes('attach', 'none', BOX, at(0));
  const s1 = motionShapes('attach', 'none', BOX, at(1));
  const [b0] = rects(s0);
  const [b1] = rects(s1);
  assert.equal(b0.width, 0);
  assert.ok(near(b1.x, BOX.left - 12) && near(b1.width, BOX.width + 24));
  assert.ok(near(rectCenter(b1).y, CY) && near(b1.height, 0.14 * BOX.height));
  assert.equal(b1.filled, true);
  const [line] = paths(s1);
  assert.equal(line.dashed, true);
  for (const p of [0, 0.5, 1]) assertPoints(heads(motionShapes('attach', 'none', BOX, at(p)))[0], [1, 0], `attach p=${p}`);
  const [tip] = pathPoints(heads(s1)[0].d);
  assert.ok(near(tip[0], BOX.left + BOX.width + 12 + 12), 'tip one arrow length past the band end');
  // A thin box still gets an 8 px band.
  assert.ok(near(rects(motionShapes('attach', 'none', { left: 0, top: 0, width: 60, height: 20 }, at(1)))[0].height, 8));
});

test('fold: the flap stands on the top edge at p=0 and lies folded over the box at p=1; the arc runs beside it', () => {
  const flap = (p: number) => boundsOf(paths(motionShapes('fold', 'none', BOX, at(p)))[1]);
  const half = BOX.height / 2;
  assert.ok(near(flap(0).minY, BOX.top - half) && near(flap(0).maxY, BOX.top), 'upright above the hinge');
  assert.ok(near(flap(1).maxY, BOX.top + half) && near(flap(1).minY, BOX.top), 'folded down over the box');
  assert.ok(flap(0.5).maxY - flap(0.5).minY < 1e-6 + 0.01, 'edge-on mid-fold');
  const [seam, , arc] = paths(motionShapes('fold', 'none', BOX, at(1)));
  assert.equal(seam.dashed, true);
  assert.ok(near(boundsOf(seam).minY, BOX.top), 'the hinge is the top edge');
  const ab = boundsOf(arc);
  assert.ok(ab.minX >= BOX.left + BOX.width, 'arc right of the box');
  // The arrowhead turns from right (top of the arc) through down to left (bottom).
  assertPoints(heads(motionShapes('fold', 'none', BOX, at(0)))[0], [1, 0], 'fold p=0');
  assertPoints(heads(motionShapes('fold', 'none', BOX, at(0.5)))[0], [0, 1], 'fold p=0.5');
  assertPoints(heads(motionShapes('fold', 'none', BOX, at(1)))[0], [-1, 0], 'fold p=1');
  // A small box gets a 12 px flap.
  const small = { left: 100, top: 100, width: 60, height: 20 };
  assert.ok(near(boundsOf(paths(motionShapes('fold', 'none', small, at(0)))[1]).minY, 100 - 12));
});

test('fold / flip: on a box at the right edge the arc goes to the left side and stays in the frame', () => {
  const box: PxBox = { left: 560, top: 150, width: 80, height: 60 };
  for (const action of ['fold', 'flip'] as const) {
    for (let i = 0; i <= 20; i += 1) {
      const shapes = motionShapes(action, 'none', box, at(i / 20), STAGE);
      const arc = paths(shapes)[action === 'fold' ? 2 : 1];
      assert.ok(boundsOf(arc).maxX <= box.left + 1e-6, `${action} arc on the left p=${i / 20}`);
      // Everything but the dashed seam (it overhangs the box by 6 px on both sides, like open/close).
      for (const sh of shapes.slice(1)) assert.ok(insideFrame(boundsOf(sh)), `${action} p=${i / 20} ${sh.kind}`);
    }
    // Mirrored side: the head starts pointing left.
    assertPoints(heads(motionShapes(action, 'none', box, at(0), STAGE))[0], [-1, 0], `${action} left p=0`);
  }
});

test('attach / fold / flip at each frame edge stay inside (dashed seams excepted); fold hangs its flap below at the top', () => {
  const edgeBoxes: PxBox[] = [
    { left: 280, top: 0, width: 80, height: 60 },
    { left: 280, top: 340, width: 80, height: 60 },
    { left: 0, top: 150, width: 80, height: 60 },
    { left: 560, top: 150, width: 80, height: 60 },
  ];
  for (const box of edgeBoxes) {
    for (const action of ['attach', 'fold', 'flip'] as const) {
      for (let i = 0; i <= 20; i += 1) {
        const shapes = motionShapes(action, 'none', box, at(i / 20), STAGE);
        for (const sh of shapes) {
          if (sh.kind === 'path' && sh.dashed && action !== 'attach' && sh === shapes[0]) continue;
          assert.ok(insideFrame(boundsOf(sh)), `${action} ${JSON.stringify(box)} p=${i / 20} ${sh.kind}`);
        }
      }
    }
  }
  const top = edgeBoxes[0];
  const flap = (p: number) => boundsOf(paths(motionShapes('fold', 'none', top, at(p), STAGE))[1]);
  assert.ok(near(flap(0).minY, top.top + top.height) && near(flap(0).maxY, top.top + top.height + 30), 'hangs below the bottom edge');
  assert.ok(near(flap(1).minY, top.top + top.height - 30), 'folds up over the box');
  assertPoints(heads(motionShapes('fold', 'none', top, at(0), STAGE))[0], [1, 0], 'fold from below p=0');
  assertPoints(heads(motionShapes('fold', 'none', top, at(0.5), STAGE))[0], [0, -1], 'fold from below goes up');
});

test('place: the ghost lifts over an arc and lands on the provisional destination; the landing mark shows late', () => {
  const target = provisionalTarget('place', BOX);
  assert.ok(near(target.left, BOX.left + BOX.width + 60), 'gap of max(60, 0.4·w) to the right');
  assert.ok(near(target.width, Math.max(BOX.width * 1.3, 80)) && near(target.height, Math.max(BOX.height * 0.45, 24)));
  assert.ok(near(target.top, BOX.top + BOX.height * 0.9));
  const ghost = (p: number) => rects(motionShapes('place', 'none', BOX, at(p)))[0];
  assert.deepEqual([ghost(0).x, ghost(0).y, ghost(0).width, ghost(0).height], [BOX.left, BOX.top, BOX.width, BOX.height]);
  const g1 = ghost(1);
  assert.ok(near(g1.y + g1.height, target.top), 'rests on the destination top');
  assert.ok(near(rectCenter(g1).x, target.left + target.width / 2));
  assert.ok(ghost(0.5).y < Math.min(ghost(0).y, g1.y) - 10, 'lifted mid-way');
  const mark = (p: number) => paths(motionShapes('place', 'none', BOX, at(p)))[1];
  assert.equal(mark(0).opacity, 0);
  assert.ok(mark(1).opacity > 0);
  assert.ok(near(boundsOf(mark(1)).maxY, target.top - 2) && near(boundsOf(mark(1)).minX, target.left));
  // An explicit destination wins.
  const explicit: PxBox = { left: 20, top: 300, width: 100, height: 30 };
  const e1 = rects(motionShapes('place', 'none', BOX, at(1), undefined, explicit))[0];
  assert.ok(near(e1.y + e1.height, 300) && near(rectCenter(e1).x, 70));
});

test('provisional destination: flips left without room on the right and stays 12 px inside the frame', () => {
  const right: PxBox = { left: 520, top: 150, width: 80, height: 60 };
  for (const action of ['place', 'fit'] as const) {
    const t = provisionalTarget(action, right, STAGE);
    assert.ok(t.left + t.width <= right.left, `${action} goes left`);
    assert.ok(t.left >= 12 && t.top >= 12 && t.left + t.width <= STAGE.width - 12 && t.top + t.height <= STAGE.height - 12, action);
  }
  const fit = provisionalTarget('fit', BOX);
  assert.deepEqual([fit.top, fit.width, fit.height], [BOX.top, BOX.width, BOX.height]);
});

test('place / fit near the frame edges keep every shape inside the frame', () => {
  const boxes: PxBox[] = [
    { left: 520, top: 150, width: 80, height: 60 },
    { left: 560, top: 0, width: 80, height: 60 },
    { left: 0, top: 340, width: 80, height: 60 },
    { left: 280, top: 0, width: 80, height: 40 },
  ];
  for (const box of boxes) {
    for (const action of ['place', 'fit'] as const) {
      for (let i = 0; i <= 20; i += 1) {
        for (const sh of motionShapes(action, 'none', box, at(i / 20), STAGE)) {
          assert.ok(insideFrame(boundsOf(sh)), `${action} ${JSON.stringify(box)} p=${i / 20} ${sh.kind}`);
        }
      }
    }
  }
});

test('fit: the ghost slides up against the destination; two arrowheads meet at the junction', () => {
  const t = provisionalTarget('fit', BOX);
  const s1 = motionShapes('fit', 'none', BOX, at(1));
  const [g1] = rects(s1);
  assert.ok(near(g1.x + g1.width, t.left) && near(g1.y, t.top), 'touching the destination');
  assert.deepEqual([rects(motionShapes('fit', 'none', BOX, at(0)))[0].x], [BOX.left]);
  const [junction, a, b] = paths(s1);
  assert.equal(junction.dashed, true);
  assert.ok(near(boundsOf(junction).minX, t.left) && near(boundsOf(junction).maxX, t.left));
  assertPoints(a, [1, 0], 'fit left head');
  assertPoints(b, [-1, 0], 'fit right head');
  assert.equal(heads(motionShapes('fit', 'none', BOX, at(0))).every((hd) => hd.opacity === 0), true, 'heads late');
  // A destination on the left: the ghost ends at its right edge.
  const left: PxBox = { left: 20, top: 150, width: 120, height: 80 };
  const gl = rects(motionShapes('fit', 'none', { ...BOX, left: 300 }, at(1), undefined, left))[0];
  assert.ok(near(gl.x, 140));
});

test('screw: tighten turns clockwise, loosen counterclockwise (mirror images); head ≥8 px', () => {
  // At p=0 the tip is at the top: clockwise goes right. After 1.5 turns it is at the bottom: clockwise goes left.
  assertPoints(heads(motionShapes('screw', 'tighten', BOX, at(0)))[0], [1, 0], 'tighten p=0');
  assertPoints(heads(motionShapes('screw', 'loosen', BOX, at(0)))[0], [-1, 0], 'loosen p=0');
  assertPoints(heads(motionShapes('screw', 'tighten', BOX, at(1)))[0], [-1, 0], 'tighten p=1');
  assertPoints(heads(motionShapes('screw', 'loosen', BOX, at(1)))[0], [1, 0], 'loosen p=1');
  for (const p of [0.3, 0.7, 1]) {
    const cw = paths(motionShapes('screw', 'tighten', BOX, at(p)));
    const ccw = paths(motionShapes('screw', 'loosen', BOX, at(p)));
    cw.forEach((shape, i) => {
      const byYX = (u: [number, number], v: [number, number]) => u[1] - v[1] || u[0] - v[0];
      const a = pathPoints(shape.d).map(([x, y]): [number, number] => [2 * CX - x, y]).sort(byYX);
      const b = pathPoints(ccw[i].d).sort(byYX);
      a.forEach(([x, y], j) => assert.ok(near(x, b[j][0], 0.02) && near(y, b[j][1], 0.02), `screw mirror p=${p} #${i}.${j}`));
    });
  }
  const [head] = circles(motionShapes('screw', 'tighten', BOX, at(1)));
  assert.ok(near(head.r, 0.22 * BOX.height) && near(head.cx, CX) && near(head.cy, CY));
  assert.equal(circles(motionShapes('screw', 'tighten', { left: 0, top: 0, width: 4, height: 4 }, at(1)))[0].r, 8);
  // The arc stays on its circle (head radius + 10).
  const r = 0.22 * BOX.height + 10;
  for (const [x, y] of pathPoints(paths(motionShapes('screw', 'tighten', BOX, at(0.4)))[1].d)) assert.ok(near(Math.hypot(x - CX, y - CY), r, 0.02));
});

test('hold: the press ring contracts first, then the progress ring closes; dot from 35 %', () => {
  const ring = (p: number) => circles(motionShapes('hold', 'none', BOX, at(p)))[0];
  const dot = (p: number) => circles(motionShapes('hold', 'none', BOX, at(p)))[1];
  const progress = (p: number) => paths(motionShapes('hold', 'none', BOX, at(p)))[0];
  assert.ok(near(ring(0).r, 0.55 * Math.max(BOX.width, BOX.height)));
  assert.ok(near(ring(0.35).r, ring(1).r), 'contracted by 35 %');
  assert.ok(ring(1).r < ring(0).r);
  assert.equal(dot(0.2).opacity, 0);
  assert.ok(dot(0.35).opacity > 0);
  assert.equal(progress(0.2).opacity, 0);
  assert.ok(progress(0.6).opacity > 0);
  const pts = pathPoints(progress(1).d);
  const rRing = ring(1).r + 10;
  for (const [x, y] of pts) assert.ok(near(Math.hypot(x - CX, y - CY), rRing, 0.02));
  assert.ok(near(pts[0][0], pts[pts.length - 1][0], 0.02) && near(pts[0][1], pts[pts.length - 1][1], 0.02), 'closed at p=1');
  // Small box: the starting ring is ≥28 px.
  assert.ok(circles(motionShapes('hold', 'none', { left: 290, top: 190, width: 60, height: 20 }, at(0)))[0].r >= 28);
});

test('flip: the ghost squashes to a line mid-way and opens again; the arc runs down the right side', () => {
  const ghost = (p: number) => rects(motionShapes('flip', 'none', BOX, at(p)))[0];
  assert.ok(near(ghost(0).height, BOX.height) && near(ghost(1).height, BOX.height));
  assert.ok(near(ghost(0.5).height, 2), 'edge-on');
  for (const p of [0, 0.5, 1]) assert.ok(near(rectCenter(ghost(p)).y, CY) && near(ghost(p).width, BOX.width));
  const arc = boundsOf(paths(motionShapes('flip', 'none', BOX, at(1)))[1]);
  assert.ok(arc.minX >= BOX.left + BOX.width + 8 - 0.01 && near(arc.minY, CY - BOX.height / 2 - 8, 0.02) && near(arc.maxY, CY + BOX.height / 2 + 8, 0.02));
  assertPoints(heads(motionShapes('flip', 'none', BOX, at(0)))[0], [1, 0], 'flip p=0');
  assertPoints(heads(motionShapes('flip', 'none', BOX, at(0.5)))[0], [0, 1], 'flip p=0.5');
  assertPoints(heads(motionShapes('flip', 'none', BOX, at(1)))[0], [-1, 0], 'flip p=1');
});

const VEC = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] } as const;

test('pull / push: the ghost shifts the given way and the arrowhead points it, in all four directions', () => {
  for (const action of ['pull', 'push'] as const) {
    for (const dir of ['up', 'down', 'left', 'right'] as const) {
      const [vx, vy] = VEC[dir];
      const side = vx !== 0 ? BOX.width : BOX.height;
      const g0 = rects(motionShapes(action, dir, BOX, at(0)))[0];
      const g1 = rects(motionShapes(action, dir, BOX, at(1)))[0];
      assert.deepEqual([g0.x, g0.y], [BOX.left, BOX.top]);
      const shift = Math.min(48, Math.max(16, 0.3 * side));
      assert.ok(near(g1.x - BOX.left, vx * shift) && near(g1.y - BOX.top, vy * shift), `${action}:${dir} shift`);
      for (const p of [0.5, 1]) assertPoints(heads(motionShapes(action, dir, BOX, at(p)))[0], [vx, vy], `${action}:${dir} p=${p}`);
      // pull draws ahead of the ghost's leading edge; push behind its trailing edge.
      const head = boundsOf(heads(motionShapes(action, dir, BOX, at(1)))[0]);
      const along = (b: Bounds) => (vx > 0 ? b.minX : vx < 0 ? -b.maxX : vy > 0 ? b.minY : -b.maxY);
      const leading = vx > 0 ? g1.x + g1.width : vx < 0 ? -g1.x : vy > 0 ? g1.y + g1.height : -g1.y;
      const trailing = vx > 0 ? g1.x : vx < 0 ? -(g1.x + g1.width) : vy > 0 ? g1.y : -(g1.y + g1.height);
      if (action === 'pull') assert.ok(along(head) >= leading, `pull:${dir} head ahead`);
      else {
        const tipAlong = vx > 0 ? head.maxX : vx < 0 ? -head.minX : vy > 0 ? head.maxY : -head.minY;
        assert.ok(tipAlong <= trailing + 1e-6, `push:${dir} head behind`);
      }
    }
  }
});

test('pull / push at the frame edges: everything stays inside, the shift shrinks to the room', () => {
  const edgeBoxes: Record<'up' | 'down' | 'left' | 'right', PxBox> = {
    up: { left: 280, top: 0, width: 80, height: 60 },
    down: { left: 280, top: 340, width: 80, height: 60 },
    left: { left: 0, top: 150, width: 80, height: 60 },
    right: { left: 560, top: 150, width: 80, height: 60 },
  };
  const opposite = { up: 'down', down: 'up', left: 'right', right: 'left' } as const;
  for (const action of ['pull', 'push'] as const) {
    for (const dir of ['up', 'down', 'left', 'right'] as const) {
      // Box touching the edge it moves toward, and the edge behind it.
      for (const box of [edgeBoxes[dir], edgeBoxes[opposite[dir]]]) {
        for (let i = 0; i <= 20; i += 1) {
          for (const sh of motionShapes(action, dir, box, at(i / 20), STAGE)) {
            assert.ok(insideFrame(boundsOf(sh)), `${action}:${dir} ${JSON.stringify(box)} p=${i / 20} ${sh.kind}`);
          }
        }
        assertPoints(heads(motionShapes(action, dir, box, at(1), STAGE))[0], [...VEC[dir]], `${action}:${dir} edge`);
      }
      const g = rects(motionShapes(action, dir, edgeBoxes[dir], at(1), STAGE))[0];
      assert.ok(near(g.x, edgeBoxes[dir].left) && near(g.y, edgeBoxes[dir].top), `${action}:${dir} no shift at the edge`);
    }
  }
  // Some room: push:right shifts to the edge minus 6 px at most.
  const near20: PxBox = { left: 540, top: 150, width: 80, height: 60 };
  const g = rects(motionShapes('push', 'right', near20, at(1), STAGE))[0];
  assert.ok(near(g.x + g.width, STAGE.width - 6), `${g.x + g.width}`);
});

test('new actions: p=0 and p=1 stay near the box (no runaway geometry)', () => {
  for (const [action, direction] of NEW) {
    for (const p of [0, 1]) {
      for (const sh of motionShapes(action, direction, BOX, at(p), STAGE)) {
        const b = boundsOf(sh);
        assert.ok(b.minX > -1 && b.minY > -1 && b.maxX < STAGE.width + 1 && b.maxY < STAGE.height + 1, `${action}:${direction} p=${p}`);
      }
    }
  }
});

// ---------------------------------------------------------------- review 2026-10-03: frame clamping of the new actions

/** Every vertex of `shape` (rect corners, path points, circle bounds), unclipped. */
function vertices(shape: MotionShape): Array<[number, number]> {
  if (shape.kind === 'rect') return [[shape.x, shape.y], [shape.x + shape.width, shape.y + shape.height]];
  if (shape.kind === 'circle') return [[shape.cx - shape.r, shape.cy - shape.r], [shape.cx + shape.r, shape.cy + shape.r]];
  return pathPoints(shape.d);
}
/** Path numbers are rounded to 2 decimals. */
const ROUND_EPS = 0.006;
function assertVerticesInFrame(action: Parameters<typeof motionShapes>[0], direction: Parameters<typeof motionShapes>[1], box: PxBox, tag: string) {
  for (let i = 0; i <= 20; i += 1) {
    for (const sh of motionShapes(action, direction, box, at(i / 20), STAGE)) {
      for (const [x, y] of vertices(sh)) {
        assert.ok(
          x >= -ROUND_EPS && y >= -ROUND_EPS && x <= STAGE.width + ROUND_EPS && y <= STAGE.height + ROUND_EPS,
          `${tag} ${action}:${direction} ${JSON.stringify(box)} p=${i / 20} ${sh.kind} (${x}, ${y})`
        );
      }
    }
  }
}

test('review #1 fit: a wide box keeps the end ghost (and destination) inside the frame', () => {
  const box: PxBox = { left: 200, top: 150, width: 400, height: 80 };
  assertVerticesInFrame('fit', 'none', box, 'wide');
  const g1 = rects(motionShapes('fit', 'none', box, at(1), STAGE))[0];
  assert.ok(g1.x >= 0 && g1.x + g1.width <= STAGE.width, `${g1.x}..${g1.x + g1.width}`);
  assertVerticesInFrame('fit', 'none', { left: 200, top: 150, width: 250, height: 80 }, 'no pair spot');
  // When the pair fits, the provisional destination and the end ghost both lie inside, touching.
  // (A 250 px box at x=200 has no such spot without the ghost moving away from the destination: the gap is then 0.)
  for (const b of [{ left: 200, top: 150, width: 150, height: 80 }, { left: 30, top: 150, width: 250, height: 80 }, { left: 380, top: 20, width: 240, height: 60 }]) {
    const t = provisionalTarget('fit', b, STAGE);
    assert.ok(t.left >= 12 - 1e-6 && t.left + t.width <= STAGE.width - 12 + 1e-6, `target ${JSON.stringify(t)}`);
    const g = rects(motionShapes('fit', 'none', b, at(1), STAGE))[0];
    assert.ok(near(g.x + g.width, t.left) || near(g.x, t.left + t.width), `touching ${JSON.stringify(g)} ${JSON.stringify(t)}`);
    assertVerticesInFrame('fit', 'none', b, 'pair');
  }
});

test('review #2 fold / flip: with no room on either side the arc and its head shrink and move inside', () => {
  assertVerticesInFrame('fold', 'none', { left: 200, top: 50, width: 120, height: 300 }, 'tall');
  assertVerticesInFrame('flip', 'none', { left: 10, top: 150, width: 620, height: 60 }, 'wide');
  assertVerticesInFrame('fold', 'none', { left: 10, top: 10, width: 620, height: 380 }, 'full');
  assertVerticesInFrame('flip', 'none', { left: 10, top: 10, width: 620, height: 380 }, 'full');
});

test('review #3 place: the ghost lands exactly on the destination surface, also at the frame top', () => {
  for (const box of [{ left: 200, top: 0, width: 80, height: 60 }, BOX, { left: 520, top: 300, width: 80, height: 60 }, { left: 280, top: 0, width: 80, height: 300 }]) {
    const g1 = rects(motionShapes('place', 'none', box, at(1), STAGE))[0];
    const mark = paths(motionShapes('place', 'none', box, at(1), STAGE))[1];
    const pts = pathPoints(mark.d);
    // The mark's flat line sits 2 px above the surface; the ghost's bottom is the surface.
    assert.ok(near(g1.y + g1.height, pts[1][1] + 2, ROUND_EPS), `${JSON.stringify(box)}: ghost bottom ${g1.y + g1.height} vs surface ${pts[1][1] + 2}`);
    const t = provisionalTarget('place', box, STAGE);
    assert.ok(near(pts[1][1] + 2, t.top, ROUND_EPS), 'the provisional destination is the surface');
    assertVerticesInFrame('place', 'none', box, 'place');
  }
});

test('review #4 pull / push: the hook, bar and arrowheads stay inside across the motion too', () => {
  for (const action of ['pull', 'push'] as const) {
    for (const dir of ['up', 'down', 'left', 'right'] as const) {
      for (const box of [
        { left: 100, top: 0, width: 60, height: 4 },
        { left: 100, top: 396, width: 60, height: 4 },
        { left: 0, top: 100, width: 4, height: 60 },
        { left: 636, top: 100, width: 4, height: 60 },
        { left: 0, top: 0, width: 4, height: 4 },
      ]) {
        assertVerticesInFrame(action, dir, box, 'thin');
      }
    }
  }
});

test('frame-aware new actions: every vertex inside the frame, unclipped, for edge, corner, tiny, wide and tall boxes', () => {
  const boxes: PxBox[] = [
    BOX,
    { left: 280, top: 0, width: 80, height: 60 },
    { left: 280, top: 340, width: 80, height: 60 },
    { left: 0, top: 150, width: 80, height: 60 },
    { left: 560, top: 150, width: 80, height: 60 },
    { left: 0, top: 0, width: 4, height: 4 },
    { left: 636, top: 396, width: 4, height: 4 },
    { left: 290, top: 190, width: 60, height: 20 },
    { left: 200, top: 150, width: 400, height: 80 },
    { left: 200, top: 50, width: 120, height: 300 },
    { left: 10, top: 150, width: 620, height: 60 },
    { left: 10, top: 10, width: 620, height: 380 },
  ];
  const frameAware = NEW.filter(([action]) => action !== 'screw' && action !== 'hold');
  for (const box of boxes) for (const [action, direction] of frameAware) assertVerticesInFrame(action, direction, box, 'all');
});

test('motionKey is deterministic and distinct per part', () => {
  const base = {
    runId: 3,
    planRevision: 1 as number | null,
    stepId: 's1',
    anchorId: 'a1',
    trackRunId: 'trk-run-1',
    trackId: 'trk-1',
    generation: 2,
  };
  assert.equal(motionKey(base), motionKey({ ...base }));
  const variants = [
    base,
    { ...base, runId: 4 },
    { ...base, planRevision: 2 },
    { ...base, planRevision: null },
    { ...base, stepId: 's2' },
    { ...base, anchorId: 'a2' },
    // A manual reselection keeps anchor id 'a1' but binds a new tracker run / track / generation.
    { ...base, trackRunId: 'trk-run-2' },
    { ...base, trackId: 'trk-2' },
    { ...base, generation: 3 },
  ];
  assert.equal(new Set(variants.map(motionKey)).size, variants.length);
  // Separator-safe: parts that concatenate to the same string still differ.
  assert.notEqual(
    motionKey({ ...base, stepId: 's1:a', anchorId: 'b' }),
    motionKey({ ...base, stepId: 's1', anchorId: 'a:b' })
  );
});

// ---------------------------------------------------------------- schematic glyphs (align / connect / disconnect / bend)

/** The two arrowheads of one path (each a 3-point triangle) as unit vectors, tip minus base midpoint. */
function headPair(shape: MotionShape): Array<[number, number]> {
  const pts = pathPoints(shape.d);
  assert.equal(pts.length, 6, 'two arrowheads in one path');
  return [0, 3].map((i) => {
    const [tip, b1, b2] = [pts[i], pts[i + 1], pts[i + 2]];
    const dx = tip[0] - (b1[0] + b2[0]) / 2;
    const dy = tip[1] - (b1[1] + b2[1]) / 2;
    const len = Math.hypot(dx, dy);
    return [dx / len, dy / len];
  });
}
/** Whether a unit vector points along the x axis toward `side` (inward: +1 is rightward). */
const alongX = (v: [number, number], side: 1 | -1) => v[0] * side > 0.99 && Math.abs(v[1]) < 0.01;

test('schematic glyphs stay inside the frame for corner, edge, tiny and large boxes', () => {
  const boxes: PxBox[] = [
    { left: 0, top: 0, width: 60, height: 20 }, // flush at the top-left corner
    { left: 580, top: 380, width: 60, height: 20 }, // flush at the bottom-right corner
    { left: 0, top: 190, width: 20, height: 20 }, // flush at the left edge
    { left: 300, top: 190, width: 4, height: 4 }, // tiny, away from the edges
    { left: 40, top: 40, width: 560, height: 320 }, // nearly the whole stage
    { left: 290, top: 190, width: 60, height: 20 }, // the preview's small preset
  ];
  for (const box of boxes) {
    for (const [action, direction] of SCHEMATIC) {
      const where = ` ${action} box=${box.left},${box.top} ${box.width}×${box.height}`;
      for (let i = 0; i <= 40; i += 1) {
        const p = i / 40;
        for (const shape of motionShapes(action, direction, box, at(p), STAGE)) {
          const b = boundsOf(shape);
          assert.ok(insideFrame(b), `${where} p=${p} ${shape.kind}: ${JSON.stringify(b)}`);
        }
      }
      // The frame-clamped extent is inside the frame too, and still contains every shape it was fitted for.
      const e = motionExtent(action, direction, box, STAGE);
      assert.ok(insideFrame({ minX: e.left, minY: e.top, maxX: e.left + e.width, maxY: e.top + e.height }), `${where} extent`);
      for (let i = 0; i <= 40; i += 1) {
        const p = i / 40;
        for (const shape of motionShapes(action, direction, box, at(p), STAGE)) {
          const b = boundsOf(shape);
          assert.ok(
            b.minX - 1.5 >= e.left - 1e-6 &&
              b.minY - 1.5 >= e.top - 1e-6 &&
              b.maxX + 1.5 <= e.left + e.width + 1e-6 &&
              b.maxY + 1.5 <= e.top + e.height + 1e-6,
            `${where} p=${p} ${shape.kind} outside the extent`
          );
        }
      }
    }
  }
});

test('align: the component starts offset and settles centred on the dashed guide', () => {
  const ghostCenter = (p: number) => rectCenter(rects(motionShapes('align', 'none', BOX, at(p)))[0]);
  const start = ghostCenter(0);
  const end = motionFrame(0, true); // reduced motion: the held end pose
  assert.equal(end.progress, 1);
  const endCenter = ghostCenter(end.progress);
  assert.ok(start.x > CX + 10, `starts right of the guide: ${(start.x - CX).toFixed(1)}`);
  assert.ok(near(start.y, CY), 'runs on the box centre line');
  assert.ok(near(endCenter.x, CX) && near(endCenter.y, CY), 'ends centred on the guide');
  // The guide: the dashed line through the box centre, longer than the component it takes in.
  const guide = paths(motionShapes('align', 'none', BOX, end))[0];
  const gb = boundsOf(guide);
  assert.ok(guide.dashed === true && gb.minX < CX && gb.maxX > CX && gb.maxY - gb.minY > BOX.height / 2, 'guide crosses the centre');
  // The arrowhead leads the component leftward and survives into the held end pose.
  const head = heads(motionShapes('align', 'none', BOX, end))[0];
  assert.ok(head.opacity > 0);
  assertPoints(head, [-1, 0], 'align end');
});

test('connect mates the two halves on the middle line; disconnect pulls them apart', () => {
  /** The gap between the halves' facing edges (pins included) and the way their arrowheads point. */
  const pose = (action: 'connect' | 'disconnect', p: number) => {
    const shapes = motionShapes(action, 'none', BOX, at(p));
    const [left, right, guide] = paths(shapes);
    const head = heads(shapes)[0];
    return { gap: boundsOf(right).minX - boundsOf(left).maxX, heads: headPair(head), headOpacity: head.opacity, guide };
  };
  const end = motionFrame(0, true).progress; // reduced motion: the held end pose
  const connectEnd = pose('connect', end);
  const disconnectEnd = pose('disconnect', end);
  assert.ok(near(connectEnd.gap, 0, 0.02), `connect ends mated: ${connectEnd.gap.toFixed(2)}`);
  assert.ok(disconnectEnd.gap > 30, `disconnect ends apart: ${disconnectEnd.gap.toFixed(1)}`);
  // Each held end pose carries its own direction: pushing together vs pulling apart.
  assert.ok(alongX(connectEnd.heads[0], 1) && alongX(connectEnd.heads[1], -1), 'connect heads point inward');
  assert.ok(alongX(disconnectEnd.heads[0], -1) && alongX(disconnectEnd.heads[1], 1), 'disconnect heads point outward');
  assert.ok(connectEnd.headOpacity > 0 && disconnectEnd.headOpacity > 0, 'heads visible in the end pose');
  // The two start where the other ends, and both keep the dashed mating guide on the middle line.
  assert.ok(pose('connect', 0).gap > 30 && near(pose('disconnect', 0).gap, 0, 0.02), 'start poses are the reverse');
  for (const action of ['connect', 'disconnect'] as const) {
    for (const p of [0, 0.5, 1]) {
      assert.ok(pose(action, p).guide.dashed === true, `${action} mating guide p=${p}`);
    }
  }
});

test('bend runs the lead to a hinge and swings the leg off the straight continuation', () => {
  const pose = (p: number) => {
    const shapes = motionShapes('bend', 'none', BOX, at(p));
    const [lead, guide, arc] = paths(shapes);
    const [entry, hinge, tip] = pathPoints(lead.d);
    return { guide, arc, entry, hinge, tip, joint: circles(shapes)[0], head: heads(shapes)[0] };
  };
  const start = pose(0);
  const end = pose(motionFrame(0, true).progress); // reduced motion: the held end pose
  assert.ok(near(start.tip[0], start.hinge[0]) && start.tip[1] > start.hinge[1], 'starts straight out of the box');
  assert.ok(near(start.entry[0], CX) && near(start.entry[1], BOX.top + BOX.height), 'leaves the box at its bottom edge');
  assert.ok(start.hinge[1] > BOX.top + BOX.height, 'the hinge sits past the box edge');
  const legLen = (q: { hinge: [number, number]; tip: [number, number] }) => Math.hypot(q.tip[0] - q.hinge[0], q.tip[1] - q.hinge[1]);
  const angle = (Math.atan2(end.tip[1] - end.hinge[1], end.tip[0] - end.hinge[0]) * 180) / Math.PI;
  assert.ok(end.tip[0] - end.hinge[0] > 10, 'the leg ends swung to the side');
  assert.ok(angle > 15 && angle < 60, `bent to ${angle.toFixed(0)}° off straight`);
  assert.ok(near(legLen(end), legLen(start), 0.05) && legLen(start) > 20, 'the leg keeps its length');
  // The end pose still shows the joint, the unbent continuation and the angle bent between them.
  assert.ok(near(end.joint.cx, end.hinge[0]) && near(end.joint.cy, end.hinge[1]) && end.joint.filled === true, 'joint dot at the hinge');
  const gb = boundsOf(end.guide);
  assert.ok(end.guide.dashed === true && near(gb.minX, end.hinge[0]) && gb.maxY - gb.minY > 30, 'straight guide drawn');
  const arcEnd = boundsOf(end.arc);
  const arcStart = boundsOf(start.arc);
  assert.ok(arcEnd.maxX - arcEnd.minX > 4 && arcEnd.maxY - arcEnd.minY > 4, 'the arc opens with the swing');
  assert.ok(near(arcStart.maxX - arcStart.minX, 0, 0.02) && near(arcStart.maxY - arcStart.minY, 0, 0.02), 'no angle at the start');
  const tip = pathPoints(end.head.d)[0];
  assert.ok(near(tip[0], end.tip[0], 0.05) && near(tip[1], end.tip[1], 0.05), 'the head rides the free end');
});
