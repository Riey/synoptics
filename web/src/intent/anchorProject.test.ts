/**
 * Anchor binding, focus-ring math and label placement (pure; no value imports, so node runs it directly).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  actionShortLabel,
  isValidActionDirection,
  DEFAULT_FOCUS_PAD,
  PRIMARY_ANCHOR_ID,
  anchorLabel,
  bindStepCommands,
  bindingFromSnapshot,
  buildAnchorRef,
  computeLeaderLine,
  echoOf,
  focusRing,
  labelPlacement,
  placeLabel,
  resolveFocus,
  snapshotMatchesBinding,
} from './anchorProject.ts';

const BOX = { x: 0.4, y: 0.4, width: 0.2, height: 0.1 };
const near = (a: number, b: number) => Math.abs(a - b) < 1e-9;

function snap(overrides: Record<string, unknown> = {}) {
  return {
    runId: 'run-1',
    trackId: 'trk-1',
    generation: 2,
    state: 'tracking' as const,
    box: BOX,
    capturedAt: 1000,
    version: 1,
    target: 'glasses',
    ...overrides,
  };
}

test('focus ring grows the box by pad on every side, in box units', () => {
  const ring = focusRing(BOX, 0.25);
  assert.ok(near(ring.x, 0.35) && near(ring.y, 0.375), JSON.stringify(ring));
  assert.ok(near(ring.width, 0.3) && near(ring.height, 0.15), JSON.stringify(ring));
});

test('focus ring uses the contract default pad and clamps pad to [0, 0.5]', () => {
  const byDefault = focusRing(BOX);
  assert.ok(near(byDefault.x, BOX.x - BOX.width * DEFAULT_FOCUS_PAD));
  const huge = focusRing(BOX, 9);
  assert.ok(near(huge.width, BOX.width * 2), 'pad above 0.5 is clamped to 0.5');
  const negative = focusRing(BOX, -1);
  assert.ok(near(negative.width, BOX.width) && near(negative.x, BOX.x));
});

test('focus ring never leaves the frame', () => {
  const ring = focusRing({ x: 0, y: 0.95, width: 0.3, height: 0.05 }, 0.5);
  assert.equal(ring.x, 0);
  assert.ok(ring.y + ring.height <= 1 + 1e-12);
  assert.ok(ring.x >= 0 && ring.y >= 0);
});

test('label sits offset px above the box top-left in overlay CSS px', () => {
  const placed = labelPlacement(BOX, { width: 1000, height: 500 }, 8, { width: 120, height: 24 });
  assert.deepEqual(placed, { left: 400, top: 192, placement: 'above' });
});

test('label goes below the box when there is no room above, and stays inside horizontally', () => {
  const top = { x: 0.95, y: 0.01, width: 0.04, height: 0.1 };
  const placed = labelPlacement(top, { width: 1000, height: 500 }, 8, { width: 120, height: 24 });
  assert.equal(placed.placement, 'below');
  assert.ok(near(placed.top, (0.01 + 0.1) * 500 + 8));
  assert.equal(placed.left, 1000 - 120);
});

test('label placement scales with the overlay size (positions, never text scaling)', () => {
  const small = labelPlacement(BOX, { width: 500, height: 250 }, 8);
  const large = labelPlacement(BOX, { width: 1000, height: 500 }, 8);
  assert.equal(large.left, small.left * 2);
  assert.equal(small.top, 0.4 * 250 - 8);
});

test('plan commands bind from the role to the anchor id; other anchors are dropped', () => {
  const bound = bindStepCommands(
    {
      commands: [
        { kind: 'focus', anchor: 'target' },
        { kind: 'label', anchor: 'target', text: '안경' },
        { kind: 'focus', anchor: 'a7', pad: 0.2 },
      ],
    },
    PRIMARY_ANCHOR_ID
  );
  assert.deepEqual(bound, [
    { kind: 'focus', anchor: 'a1', pad: DEFAULT_FOCUS_PAD },
    { kind: 'label', anchor: 'a1', text: '안경' },
  ]);
  assert.deepEqual(bindStepCommands(null, 'a1'), []);
});

test('binding is taken only from a tracking snapshot and matches run, track and generation', () => {
  assert.equal(bindingFromSnapshot(snap({ state: 'occluded', box: null })), null);
  const binding = bindingFromSnapshot(snap());
  assert.deepEqual(binding, { anchorId: 'a1', runId: 'run-1', trackId: 'trk-1', generation: 2 });
  assert.equal(snapshotMatchesBinding(snap(), binding), true);
  assert.equal(snapshotMatchesBinding(snap({ generation: 3 }), binding), false);
  assert.equal(snapshotMatchesBinding(snap({ runId: 'run-2' }), binding), false);
  assert.equal(snapshotMatchesBinding(null, binding), false);
});

test('AnchorRef carries the box only while tracking and is never the role id', () => {
  const tracking = buildAnchorRef(snap(), 'glasses');
  assert.deepEqual(tracking, {
    anchor_id: 'a1',
    role: 'target',
    label: 'glasses',
    run_id: 'run-1',
    track_id: 'trk-1',
    generation: 2,
    state: 'tracking',
    box: BOX,
  });
  const lost = buildAnchorRef(snap({ state: 'lost', box: null }), 'glasses');
  assert.equal(lost?.state, 'lost');
  assert.equal('box' in (lost ?? {}), false);
  assert.equal(buildAnchorRef(snap(), 'glasses', 'target'), null);
  assert.equal(buildAnchorRef(null, 'glasses'), null);
  assert.deepEqual(echoOf(tracking), [{ anchor_id: 'a1', track_id: 'trk-1', generation: 2 }]);
  assert.deepEqual(echoOf(null), []);
});

test('anchor label prefers the plan noun, falls back, and respects the 40-char limit', () => {
  assert.equal(anchorLabel('glasses', 'user-selected object'), 'glasses');
  assert.equal(anchorLabel('  ', 'user-selected object'), 'user-selected object');
  assert.equal(anchorLabel(null, null), 'object');
  assert.equal(anchorLabel('x'.repeat(60), null).length, 40);
});

test('placeLabel: above the box when free, below when the top is taken by an obstacle, never over the box', () => {
  const overlay = { width: 1000, height: 500 };
  const size = { width: 120, height: 24 };
  const box = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 }; // px 400..600 x 200..300
  assert.deepEqual(placeLabel(box, overlay, size), { left: 400, top: 168, placement: 'above', overlap: 0 });
  // A narration band across the top 0..190 px: the label goes below the box.
  const band = { left: 0, top: 0, width: 1000, height: 190 };
  assert.deepEqual(placeLabel(box, overlay, size, [band]), { left: 400, top: 308, placement: 'below', overlap: 0 });
  // Top-right box under a chip at the top right: below, kept inside horizontally.
  const topRight = { x: 0.8, y: 0.12, width: 0.18, height: 0.3 };
  const chip = { left: 860, top: 36, width: 132, height: 26 };
  const placed = placeLabel(topRight, overlay, size, [chip]);
  assert.equal(placed.placement, 'below');
  assert.equal(placed.overlap, 0);
  assert.ok(placed.left + size.width <= 1000);
  // Bottom-centre box with the band at the top: room above → above.
  const bottom = { x: 0.4, y: 0.7, width: 0.2, height: 0.28 };
  const low = placeLabel(bottom, overlay, size, [band]);
  assert.equal(low.placement, 'above');
  assert.ok(low.top + size.height <= 0.7 * 500);
});

test('placeLabel: beside the box when neither above nor below fits; least overlap as the last resort', () => {
  const overlay = { width: 1000, height: 500 };
  const size = { width: 120, height: 24 };
  const tall = { x: 0.3, y: 0.02, width: 0.2, height: 0.96 };
  const side = placeLabel(tall, overlay, size);
  assert.equal(side.placement, 'right');
  assert.equal(side.left, 0.5 * 1000 + 8);
  const blocked = placeLabel(tall, overlay, size, [{ left: 500, top: 0, width: 500, height: 500 }]);
  assert.equal(blocked.placement, 'left');
  assert.equal(blocked.left + size.width, 300 - 8);
  // Everything blocked: an inside spot with the least overlap, still not over the box.
  const all = placeLabel(tall, overlay, size, [{ left: 0, top: 0, width: 1000, height: 500 }]);
  assert.ok(all.overlap > 0);
  assert.ok(all.left + size.width <= 300 || all.left >= 500, 'never covers the box');
});

test('placeLabel: beside a tall box, the label moves down the side to clear a band at the top', () => {
  const overlay = { width: 572, height: 322 };
  const size = { width: 120, height: 24 };
  const tall = { x: 0.6, y: 0.05, width: 0.2, height: 0.93 }; // px 343..458 x 16..315: no room above or below
  const band = { left: 8, top: 40, width: 375, height: 92 };
  const chip = { left: 410, top: 40, width: 155, height: 26 };
  const placed = placeLabel(tall, overlay, size, [band, chip]);
  assert.equal(placed.overlap, 0);
  assert.equal(placed.placement, 'left');
  assert.ok(placed.top >= 132, `${placed.top} clears the band`);
  assert.ok(placed.left + size.width <= 0.6 * 572 - 8 + 1e-9, 'never over the box');
});

test('bindStepCommands preserves valid action commands and enforces directional rules', () => {
  const bound = bindStepCommands(
    {
      commands: [
        { kind: 'action', anchor: 'target', action: 'move', direction: 'up' },
        { kind: 'action', anchor: 'target', action: 'rotate', direction: 'clockwise' },
        { kind: 'action', anchor: 'target', action: 'press', direction: 'none' },
        // Invalid combinations:
        { kind: 'action', anchor: 'target', action: 'move', direction: 'none' }, // move requires cardinal
        { kind: 'action', anchor: 'target', action: 'rotate', direction: 'up' }, // rotate requires cw/ccw
        { kind: 'action', anchor: 'target', action: 'press', direction: 'up' }, // press requires none
        { kind: 'action', anchor: 'target', action: 'press' }, // missing direction is not inferred
        { kind: 'action', anchor: 'other', action: 'press', direction: 'none' }, // other anchor dropped
      ],
    },
    PRIMARY_ANCHOR_ID
  );

  assert.deepEqual(bound, [
    { kind: 'action', anchor: 'a1', action: 'move', direction: 'up' },
    { kind: 'action', anchor: 'a1', action: 'rotate', direction: 'clockwise' },
    { kind: 'action', anchor: 'a1', action: 'press', direction: 'none' },
  ]);
});


test('bindStepCommands: the new actions follow the contract direction rules', () => {
  const ok = [
    ['attach', 'none'], ['fold', 'none'], ['place', 'none'], ['fit', 'none'], ['hold', 'none'], ['flip', 'none'],
    ['screw', 'tighten'], ['screw', 'loosen'],
    ['pull', 'up'], ['pull', 'down'], ['pull', 'left'], ['pull', 'right'],
    ['push', 'up'], ['push', 'down'], ['push', 'left'], ['push', 'right'],
    // The graph core's schematic actions (2026-10-05): no direction.
    ['align', 'none'], ['connect', 'none'], ['disconnect', 'none'], ['bend', 'none'],
  ];
  const bad = [
    ['screw', 'none'], ['screw', 'clockwise'], ['pull', 'none'], ['push', 'tighten'], ['fold', 'up'],
    ['place', 'loosen'], ['rotate', 'tighten'], ['move', 'loosen'], ['twist', 'none'],
    ['align', 'up'], ['connect', 'tighten'], ['disconnect', 'left'], ['bend', 'clockwise'],
  ];
  const commands = [...ok, ...bad].map(([action, direction]) => ({ kind: 'action', anchor: 'target', action, direction }));
  const bound = bindStepCommands({ commands } as never, PRIMARY_ANCHOR_ID);
  assert.deepEqual(
    bound.map((c) => (c.kind === 'action' ? `${c.action}:${c.direction}` : c.kind)),
    ok.map(([a, d]) => `${a}:${d}`)
  );
  assert.equal(isValidActionDirection('screw', 'tighten'), true);
  assert.equal(isValidActionDirection('hold', 'tighten'), false);
});

test('actionShortLabel: Korean labels for every action and direction', () => {
  const want: Array<[Parameters<typeof actionShortLabel>[0], Parameters<typeof actionShortLabel>[1], string]> = [
    ['move', 'up', '위로 이동'],
    ['rotate', 'counterclockwise', '반시계방향 회전'],
    ['press', 'none', '누르기'],
    ['attach', 'none', '붙이기'],
    ['fold', 'none', '접기'],
    ['place', 'none', '올려놓기'],
    ['fit', 'none', '맞추기'],
    ['screw', 'tighten', '조이기'],
    ['screw', 'loosen', '풀기'],
    ['hold', 'none', '누르고 있기'],
    ['flip', 'none', '뒤집기'],
    ['pull', 'down', '아래로 당기기'],
    ['pull', 'left', '왼쪽으로 당기기'],
    ['push', 'up', '위로 밀기'],
    ['push', 'right', '오른쪽으로 밀기'],
    // The graph core's schematic actions: one distinct label each (align is 정렬, not fit's 맞추기).
    ['align', 'none', '정렬'],
    ['connect', 'none', '연결하기'],
    ['disconnect', 'none', '분리하기'],
    ['bend', 'none', '구부리기'],
  ];
  for (const [action, direction, label] of want) assert.equal(actionShortLabel(action, direction), label, `${action}:${direction}`);
});

test('computeLeaderLine connects callout edge to target box edge according to placement', () => {
  const avoid = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 };
  const overlay = { width: 1000, height: 1000 };
  // Box in px: left 400, top 400, right 600, bottom 600

  // Placed above
  const aboveCallout = { left: 450, top: 350, width: 100, height: 40 };
  const aboveLine = computeLeaderLine(avoid, overlay, aboveCallout, 'above');
  assert.equal(aboveLine.from.y, 390); // callout bottom
  assert.equal(aboveLine.to.y, 400); // box top

  // Placed below
  const belowCallout = { left: 450, top: 610, width: 100, height: 40 };
  const belowLine = computeLeaderLine(avoid, overlay, belowCallout, 'below');
  assert.equal(belowLine.from.y, 610); // callout top
  assert.equal(belowLine.to.y, 600); // box bottom

  // Placed right
  const rightCallout = { left: 610, top: 450, width: 100, height: 40 };
  const rightLine = computeLeaderLine(avoid, overlay, rightCallout, 'right');
  assert.equal(rightLine.from.x, 610); // callout left
  assert.equal(rightLine.to.x, 600); // box right

  // Placed left
  const leftCallout = { left: 290, top: 450, width: 100, height: 40 };
  const leftLine = computeLeaderLine(avoid, overlay, leftCallout, 'left');
  assert.equal(leftLine.from.x, 390); // callout right (290 + 100)
  assert.equal(leftLine.to.x, 400); // box left
});

test('narrow camera callout avoids the instruction strip, status chips and target', () => {
  const overlay = { width: 299, height: 199.328125 };
  const size = { width: 152, height: 40 };
  const obstacles = [
    { left: 141.28125, top: 40, width: 149.71875, height: 26.1875 },
    { left: 8, top: 70, width: 92, height: 80.1875 },
    { left: 177.859375, top: -4.203125, width: 113.140625, height: 29.5 },
  ];
  const target = focusRing({ x: 0.34, y: 0.53, width: 0.22, height: 0.28 }, 0.1);
  const placed = placeLabel(target, overlay, size, obstacles);
  assert.equal(placed.overlap, 0);
  assert.ok(placed.left >= 0 && placed.top >= 0);
  assert.ok(placed.left + size.width <= overlay.width && placed.top + size.height <= overlay.height);
  for (const rect of [...obstacles, {
    left: target.x * overlay.width, top: target.y * overlay.height,
    width: target.width * overlay.width, height: target.height * overlay.height,
  }]) {
    assert.ok(placed.left + size.width <= rect.left || placed.left >= rect.left + rect.width
      || placed.top + size.height <= rect.top || placed.top >= rect.top + rect.height);
  }
});

test('resolveFocus: the focus command wins; an action alone implies the default ring; a label alone has none', () => {
  const focus = { kind: 'focus' as const, anchor: 'a1', pad: 0.3 };
  const action = { kind: 'action' as const, anchor: 'a1', action: 'remove' as const, direction: 'none' as const };
  const label = { kind: 'label' as const, anchor: 'a1', text: '안경' };
  assert.deepEqual(resolveFocus([focus, action]), { pad: 0.3, source: 'command' });
  assert.deepEqual(resolveFocus([action, label]), { pad: DEFAULT_FOCUS_PAD, source: 'implied' });
  assert.deepEqual(resolveFocus([action]), { pad: DEFAULT_FOCUS_PAD, source: 'implied' });
  assert.equal(resolveFocus([label]), null);
  assert.equal(resolveFocus([]), null);
});

test('placeLabel keeps the preferred side while it still fits, and switches when it no longer does', () => {
  const overlay = { width: 1000, height: 500 };
  const size = { width: 120, height: 24 };
  const box = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 }; // px 400..600 x 200..300
  // Both above and below are free: no preference → above (first); preferred below → below.
  assert.equal(placeLabel(box, overlay, size).placement, 'above');
  assert.deepEqual(placeLabel(box, overlay, size, [], 8, 'below'), { left: 400, top: 308, placement: 'below', overlap: 0 });
  assert.equal(placeLabel(box, overlay, size, [], 8, 'right').placement, 'right');
  // The preferred side is now covered by an obstacle: the best free position wins.
  const lowBand = { left: 0, top: 305, width: 1000, height: 60 };
  assert.deepEqual(placeLabel(box, overlay, size, [lowBand], 8, 'below'), { left: 400, top: 168, placement: 'above', overlap: 0 });
  // A preferred 'above' never sends the callout to a far corner when the adjacent above spots are taken.
  // (Only the strip right above the box is taken; the top-left corner (0,0) is free but is not "above the box".)
  const overTop = { left: 380, top: 150, width: 260, height: 50 };
  const placed = placeLabel(box, overlay, size, [overTop], 8, 'above');
  assert.equal(placed.placement, 'below');
  assert.equal(placed.overlap, 0);
});
