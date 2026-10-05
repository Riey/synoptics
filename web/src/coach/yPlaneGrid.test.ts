/** The Y-plane appearance grid (v2 stream samples): exact area means, range mapping, mirror, alignment. */
import test from 'node:test';
import assert from 'node:assert/strict';

const { evenRect, yPlaneToGrid } = await import('./yPlaneGrid.ts');

test('the crop is grown to even coordinates and clamped to the frame', () => {
  assert.deepEqual(evenRect({ x: 71, y: 101, width: 499, height: 375 }, { width: 1280, height: 720 }), {
    x: 70,
    y: 100,
    width: 500,
    height: 376,
  });
  assert.deepEqual(evenRect({ x: 1279, y: 0, width: 5, height: 5 }, { width: 1280, height: 720 }), {
    x: 1278,
    y: 0,
    width: 2,
    height: 6,
  });
  assert.equal(evenRect({ x: 0, y: 0, width: 1, height: 1 }, { width: 1, height: 1 }), null);
  assert.deepEqual(evenRect({ x: 1270, y: 715, width: 20, height: 20 }, { width: 1280, height: 720 }), {
    x: 1270,
    y: 714,
    width: 10,
    height: 6,
  });
});

test('each cell is the exact mean of its samples; limited range maps 16..235 to 0..1', () => {
  // 4×2 plane with stride 6 (2 padding bytes per row), grid 2: left cells 16/235, right cells mixed.
  const plane = new Uint8Array([16, 16, 235, 16, 99, 99, 16, 16, 235, 235, 99, 99]);
  const grid = yPlaneToGrid(plane, 4, 2, 6, 2, false, false);
  assert.equal(grid[0], 0);
  assert.ok(Math.abs(grid[1] - 0.5) < 1e-6, 'right half of row 0: (235+16)/2 → 0.5');
  assert.equal(grid[2], 0);
  assert.equal(grid[3], 1);
  const full = yPlaneToGrid(new Uint8Array([0, 255, 0, 255]), 2, 2, 2, 2, true, false);
  assert.deepEqual([...full], [0, 1, 0, 1]);
});

test('mirror reverses the columns only', () => {
  const plane = new Uint8Array([0, 51, 102, 153, 204, 255, 0, 51, 102, 153, 204, 255]);
  const plain = yPlaneToGrid(plane, 6, 2, 6, 2, true, false);
  const mirrored = yPlaneToGrid(plane, 6, 2, 6, 2, true, true);
  assert.deepEqual([...mirrored], [plain[1], plain[0], plain[3], plain[2]]);
});

test('a crop smaller than the grid still fills every cell', () => {
  const grid = yPlaneToGrid(new Uint8Array([100, 200, 100, 200]), 2, 2, 2, 4, true, false);
  assert.equal(grid.length, 16);
  assert.ok(grid.every((value) => value > 0));
});
