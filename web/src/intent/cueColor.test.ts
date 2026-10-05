/**
 * Cue colour: contrast accent choice, tone, and the frame sample's background/object means (pure).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  ACCENT_CANDIDATES,
  BG_MIN_CONTRAST,
  PREFERRED_ACCENT,
  SAMPLE_WIDTH_PX,
  accentTone,
  contrastRatio,
  hexRgb,
  hueChroma,
  hueGap,
  mirrorUnitBox,
  pickContrastAccent,
  relativeLuminance,
  rgbHex,
  sampleSize,
  sceneColorsFromPixels,
} from './cueColor.ts';

/** The designer's five preview scenes (background, object). */
const SCENES = {
  dark: { bg: '#0f172a', obj: '#334155' },
  white: { bg: '#eef2f6', obj: '#c7d0db' },
  wood: { bg: '#9a6a3a', obj: '#2b2523' },
  sky: { bg: '#7dd3fc', obj: '#e0f2fe' },
  green: { bg: '#166534', obj: '#fde047' },
} as const;

test('hex parsing and formatting', () => {
  assert.deepEqual(hexRgb('#fff'), [1, 1, 1]);
  assert.deepEqual(hexRgb('000000'), [0, 0, 0]);
  assert.deepEqual(hexRgb('#38bdf8').map((v) => Math.round(v * 255)), [0x38, 0xbd, 0xf8]);
  assert.deepEqual(hexRgb('not a colour'), [0, 0, 0]);
  assert.deepEqual(hexRgb('#12345'), [0, 0, 0]);
  assert.equal(rgbHex(56, 189, 248), '#38bdf8');
  assert.equal(rgbHex(-5, 300, Number.NaN), '#00ff00');
});

test('luminance, contrast, hue', () => {
  assert.equal(relativeLuminance([0, 0, 0]), 0);
  assert.ok(Math.abs(relativeLuminance([1, 1, 1]) - 1) < 1e-9);
  assert.ok(Math.abs(contrastRatio([0, 0, 0], [1, 1, 1]) - 21) < 1e-9);
  assert.equal(contrastRatio([0.4, 0.4, 0.4], [0.4, 0.4, 0.4]), 1);
  assert.equal(Math.round(hueChroma([1, 0, 0]).h), 0);
  assert.equal(Math.round(hueChroma([0, 1, 0]).h), 120);
  assert.equal(Math.round(hueChroma([0, 0, 1]).h), 240);
  assert.equal(hueChroma([0.5, 0.5, 0.5]).c, 0);
  assert.equal(hueGap(350, 10), 20);
  assert.equal(hueGap(0, 180), 180);
});

test('the preferred sky accent stays on a dark scene', () => {
  assert.equal(pickContrastAccent(SCENES.dark.bg, SCENES.dark.obj), PREFERRED_ACCENT);
  assert.equal(PREFERRED_ACCENT, '#38bdf8');
});

test('the designer scenes get an accent with ≥3:1 contrast to the background', () => {
  const expected: Record<keyof typeof SCENES, string> = {
    dark: '#38bdf8',
    white: '#1d4ed8',
    wood: '#a3e635',
    sky: '#6d28d9',
    green: '#38bdf8',
  };
  for (const [name, scene] of Object.entries(SCENES) as Array<[keyof typeof SCENES, { bg: string; obj: string }]>) {
    const accent = pickContrastAccent(scene.bg, scene.obj);
    assert.equal(accent, expected[name], name);
    assert.ok(contrastRatio(hexRgb(accent), hexRgb(scene.bg)) >= BG_MIN_CONTRAST, name);
  }
  // A sky-blue cloth never gets the sky accent (same hue, too little contrast).
  assert.notEqual(pickContrastAccent(SCENES.sky.bg, SCENES.sky.obj), '#38bdf8');
});

test('a passing preferred accent wins; otherwise the candidates are tried in order', () => {
  assert.equal(pickContrastAccent('#000000', '#000000', '#facc15'), '#facc15');
  // Yellow on white fails; the first passing candidate is the dark blue.
  assert.equal(pickContrastAccent('#ffffff', '#ffffff', '#facc15'), '#1d4ed8');
});

test('with no candidate passing, the best partial score is returned (always a candidate)', () => {
  // Mid grey: nothing reaches 3:1 against it except white/dark navy, whose object contrast decides.
  const accent = pickContrastAccent('#767676', '#767676');
  assert.ok([PREFERRED_ACCENT, ...ACCENT_CANDIDATES].includes(accent), accent);
  for (const bg of ['#808080', '#777777', '#5a5a5a', '#999999']) {
    const a = pickContrastAccent(bg, bg);
    assert.match(a, /^#[0-9a-f]{6}$/);
  }
});

test('accent tone: light accent → dark halo and callout; dark accent → light ones', () => {
  const light = accentTone('#38bdf8');
  assert.equal(light.halo, 'rgba(2, 6, 23, 0.95)');
  assert.equal(light.bubbleBg, 'rgba(15, 23, 42, 0.94)');
  assert.equal(light.bubbleFg, '#f8fafc');
  assert.equal(light.glow, 'rgba(56, 189, 248, 0.5)');
  assert.equal(light.accent, '#38bdf8');
  const dark = accentTone('#1d4ed8');
  assert.equal(dark.halo, 'rgba(248, 250, 252, 0.95)');
  assert.equal(dark.bubbleFg, '#0f172a');
  assert.equal(dark.glow, 'rgba(29, 78, 216, 0.35)');
  // The callout text keeps ≥4.5:1 on its own background in both tones.
  assert.ok(contrastRatio(hexRgb(light.bubbleFg), hexRgb('#0f172a')) >= 4.5);
  assert.ok(contrastRatio(hexRgb(dark.bubbleFg), hexRgb('#f8fafc')) >= 4.5);
});

/** An RGBA image of `w`×`h` filled with `bg`, with `obj` painted over the pixel rect [x0, x1)×[y0, y1). */
function image(w: number, h: number, bg: number[], obj: number[], rect: [number, number, number, number]) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const inside = x >= rect[0] && x < rect[1] && y >= rect[2] && y < rect[3];
      const c = inside ? obj : bg;
      data.set([c[0], c[1], c[2], 255], (y * w + x) * 4);
    }
  }
  return data;
}

test('scene sample: band around the box = background, inner 60 % = object', () => {
  const W = 96;
  const H = 54;
  // Object pixels 30..60 × 15..40; the box is exactly that rect.
  const data = image(W, H, [154, 106, 58], [43, 37, 35], [30, 60, 15, 40]);
  const box = { x: 30 / W, y: 15 / H, width: 30 / W, height: 25 / H };
  assert.deepEqual(sceneColorsFromPixels(data, W, H, box), { bg: '#9a6a3a', obj: '#2b2523' });
  // A frame stripe far from the box does not count as background.
  const striped = image(W, H, [154, 106, 58], [43, 37, 35], [30, 60, 15, 40]);
  for (let y = 0; y < H; y += 1) striped.set([255, 0, 0, 255], (y * W + 0) * 4);
  assert.equal(sceneColorsFromPixels(striped, W, H, box)?.bg, '#9a6a3a');
});

test('scene sample: tiny box falls back to the whole box, then to the background; full-frame box → null', () => {
  const W = 96;
  const H = 54;
  const data = image(W, H, [10, 20, 30], [200, 210, 220], [40, 41, 20, 21]);
  // A one-pixel box: its inner 60 % holds no pixel centre, the whole box does.
  const one = sceneColorsFromPixels(data, W, H, { x: 40 / W, y: 20 / H, width: 1 / W, height: 1 / H });
  assert.deepEqual(one, { bg: '#0a141e', obj: '#c8d2dc' });
  // A sub-pixel box between pixel centres: object = background.
  const none = sceneColorsFromPixels(data, W, H, { x: 10.6 / W, y: 10.6 / H, width: 0.2 / W, height: 0.2 / H });
  assert.deepEqual(none, { bg: '#0a141e', obj: '#0a141e' });
  assert.equal(sceneColorsFromPixels(data, W, H, { x: 0, y: 0, width: 1, height: 1 }), null);
  assert.equal(sceneColorsFromPixels(data, 0, H, { x: 0.1, y: 0.1, width: 0.1, height: 0.1 }), null);
  assert.equal(sceneColorsFromPixels(new Uint8ClampedArray(4), W, H, { x: 0.1, y: 0.1, width: 0.1, height: 0.1 }), null);
  assert.equal(sceneColorsFromPixels(data, W, H, { x: Number.NaN, y: 0.1, width: 0.1, height: 0.1 }), null);
});

test('sample size keeps the frame aspect at 96 px wide; mirroring flips x only', () => {
  assert.deepEqual(sampleSize(1920, 1080), { width: SAMPLE_WIDTH_PX, height: 54 });
  assert.deepEqual(sampleSize(1620, 1080), { width: 96, height: 64 });
  assert.deepEqual(sampleSize(0, 0), { width: 96, height: 1 });
  const m = mirrorUnitBox({ x: 0.1, y: 0.2, width: 0.3, height: 0.4 });
  assert.ok(Math.abs(m.x - 0.6) < 1e-12 && m.y === 0.2 && m.width === 0.3 && m.height === 0.4, JSON.stringify(m));
});
