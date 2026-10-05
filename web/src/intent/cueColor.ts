/**
 * The overlay's cue colour: one accent per motion run, picked for contrast against the scene around and on the
 * tracked object, plus the halo, glow and callout colours that go with it (designer look, 2026-10-03).
 *
 * - `pickContrastAccent(bg, obj, preferred)`: the first candidate (the preferred accent first) whose contrast
 *   with the background is ≥ 3:1, whose hue is ≥ 45° away from a colourful background, and that stands off the
 *   object (contrast ≥ 1.5:1, or a hue ≥ 60° away from a colourful object). With no candidate passing, the best
 *   partial score wins. The overlay calls it ONCE per motion run and keeps the result until the run changes,
 *   so the cue never flickers between frames.
 * - `accentTone(accent)`: a light accent gets a dark halo and a dark callout; a dark accent a light halo and a
 *   light callout.
 * - `sceneColorsFromPixels`: the mean colour of a band around the box (background) and of the box's inner 60 %
 *   (object) in a small RGBA sample of the camera frame. The overlay draws the frame into a 96 px wide canvas
 *   once per run and passes the pixels here.
 *
 * Pure: nothing here imports a value from another module, so `node --test` runs it directly.
 */

/** The default accent (sky). Used as is when the scene cannot be sampled. */
export const PREFERRED_ACCENT = '#38bdf8';
/** Tried after the preferred accent, in order. */
export const ACCENT_CANDIDATES: readonly string[] = Object.freeze([
  '#38bdf8',
  '#facc15',
  '#f472b6',
  '#a3e635',
  '#fb923c',
  '#1d4ed8',
  '#6d28d9',
  '#be123c',
  '#ffffff',
  '#172554',
]);
/** WCAG contrast an accent needs against the background. */
export const BG_MIN_CONTRAST = 3;
/** Contrast an accent needs against the object (unless its hue differs enough). */
export const OBJ_MIN_CONTRAST = 1.5;
/** Chroma (max − min of 0..1 RGB) from which a colour counts as colourful (its hue matters). */
export const COLORFUL_CHROMA = 0.15;
/** Hue distance (degrees) a colourful accent keeps from a colourful background / object. */
export const BG_MIN_HUE_GAP = 45;
export const OBJ_MIN_HUE_GAP = 60;
/** Relative luminance from which an accent is "light" (dark halo and callout). */
export const LIGHT_ACCENT_LUMINANCE = 0.3;
/** Width of the frame sample (px); the height follows the frame's aspect ratio. */
export const SAMPLE_WIDTH_PX = 96;
/** Background band around the box: this fraction of the box's longer side on every side. */
export const SAMPLE_BAND_FRACTION = 0.5;
/** The object sample is the box's inner part, this fraction cut from every side (inner 60 %). */
export const SAMPLE_CORE_INSET_FRACTION = 0.2;

export type Rgb = [number, number, number];

export interface CueTone {
  accent: string;
  halo: string;
  glow: string;
  bubbleBg: string;
  bubbleFg: string;
  /** Text colour on an accent-filled surface. */
  accentFg: string;
}

export interface SceneColors {
  bg: string;
  obj: string;
}

/** A normalized (0..1 of the frame) box. */
export interface UnitBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** `#rgb` / `#rrggbb` → 0..1 channels; anything unparsable → black. */
export function hexRgb(hex: string): Rgb {
  const raw = String(hex).trim().replace(/^#/, '');
  if (!/^(?:[0-9a-f]{3}|[0-9a-f]{6})$/i.test(raw)) return [0, 0, 0];
  const full = raw.length === 3 ? raw.split('').map((c) => c + c).join('') : raw;
  const ch = (i: number) => parseInt(full.slice(i, i + 2), 16) / 255;
  return [ch(0), ch(2), ch(4)];
}

/** 0..255 channels → `#rrggbb` (clamped, rounded). */
export function rgbHex(r: number, g: number, b: number): string {
  const part = (v: number) => Math.round(Math.min(255, Math.max(0, Number.isFinite(v) ? v : 0))).toString(16).padStart(2, '0');
  return `#${part(r)}${part(g)}${part(b)}`;
}

/** WCAG relative luminance of 0..1 sRGB channels. */
export function relativeLuminance(rgb: Rgb): number {
  const lin = rgb.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}

/** WCAG contrast ratio (1..21). */
export function contrastRatio(a: Rgb, b: Rgb): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** Hue (0..360) and chroma (0..1) of 0..1 RGB. */
export function hueChroma(rgb: Rgb): { h: number; c: number } {
  const [r, g, b] = rgb;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const c = max - min;
  let h = 0;
  if (c > 0) h = max === r ? ((g - b) / c) % 6 : max === g ? (b - r) / c + 2 : (r - g) / c + 4;
  return { h: (h * 60 + 360) % 360, c };
}

/** Shortest distance between two hues, 0..180. */
export function hueGap(a: number, b: number): number {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

/** The accent for a scene whose background / object mean colours are `bgHex` / `objHex` (see the module doc). */
export function pickContrastAccent(bgHex: string, objHex: string, preferred: string = PREFERRED_ACCENT): string {
  const bg = hexRgb(bgHex);
  const obj = hexRgb(objHex);
  const hb = hueChroma(bg);
  const ho = hueChroma(obj);
  const list = [preferred, ...ACCENT_CANDIDATES.filter((c) => c.toLowerCase() !== preferred.toLowerCase())];
  let best = list[0];
  let bestScore = -1;
  for (const candidate of list) {
    const rgb = hexRgb(candidate);
    const hc = hueChroma(rgb);
    const crBg = contrastRatio(rgb, bg);
    const crObj = contrastRatio(rgb, obj);
    const colorful = hc.c >= COLORFUL_CHROMA;
    const bgHueOk = !colorful || hb.c < COLORFUL_CHROMA || hueGap(hc.h, hb.h) >= BG_MIN_HUE_GAP;
    const objOk = crObj >= OBJ_MIN_CONTRAST || (colorful && ho.c >= COLORFUL_CHROMA && hueGap(hc.h, ho.h) >= OBJ_MIN_HUE_GAP);
    if (crBg >= BG_MIN_CONTRAST && bgHueOk && objOk) return candidate;
    const score = Math.min(crBg / BG_MIN_CONTRAST, 1) + 0.5 * Math.min(crObj / OBJ_MIN_CONTRAST, 1) + (bgHueOk ? 0.25 : 0);
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return best;
}

/** The halo, glow and callout colours that go with `accent`. */
export function accentTone(accent: string): CueTone {
  const rgb = hexRgb(accent);
  const light = relativeLuminance(rgb) >= LIGHT_ACCENT_LUMINANCE;
  const glow = `rgba(${rgb.map((v) => Math.round(v * 255)).join(', ')}, ${light ? 0.5 : 0.35})`;
  return light
    ? { accent, halo: 'rgba(2, 6, 23, 0.95)', glow, bubbleBg: 'rgba(15, 23, 42, 0.94)', bubbleFg: '#f8fafc', accentFg: '#0b1020' }
    : { accent, halo: 'rgba(248, 250, 252, 0.95)', glow, bubbleBg: 'rgba(248, 250, 252, 0.96)', bubbleFg: '#0f172a', accentFg: '#ffffff' };
}

/** The sample canvas size for a frame of `frameWidth`×`frameHeight` (96 px wide, aspect kept, ≥1 px). */
export function sampleSize(frameWidth: number, frameHeight: number): { width: number; height: number } {
  const ok = frameWidth > 0 && frameHeight > 0 && Number.isFinite(frameWidth) && Number.isFinite(frameHeight);
  return { width: SAMPLE_WIDTH_PX, height: ok ? Math.max(1, Math.round((SAMPLE_WIDTH_PX * frameHeight) / frameWidth)) : 1 };
}

/**
 * Mean background and object colours in an RGBA sample (`data`, `width`×`height`) of the frame `box` is
 * normalized to. Background = a band of half the box's longer side around it (outside the box); object = the
 * box's inner 60 %, or the whole box when that holds no sample pixel. Null when the band holds no pixel (the
 * box covers the frame) or the input is malformed.
 */
export function sceneColorsFromPixels(
  data: ArrayLike<number>,
  width: number,
  height: number,
  box: UnitBox
): SceneColors | null {
  if (!(width > 0 && height > 0) || data.length < width * height * 4) return null;
  if (![box.x, box.y, box.width, box.height].every(Number.isFinite)) return null;
  const left = box.x * width;
  const top = box.y * height;
  const right = (box.x + Math.max(0, box.width)) * width;
  const bottom = (box.y + Math.max(0, box.height)) * height;
  const bw = right - left;
  const bh = bottom - top;
  const pad = SAMPLE_BAND_FRACTION * Math.max(bw, bh);
  const inset = SAMPLE_CORE_INSET_FRACTION;
  const bg = [0, 0, 0, 0];
  const core = [0, 0, 0, 0];
  const whole = [0, 0, 0, 0];
  for (let y = 0; y < height; y += 1) {
    // Pixel centres, so a box covering half a pixel does not claim it.
    const fy = y + 0.5;
    for (let x = 0; x < width; x += 1) {
      const fx = x + 0.5;
      const inBox = fx >= left && fx < right && fy >= top && fy < bottom;
      const inCore = fx >= left + bw * inset && fx < right - bw * inset && fy >= top + bh * inset && fy < bottom - bh * inset;
      const inBand = !inBox && fx >= left - pad && fx < right + pad && fy >= top - pad && fy < bottom + pad;
      const targets = inCore ? [core, whole] : inBox ? [whole] : inBand ? [bg] : [];
      if (targets.length === 0) continue;
      const i = (y * width + x) * 4;
      for (const acc of targets) {
        acc[0] += data[i];
        acc[1] += data[i + 1];
        acc[2] += data[i + 2];
        acc[3] += 1;
      }
    }
  }
  if (bg[3] === 0) return null;
  const mean = (acc: number[]) => rgbHex(acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3]);
  const bgHex = mean(bg);
  const objHex = core[3] > 0 ? mean(core) : whole[3] > 0 ? mean(whole) : bgHex;
  return { bg: bgHex, obj: objHex };
}

/** Horizontally mirrored box (the stage video can be shown mirrored while the frame is read unmirrored). */
export function mirrorUnitBox(box: UnitBox): UnitBox {
  return { x: 1 - box.x - box.width, y: box.y, width: box.width, height: box.height };
}
