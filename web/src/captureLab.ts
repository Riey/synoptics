/**
 * Dev-only capture lab (`vite` dev server: /capture-lab.html; not a build input). Compares the capture paths'
 * outputs on the SAME camera frame — run it with a still fake camera (a one-frame y4m), so every read of the
 * video sees identical pixels — and probes whether CPU throttling reaches workers. Driven from Playwright
 * through `window.lab`.
 */
import type { Box } from '@visual-coach/visual-tools';

import { setCaptureSettingsForTest, type CaptureSettings } from './coach/captureMode';
import {
  CAPTURE_EVENT,
  captureVideoFrameAsync,
  sampleVideoAppearanceAsync,
  type CaptureEventDetail,
} from './coach/image-processor';
import { frameYGrid } from './coach/yPlaneGrid';
import { DEFAULT_APPEARANCE_CONFIG, compareAppearance } from './intent/anchorAppearance';
import { rawAppearanceCrop, sampleVideoAnchorGrid } from './intent/anchorSampler';

const video = document.getElementById('lab-video') as HTMLVideoElement;
const events: CaptureEventDetail[] = [];
window.addEventListener(CAPTURE_EVENT, (event) => events.push((event as CustomEvent<CaptureEventDetail>).detail));

function settings(mode: CaptureSettings['mode'], v2Source: CaptureSettings['v2Source'] = 'auto'): void {
  setCaptureSettingsForTest({ mode, trackScale: 'off', trackUpload: 'json', v2Source });
}

async function pixelsOf(base64: string): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const blob = await (await fetch(`data:image/jpeg;base64,${base64}`)).blob();
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  context.drawImage(bitmap, 0, 0);
  return { data: context.getImageData(0, 0, bitmap.width, bitmap.height).data, width: bitmap.width, height: bitmap.height };
}

function pixelDiff(a: Uint8ClampedArray, b: Uint8ClampedArray): { max: number; mean: number; over8: number } {
  let max = 0;
  let sum = 0;
  let over8 = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c += 1) {
      const d = Math.abs(a[i + c] - b[i + c]);
      if (d > max) max = d;
      if (d > 8) over8 += 1;
      sum += d;
      n += 1;
    }
  }
  return { max, mean: +(sum / n).toFixed(4), over8 };
}

function gridDiff(a: Float32Array | null | undefined, b: Float32Array | null | undefined) {
  if (!a || !b) return { missing: true };
  let max = 0;
  for (let i = 0; i < a.length; i += 1) max = Math.max(max, Math.abs(a[i] - b[i]));
  const score = compareAppearance(a, b, 6);
  return { max: +max.toFixed(5), scoreMean: +score.mean.toFixed(5), scorePatch: +score.patch.toFixed(5) };
}

const PATHS: Array<[string, CaptureSettings['mode'], CaptureSettings['v2Source']]> = [
  ['legacy', 'legacy', 'auto'],
  ['worker1', 'worker1', 'auto'],
  ['v2-stream', 'v2', 'auto'],
  ['v2-bitmap', 'v2', 'bitmap'],
];

const lab = {
  async start(): Promise<{ width: number; height: number }> {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    video.srcObject = stream;
    await video.play();
    while (video.videoWidth === 0) await new Promise((resolve) => setTimeout(resolve, 20));
    return { width: video.videoWidth, height: video.videoHeight };
  },

  /** Every path's JPEG, follow-reference grid and 250 ms sample grid on the same (still) frame. */
  async equivalence(mirror: boolean, box: Box) {
    const opposite = await captureVideoFrameAsync(video, { mirror: !mirror });
    const out: Record<string, { paths?: string[]; pixels: { data: Uint8ClampedArray; width: number; height: number }; reference: Float32Array | null | undefined; sample: Float32Array | null }> = {};
    for (const [name, mode, source] of PATHS) {
      settings(mode, source);
      // Twice: the first v2 stream capture attaches the stream (and may come from a bitmap meanwhile).
      await captureVideoFrameAsync(video, { mirror, appearanceBox: box });
      events.length = 0;
      const frame = await captureVideoFrameAsync(video, { mirror, appearanceBox: box });
      const sample = await sampleVideoAppearanceAsync(video, box, mirror);
      const paths = events.map((event) => `${event.kind}:${event.path}`);
      out[name] = { paths, pixels: await pixelsOf(frame.rawBase64), reference: frame.appearanceGrid, sample };
    }
    settings('legacy');
    const mainSample = sampleVideoAnchorGrid(video, box, mirror);
    const base = out.legacy;
    const report: Record<string, unknown> = {
      size: `${base.pixels.width}x${base.pixels.height}`,
      // Sanity: the other orientation must differ, or the comparisons below prove nothing.
      oppositeMirrorVsLegacy: pixelDiff((await pixelsOf(opposite.rawBase64)).data, base.pixels.data),
    };
    for (const [name] of PATHS) {
      const row = out[name];
      report[name] = {
        paths: row.paths,
        jpegVsLegacy: pixelDiff(row.pixels.data, base.pixels.data),
        referenceVsLegacyMain: gridDiff(row.reference, mainSample),
        sampleVsLegacyMain: gridDiff(row.sample, mainSample),
        sampleVsOwnReference: gridDiff(row.sample, row.reference),
      };
    }
    return report;
  },

  /**
   * The v2 stream sample's method (Y plane, `frameYGrid`) against the canvas ladder (`sampleVideoAnchorGrid`)
   * on the SAME decoded frames of a clip: per-frame difference, and whether pairwise scores (what
   * `target_changed` thresholds) agree.
   */
  async methodCompare(clipUrl: string, box: Box, mirror: boolean, times: number[], gaps: number[]) {
    const file = document.createElement('video');
    file.muted = true;
    file.src = clipUrl;
    await new Promise((resolve, reject) => {
      file.onloadeddata = resolve;
      file.onerror = reject;
    });
    const ladder: Float32Array[] = [];
    const yGrid: Float32Array[] = [];
    let format: string | null = null;
    for (const t of times) {
      file.currentTime = t;
      await new Promise((resolve) => (file.onseeked = resolve));
      const l = sampleVideoAnchorGrid(file, box, mirror);
      const frame = new VideoFrame(file);
      format = frame.format;
      const crop = rawAppearanceCrop(box, { width: frame.displayWidth, height: frame.displayHeight }, DEFAULT_APPEARANCE_CONFIG, mirror);
      const y = crop ? await frameYGrid(frame, crop, DEFAULT_APPEARANCE_CONFIG.grid, mirror) : null;
      frame.close();
      if (!l || !y) throw new Error(`no grid at ${t}: ladder ${Boolean(l)} y ${Boolean(y)} format ${format}`);
      ladder.push(l);
      yGrid.push(y);
    }
    const same = ladder.map((l, i) => compareAppearance(l, yGrid[i], 6).mean);
    const pairs: Array<{ gap: number; ladder: number; y: number }> = [];
    for (const gap of gaps) {
      for (let i = 0; i + gap < times.length; i += 1) {
        pairs.push({
          gap,
          ladder: compareAppearance(ladder[i], ladder[i + gap], 6).mean,
          y: compareAppearance(yGrid[i], yGrid[i + gap], 6).mean,
        });
      }
    }
    const threshold = DEFAULT_APPEARANCE_CONFIG.threshold;
    const absDiff = pairs.map((p) => Math.abs(p.ladder - p.y)).sort((a, b) => a - b);
    const ratio = pairs.filter((p) => p.ladder > 0.01).map((p) => p.y / p.ladder).sort((a, b) => a - b);
    return {
      format,
      frames: times.length,
      sameFrameScore: { max: Math.max(...same), mean: same.reduce((a, b) => a + b, 0) / same.length },
      pairs: pairs.length,
      pairAbsDiff: { p50: absDiff[Math.floor(absDiff.length / 2)], p95: absDiff[Math.floor(absDiff.length * 0.95)], max: absDiff[absDiff.length - 1] },
      yOverLadder: { p5: ratio[Math.floor(ratio.length * 0.05)], p50: ratio[Math.floor(ratio.length / 2)], p95: ratio[Math.floor(ratio.length * 0.95)] },
      decisionsDisagree: pairs.filter((p) => p.ladder > threshold !== p.y > threshold).length,
      ladderOverThreshold: pairs.filter((p) => p.ladder > threshold).length,
    };
  },

  /**
   * Does cancelling a worker's read of a MediaStreamTrackProcessor readable end the page's camera track?
   * (v2 cancels a lane's previous reader when it re-attaches to a new track.)
   */
  async cancelProbe(): Promise<{ before: string; afterCancel: string; videoAdvances: boolean }> {
    const track = (video.srcObject as MediaStream).getVideoTracks()[0];
    const Processor = (globalThis as unknown as { MediaStreamTrackProcessor: new (init: { track: MediaStreamTrack }) => { readable: ReadableStream } }).MediaStreamTrackProcessor;
    const { readable } = new Processor({ track });
    const source = `onmessage = async (e) => { const r = e.data.getReader(); const f = await r.read(); f.value.close(); await r.cancel(); postMessage('cancelled'); };`;
    const worker = new Worker(URL.createObjectURL(new Blob([source], { type: 'text/javascript' })));
    const before = track.readyState;
    await new Promise<void>((resolve) => {
      worker.onmessage = () => resolve();
      worker.postMessage(readable, [readable as unknown as Transferable]);
    });
    worker.terminate();
    const t0 = video.currentTime;
    await new Promise((resolve) => setTimeout(resolve, 500));
    return { before, afterCancel: track.readyState, videoAdvances: video.currentTime > t0 };
  },

  /** A fixed busy loop on the main thread and in a worker (to see what CPU throttling reaches). */
  async spin(iterations = 3e7): Promise<{ mainMs: number; workerMs: number }> {
    const loop = `(() => { let x = 0; const t = performance.now(); for (let i = 0; i < ${iterations}; i += 1) x = (x + i * 7) % 1000003; return { ms: performance.now() - t, x }; })()`;
    const mainMs = (eval(loop) as { ms: number }).ms;
    const source = `onmessage = () => { const r = ${loop}; postMessage(r.ms); };`;
    const worker = new Worker(URL.createObjectURL(new Blob([source], { type: 'text/javascript' })));
    const workerMs = await new Promise<number>((resolve) => {
      worker.onmessage = (event) => resolve(event.data as number);
      worker.postMessage(null);
    });
    worker.terminate();
    return { mainMs: Math.round(mainMs), workerMs: Math.round(workerMs) };
  },
};

(window as unknown as { lab: typeof lab }).lab = lab;
