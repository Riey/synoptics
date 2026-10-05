/**
 * Dev-only preview of the action motion layer (`/motion-preview.html`; not linked from the app).
 *
 * Mounts the real `AnchoredGuidanceOverlay` over a plain dark stage with a fake live track store whose box
 * drifts slowly (sinusoidal, ±3 % of the frame, republished every animation frame so it stays fresh) and a
 * fake guide overlay store bound to it. Controls pick the action/direction, replay (a new motionKey), force
 * reduced motion, drop the focus command (the ring is then implied by the action), set the callout text, pick a box
 * preset (normal, large, a small 60×20 box, a box touching the bottom or the right edge) and a scene (the
 * designer's five background/object colours). With no camera, the scene's colours stand in for the frame sample
 * (`sceneColors`), so the overlay's once-per-run auto-contrast accent is exercised; the chosen accent is shown
 * next to the status line. The graph core's four schematic actions (align, connect, disconnect, bend) are in the
 * action list too, and a caption under the stage keeps the "동작 예시 · 실제 핀 위치 아님" scope in view.
 * No tracker, LLM or backend is involved.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';
import './styles.css';

import { AnchoredGuidanceOverlay } from './components/AnchoredGuidanceOverlay';
import { motionKey } from './intent/actionMotion';
import { actionShortLabel, type ActionDirection, type ActionType, type BoundCommand } from './intent/anchorProject';
import { createGuideOverlayStore } from './intent/guideOverlayStore';
import { createLiveTrackStore } from './tracking/liveTrackStore';
import { PREFERRED_ACCENT, type SceneColors } from './intent/cueColor';

const COMBOS: Array<[ActionType, ActionDirection]> = [
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
  // place / fit aim at the provisional destination beside the box (no destination anchor yet).
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
  // The graph core's schematic circuit actions (2026-10-05): no direction, a guide/halves/lead glyph.
  ['align', 'none'],
  ['connect', 'none'],
  ['disconnect', 'none'],
  ['bend', 'none'],
];

/** The designer's preview scenes: frame background and object colour (stand-ins for the camera sample). */
const SCENES = {
  dark: { label: '어두운 방', bg: '#0f172a', obj: '#334155' },
  white: { label: '흰 책상', bg: '#eef2f6', obj: '#c7d0db' },
  wood: { label: '나무 테이블', bg: '#9a6a3a', obj: '#2b2523' },
  sky: { label: '하늘색 천', bg: '#7dd3fc', obj: '#e0f2fe' },
  green: { label: '초록 매트', bg: '#166534', obj: '#fde047' },
} as const;
type SceneName = keyof typeof SCENES;

/** Stage size the presets are written for (the stage is 640×400 CSS px unless the window is narrower). */
const STAGE_W = 640;
const STAGE_H = 400;

/**
 * Box presets in frame fractions. `cx`/`cy` place the box centre; `driftX`/`driftY` scale the ±3 % drift per axis
 * (0 keeps an edge-touching box on its edge).
 */
const PRESETS = {
  normal: { label: '보통', width: 0.2, height: 0.22, cx: 0.5, cy: 0.5, driftX: 1, driftY: 1 },
  large: { label: '큰 상자', width: 0.42, height: 0.5, cx: 0.5, cy: 0.5, driftX: 1, driftY: 1 },
  small: { label: '작은 상자 60×20', width: 60 / STAGE_W, height: 20 / STAGE_H, cx: 0.5, cy: 0.5, driftX: 1, driftY: 1 },
  bottom: { label: '아래 가장자리 밀착', width: 0.2, height: 0.18, cx: 0.5, cy: 1 - 0.09, driftX: 1, driftY: 0 },
  right: { label: '오른쪽 가장자리 밀착', width: 0.14, height: 0.2, cx: 1 - 0.07, cy: 0.5, driftX: 0, driftY: 1 },
} as const;
type PresetName = keyof typeof PRESETS;

const RUN = { runId: 'preview-run', trackId: 'preview-track', generation: 1 };
/** Drift amplitude (fraction of the frame) and period. */
const DRIFT = 0.03;
const DRIFT_PERIOD_MS = 6000;

function MotionPreview() {
  const liveTrack = useMemo(() => createLiveTrackStore(), []);
  const guide = useMemo(() => createGuideOverlayStore(), []);
  const [combo, setCombo] = useState(0);
  const [replay, setReplay] = useState(0);
  const [reduced, setReduced] = useState(false);
  const [withFocus, setWithFocus] = useState(true);
  const [labelText, setLabelText] = useState('');
  const [preset, setPreset] = useState<PresetName>('normal');
  const [scene, setScene] = useState<SceneName>('dark');
  const [status, setStatus] = useState('');
  const [accent, setAccent] = useState(PREFERRED_ACCENT);
  const objectRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const presetRef = useRef(preset);
  presetRef.current = preset;

  // The fake tracker: a fresh `tracking` snapshot every frame, the box drifting around the stage centre.
  useEffect(() => {
    let raf = 0;
    let version = 0;
    const start = performance.now();
    const step = () => {
      const t = (performance.now() - start) / DRIFT_PERIOD_MS;
      const { width, height, cx, cy, driftX, driftY } = PRESETS[presetRef.current];
      const box = {
        x: cx - width / 2 + driftX * DRIFT * Math.sin(2 * Math.PI * t),
        y: cy - height / 2 + driftY * DRIFT * Math.sin(2 * Math.PI * t * 0.7 + 1),
        width,
        height,
      };
      version += 1;
      liveTrack.publish({ ...RUN, state: 'tracking', box, capturedAt: Date.now(), version, target: null });
      const object = objectRef.current;
      if (object) {
        object.style.left = `${box.x * 100}%`;
        object.style.top = `${box.y * 100}%`;
        object.style.width = `${box.width * 100}%`;
        object.style.height = `${box.height * 100}%`;
      }
      const root = stageRef.current?.querySelector<HTMLElement>('[data-testid="anchored-guidance"]');
      if (root && version % 10 === 0) {
        setStatus(
          `motion=${root.dataset.motion ?? '-'} phase=${root.dataset.motionPhase ?? '-'} focus=${root.dataset.focus ?? '-'} reason=${root.dataset.guideReason ?? '-'}`
        );
        setAccent(root.dataset.cueAccent ?? PREFERRED_ACCENT);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [liveTrack]);

  // The fake guide: bound to the fake run, the chosen action, a focus ring and an optional label.
  useEffect(() => {
    const [action, direction] = COMBOS[combo];
    // Without a focus command the overlay implies the default ring for the action (resolveFocus).
    const commands: BoundCommand[] = [
      ...(withFocus ? [{ kind: 'focus' as const, anchor: 'a1', pad: 0.15 }] : []),
      { kind: 'action', anchor: 'a1', action, direction },
    ];
    if (labelText.trim()) commands.push({ kind: 'label', anchor: 'a1', text: labelText.trim() });
    guide.set({
      binding: { anchorId: 'a1', ...RUN },
      commands,
      warn: false,
      motionKey: motionKey({
        runId: replay,
        planRevision: 1,
        // The scene is part of the step id: a new scene is a new run, so the accent is chosen again.
        stepId: `${action}:${direction}:${scene}`,
        anchorId: 'a1',
        trackRunId: RUN.runId,
        trackId: RUN.trackId,
        generation: RUN.generation,
      }),
    });
  }, [guide, combo, replay, labelText, withFocus, scene]);

  // The scene sample the overlay asks for once per run (the real app samples the stage video instead).
  const sceneColors = useMemo(() => (): SceneColors => ({ bg: SCENES[scene].bg, obj: SCENES[scene].obj }), [scene]);

  return (
    <main style={{ padding: 16, fontFamily: 'system-ui, sans-serif', color: '#e2e8f0', background: '#020617', minHeight: '100vh', boxSizing: 'border-box' }}>
      <h1 style={{ fontSize: 18, margin: '0 0 12px' }}>동작 모션 미리보기 (개발용)</h1>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <label>
          동작{' '}
          <select value={combo} onChange={(e) => setCombo(Number(e.target.value))} data-testid="preview-action">
            {COMBOS.map(([action, direction], i) => (
              <option key={`${action}:${direction}`} value={i}>
                {action}:{direction} · {actionShortLabel(action, direction)}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={() => setReplay((n) => n + 1)} data-testid="preview-replay">
          다시 재생
        </button>
        <label>
          <input type="checkbox" checked={reduced} onChange={(e) => setReduced(e.target.checked)} data-testid="preview-reduced" />{' '}
          동작 줄이기
        </label>
        <label>
          <input type="checkbox" checked={withFocus} onChange={(e) => setWithFocus(e.target.checked)} data-testid="preview-focus" />{' '}
          focus 명령 포함
        </label>
        <label>
          상자{' '}
          <select value={preset} onChange={(e) => setPreset(e.target.value as PresetName)} data-testid="preview-preset">
            {(Object.keys(PRESETS) as PresetName[]).map((name) => (
              <option key={name} value={name}>
                {PRESETS[name].label}
              </option>
            ))}
          </select>
        </label>
        <label>
          배경{' '}
          <select value={scene} onChange={(e) => setScene(e.target.value as SceneName)} data-testid="preview-scene">
            {(Object.keys(SCENES) as SceneName[]).map((name) => (
              <option key={name} value={name}>
                {SCENES[name].label}
              </option>
            ))}
          </select>
        </label>
        <label>
          라벨{' '}
          <input
            type="text"
            value={labelText}
            maxLength={40}
            placeholder="비우면 동작 이름"
            onChange={(e) => setLabelText(e.target.value)}
            data-testid="preview-label"
          />
        </label>
      </div>
      <div
        ref={stageRef}
        style={{
          position: 'relative',
          width: `min(${STAGE_W}px, 100%)`,
          aspectRatio: `${STAGE_W} / ${STAGE_H}`,
          background: SCENES[scene].bg,
          overflow: 'hidden',
          borderRadius: 8,
        }}
      >
        <div
          ref={objectRef}
          aria-hidden="true"
          style={{ position: 'absolute', background: SCENES[scene].obj, borderRadius: 6, boxShadow: 'inset 0 0 0 1px rgba(148, 163, 184, 0.6)' }}
        />
        <AnchoredGuidanceOverlay liveTrack={liveTrack} guide={guide} reducedMotion={reduced} sceneColors={sceneColors} />
      </div>
      <p style={{ margin: '8px 0 0', fontSize: 12, color: '#94a3b8' }} data-testid="preview-scope-note">
        동작 예시 · 실제 핀 위치 아님 — 정렬·연결·분리·구부리기는 상자 기준 도식이고, 특정 핀·극성·전압을 가리키지 않습니다.
      </p>
      <p style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12, color: '#94a3b8' }} data-testid="preview-status">
        {status}
      </p>
      <p style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#cbd5e1' }} data-testid="preview-accent">
        <span
          aria-hidden="true"
          style={{ width: 12, height: 12, borderRadius: 999, background: accent, boxShadow: '0 0 0 1px #64748b' }}
        />
        강조색 자동 선택 {accent} (동작이 시작될 때 장면에서 한 번 정하고 그 동작 동안 유지)
      </p>
    </main>
  );
}

const rootEl = document.getElementById('root');
if (rootEl) {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      <MotionPreview />
    </React.StrictMode>
  );
}
