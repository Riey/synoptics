/**
 * Target-tracking card (contract v4 + the startup-selection flow).
 *
 * One explicit Start runs a single paid LLM analysis that chooses the object to track; every subsequent
 * frame is handled locally by the neural tracker (zero per-frame provider calls). Manual box drawing is a
 * pure correction that needs no provider. The card is independent of the paid guidance path and never
 * touches the guidance overlay.
 *
 * The hook (`useObjectTrack`) is owned by `App`, which also feeds its live store to the camera stage: the
 * live box is drawn over the playing video there. This card keeps the controls, errors and selection
 * outcome; the retained still pair (one captured frame beside the one response computed for it, with this
 * client's own capture age) is diagnostics now and sits behind a `<details>`. Manual drawing uses its own
 * freshly captured still rather than a tracker response, so a box can be drawn even before any run exists.
 */
import { useEffect, useState } from 'react';

import { isBoxDrawable } from '../tracking/trackFence';
import {
  MANUAL_SELECTION_TARGET,
  TRACK_FRESHNESS_LIMIT_MS,
  type ObjectTrack,
  type TrackPair,
} from '../tracking/useObjectTrack';
import { PLAN_MODEL_OPTION, type PlanModel, type PlanModelAvailability } from '../planMode/modelChoice';
import { CoreModeSelector } from '../planMode/CoreModeSelector';
import type { CoreMode } from '../planMode/coreChoice';
import { PlanModelSelector } from '../planMode/PlanModelSelector';
import { TargetSelectOverlay } from './TargetSelectOverlay';
import { TrackOverlay } from './TrackOverlay';

/** The user-facing label for a run started from a hand-drawn box. */
const MANUAL_TARGET_LABEL = '직접 지정한 물체';

const TRACK_STATE_LABEL: Record<string, string> = {
  acquiring: '대상 찾는 중',
  tracking: '추적 중',
  occluded: '가림',
  lost: '놓침',
  unavailable: '추적기 사용 불가',
};

const TRACK_STATE_BADGE: Record<string, string> = {
  acquiring: 'badge-manual',
  tracking: 'badge-observed',
  occluded: 'badge-warning',
  lost: 'badge-warning',
  unavailable: 'badge-warning',
};

/** The guide lane's controls, owned by `App` (the intent loop lives there). */
export interface TrackPanelGuide {
  /** A guide lane exists: health.plan_ready (any plan model) or the legacy guide_ready. */
  ready: boolean;
  followerReady: boolean;
  /** Who follows the plan (health.follow_provider), shown before starting. */
  followProviderText: string;
  /** The plan model chosen for the next run; frozen while a run is planning or running. */
  planModel: PlanModel;
  /** health.plan_models: whether each plan model's key is configured (null while unknown). */
  planModels: PlanModelAvailability;
  /** Whether the chosen plan model can plan; false holds Start rather than silently using the other model. */
  modelReady: boolean;
  onSelectPlanModel: (model: PlanModel) => void;
  /** The core chosen for the next run; independent of the plan model and frozen through a run. */
  coreMode: CoreMode;
  onSelectCoreMode: (mode: CoreMode) => void;
  planning: boolean;
  running: boolean;
  startBlockedReason?: string | null;
  onStart: () => void;
  onStop: () => void;
  onManual: () => void;
  /** "계획 다시 짜기": ask DeepSeek to rewrite the remaining steps from a fresh frame. */
  onReplan: () => void;
  /** Why a replan is unavailable right now (per-run maximum or the replan gap); null when it is available. */
  replanBlockedReason: string | null;
  /** Cost text of the guide lane (plan / follow / confirm), computed by the host. */
  costText: string;
  visualsEnabled: boolean;
  onToggleVisuals: (enabled: boolean) => void;
}

export interface TrackPanelProps {
  /** The tracking hook's result, owned by `App`. */
  track: ObjectTrack;
  consent: boolean;
  cameraLive: boolean;
  trackerReady: boolean;
  /** Whether the paid provider is configured — the startup analysis needs it, manual drawing does not. */
  analysisReady: boolean;
  /** Page-lifetime accounting text for the startup analysis, computed and owned by the host. */
  analysisCostText: string;
  onGrantConsent: () => void;
  /** Present in camera mode: “가이드 시작” replaces the separate startup analysis for the guide lane. */
  guide?: TrackPanelGuide;
}

/**
 * The `<img>` source of a still: its `data:` URL (a JSON-uploaded frame), or an object URL for the raw JPEG
 * it was uploaded as, revoked when the still is replaced or the panel unmounts.
 */
function useStillSrc(pair: TrackPair | null | undefined): string | undefined {
  const jpeg = pair && !pair.imageDataUrl ? pair.jpeg : null;
  const [objectUrl, setObjectUrl] = useState<string | undefined>(undefined);
  useEffect(() => {
    if (!jpeg) {
      setObjectUrl(undefined);
      return undefined;
    }
    const url = URL.createObjectURL(jpeg);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [jpeg]);
  return pair?.imageDataUrl ?? objectUrl;
}

export function TrackPanel(props: TrackPanelProps) {
  const { track } = props;
  const shownSrc = useStillSrc(track.shown?.pair);

  const analyzing = track.phase === 'analyzing';
  const starting = track.phase === 'starting';
  const busy = analyzing || starting || track.phase === 'running';
  // One authoritative source for "a manual drawing is open": the capture itself, so a cleared capture
  // (scene change, consent, cancel, a failed capture) can never leave the button stuck disabled.
  const drawing = track.manualCapture !== null;
  const state = track.shown?.response.state ?? null;
  const transition = track.shown?.response.transition ?? null;
  const rawBox = track.shown?.response.box;
  const box = isBoxDrawable(rawBox, state ?? '') ? rawBox : null;
  const stale = track.ageMs > TRACK_FRESHNESS_LIMIT_MS;
  const drawBox = stale ? null : box;
  const canStart = props.consent && props.cameraLive && props.trackerReady && props.analysisReady && !busy && !drawing;
  const guide = props.guide;
  const guideLane = Boolean(guide?.ready);
  const canGuide =
    guideLane && guide?.followerReady && guide?.modelReady && props.consent && props.cameraLive && props.trackerReady && !drawing && !(guide?.planning ?? false) && !guide?.startBlockedReason;
  const guideActive = Boolean(guide && (guide.planning || guide.running));
  const canDraw = props.consent && props.cameraLive && props.trackerReady && !drawing && !analyzing && !starting;

  const badgeLabel = drawing
    ? '대상 지정 중'
    : analyzing
      ? '장면 분석 중'
      : starting
        ? '대상 찾는 중'
        : state
          ? TRACK_STATE_LABEL[state] ?? state
          : '꺼짐';
  const badgeClass = drawing || analyzing || starting ? 'badge-manual' : TRACK_STATE_BADGE[state ?? ''] ?? 'badge-manual';
  const badgeState = drawing ? 'selecting' : analyzing ? 'analyzing' : starting ? 'starting' : state ?? 'idle';

  const managedLabel =
    track.target === MANUAL_SELECTION_TARGET
      ? MANUAL_TARGET_LABEL
      : track.target;

  const stateNote =
    state === null
      ? null
      : state === 'acquiring'
        ? transition === 'ambiguous'
          ? '대상이 여러 개로 보입니다. “대상 직접 지정”으로 대상을 그려주세요.'
          : '대상이 장면에서 확정되기를 기다리는 중입니다. 잘못 잡히면 “대상 직접 지정”으로 그려주세요.'
        : state === 'occluded'
          ? '대상이 가려졌습니다. 화면 표시를 멈춥니다.'
          : state === 'lost'
            ? '놓침 — “대상 직접 지정”으로 다시 그리거나 “시작”으로 새로 시작하세요.'
            : state === 'unavailable'
              ? '추적기를 사용할 수 없습니다. 추적기가 준비되면 페이지를 새로고침해 주세요.'
              : null;

  const selectionNote =
    track.selection === null
      ? null
      : track.selection.status === 'no_target'
        ? `추적할 만한 물체를 찾지 못했습니다${track.selection.rationale ? ` (${track.selection.rationale})` : ''}. “대상 직접 지정”으로 그려주세요.`
        : `대상이 확실하지 않습니다${track.selection.rationale ? ` (${track.selection.rationale})` : ''}. “대상 직접 지정”으로 그려주세요.`;

  return (
    <div className="card track-card" data-testid="track-card">
      <div className="card-header-flex">
        <h2 className="card-title">{guideLane ? '그림 안내' : '대상 추적'}</h2>
        <span
          className={`badge ${badgeClass}`}
          data-testid="track-state-badge"
          data-track-state={badgeState}
          role="status"
          aria-live="polite"
        >
          {badgeLabel}
        </span>
      </div>

      {guideLane && guide && (
        <PlanModelSelector
          value={guide.planModel}
          availability={guide.planModels}
          disabled={guideActive}
          onChange={guide.onSelectPlanModel}
        />
      )}

      {guideLane && guide && (
        <CoreModeSelector value={guide.coreMode} disabled={guideActive} onChange={guide.onSelectCoreMode} />
      )}

      <div className="button-group track-controls">
        {guideLane && guide ? (
          <button
            type="button"
            className="btn btn-primary"
            data-testid="guide-start"
            disabled={!canGuide}
            onClick={() => guide.onStart()}
          >
            {guide.running ? '가이드 다시 시작' : '가이드 시작'}
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            data-testid="track-start"
            disabled={!canStart}
            onClick={() => void track.select()}
          >
            시작
          </button>
        )}
        <button
          type="button"
          className="btn btn-secondary"
          data-testid="track-stop"
          disabled={!busy && !guideActive}
          onClick={() => {
            guide?.onStop();
            void track.stop();
          }}
        >
          정지
        </button>
        {guideLane && guide && (
          <button
            type="button"
            className="btn btn-secondary"
            data-testid="guide-manual"
            disabled={!guide.running}
            onClick={() => guide.onManual()}
          >
            지금 확인
          </button>
        )}
        {guideLane && guide && (
          <button
            type="button"
            className="btn btn-secondary"
            data-testid="guide-replan"
            disabled={!guide.running || guide.replanBlockedReason !== null}
            aria-describedby={guide.running && guide.replanBlockedReason ? 'guide-replan-reason' : undefined}
            onClick={() => guide.onReplan()}
          >
            계획 다시 짜기
          </button>
        )}
        <button
          type="button"
          className="btn btn-secondary"
          data-testid="track-reselect"
          disabled={!canDraw}
          onClick={() => track.openManualSelect()}
        >
          대상 직접 지정
        </button>
      </div>
      {guide?.startBlockedReason && <p className="form-feedback" role="status">{guide.startBlockedReason}</p>}
      <p className="card-description">
        {guideLane ? '대상 옆의 동작 기호를 따라 하세요. 자세한 문장은 필요할 때 펼쳐볼 수 있습니다.' : '시작하면 대상을 한 번 분석하고, 이후 위치는 GPU 추적기가 따라갑니다.'}
      </p>
      {guideLane && guide?.running && guide.replanBlockedReason && (
        <p className="card-description text-secondary" id="guide-replan-reason" data-testid="guide-replan-reason">
          {guide.replanBlockedReason}
        </p>
      )}
      {guideLane && !guide?.followerReady && (
        <p className="card-description" role="status">단계 확인 모델이 준비되지 않았습니다. 모델 연결 후 새로고침해 주세요. 유료 모델로 자동 전환하지 않습니다.</p>
      )}
      <details className="disclosure">
        <summary className="disclosure-summary">표시·모델·전송 정보</summary>
        {guideLane && guide ? (
          <>
            <p className="card-description" data-testid="guide-description">
              {guide.followProviderText}가 단계를 확인하고, {PLAN_MODEL_OPTION[guide.planModel].provider}가 계획과 완료 확인을 맡습니다.
              DINO·SAM 추적기가 위치를 갱신합니다. 모델 확인마다 화면 한 장이 전송되며 비용이 발생할 수 있습니다.
              기호는 대상 전체의 동작을 뜻하며, 세부 부품의 정확한 접촉 위치를 뜻하지 않습니다.
            </p>
            <label className="checkbox-label" data-testid="guide-visuals-toggle">
              <input type="checkbox" checked={guide.visualsEnabled} onChange={(event) => guide.onToggleVisuals(event.target.checked)} />
              그림 안내 표시 (끄면 문장으로 안내)
            </label>
          </>
        ) : (
          <p className="card-description">
            시작 분석은 외부 LLM에 화면 한 장을 보냅니다. 이후 추적은 사용자의 GPU에서만 처리합니다.
            “대상 직접 지정”은 LLM 없이 GPU 추적기로 전송됩니다.
          </p>
        )}
      </details>

      {analyzing && (
        <p className="card-description" data-testid="track-stage" role="status">
          장면을 분석해 추적할 대상을 고르는 중입니다. 그만두려면 “정지”를 누르세요.
        </p>
      )}

      {!props.analysisReady && (
        <p className="card-description" data-testid="track-analysis-ready">
          장면 분석(LLM)을 사용할 수 없습니다 — 제공자가 준비되지 않았습니다. “대상 직접 지정”으로 대상을
          그리면 LLM 없이 추적을 시작할 수 있습니다.
        </p>
      )}


      {managedLabel && track.shown && (
        <p className="card-description track-target-readonly" data-testid="track-target">
          추적 대상: <strong>{managedLabel}</strong>
          {track.target !== MANUAL_SELECTION_TARGET && (
            <> · 다르면 “대상 직접 지정”으로 그려주세요.</>
          )}
        </p>
      )}

      {!props.trackerReady && (
        <p className="card-description" data-testid="track-ready">
          추적기: 사용 불가 — 추적 서버가 준비되지 않았습니다. (안내 서버 준비 상태와는 별개입니다.)
        </p>
      )}

      {!props.consent && (
        <div className="consent-box" data-testid="track-consent">
          <p>
            추적을 시작하려면 AI 동의가 필요합니다. “시작”의 장면 분석은 촬영한 화면 한 장을{' '}
            <strong>설정된 외부 LLM 제공자</strong>에게 전송하고, 그 뒤 추적 프레임과 “대상 직접 지정”으로 그린
            프레임은 <strong>사용자의 GPU 추적 서버</strong>로 전송됩니다(두 경로 모두 이 기기에서 처리되지
            않습니다).
          </p>
          <button type="button" className="btn btn-secondary" data-testid="track-consent-grant" onClick={props.onGrantConsent}>
            AI 동의 켜기
          </button>
        </div>
      )}

      {selectionNote && (
        <p className="card-description" data-testid="track-selection-note">
          {selectionNote}
        </p>
      )}

      {track.manualCapture && (
        <div className="track-pair" data-testid="track-manual-capture">
          <div
            className="track-still"
            style={{ aspectRatio: `${track.manualCapture.pair.width} / ${track.manualCapture.pair.height}` }}
          >
            <img className="track-still-img" src={track.manualCapture.pair.imageDataUrl} alt="대상 지정을 위해 캡처한 프레임" />
            <TargetSelectOverlay
              onSelect={(seed) => void track.reseed(seed)}
              onCancel={() => track.cancelManualSelect()}
            />
          </div>
          <p className="card-description">
            화면에서 대상을 네모로 그려주세요. 이 프레임이 새 추적 실행의 첫 프레임이 됩니다.
          </p>
        </div>
      )}

      {stateNote && !track.manualCapture && (
        <p className="card-description" data-testid="track-note">
          {stateNote}
        </p>
      )}

      {track.shown && !track.manualCapture && (
        <details className="track-diagnostics" data-testid="track-diagnostics">
          <summary>진단: 추적 프레임 쌍</summary>
          <div className="track-pair" data-testid="track-pair">
            <div
              className="track-still"
              style={{ aspectRatio: `${track.shown.pair.width} / ${track.shown.pair.height}` }}
            >
              <img className="track-still-img" src={shownSrc} alt="추적에 보낸 프레임" />
              {drawBox && (
                <TrackOverlay
                  key={`${track.shown.response.track_id}-${track.shown.response.generation}`}
                  box={drawBox}
                />
              )}
            </div>
            <p className="card-description" data-testid="track-age" data-track-stale={stale ? 'true' : 'false'}>
              추적 프레임 · 요청 순간 캡처 약 {(track.ageMs / 1000).toFixed(1)}초 전 · 프레임 #{track.shown.pair.frameSeq} ·{' '}
              {track.processing ? '처리 중' : '대기'} · 서버 처리 {track.shown.response.ingest_age_ms}ms
              {stale ? ' · 오래되어 상자를 숨겼습니다' : ''}
            </p>
          </div>
        </details>
      )}

      <p className="card-description live-cost" data-testid="track-analysis-cost">
        {props.analysisCostText}
      </p>
      {guideLane && guide && (
        <p className="card-description live-cost" data-testid="guide-cost">
          {guide.costText}
        </p>
      )}

      {track.error && (
        <div className="alert-card alert-error" role="alert" data-testid="track-error">
          <span className="alert-icon">⚠️</span>
          <span className="alert-message">{track.error.message}</span>
        </div>
      )}
    </div>
  );
}
