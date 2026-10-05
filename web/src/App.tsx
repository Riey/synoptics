import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { BuildInfo, GuideHealthResponse, MaterialInput } from './generated/api.generated';
import { createSession, endSession, getHealth } from './coach/api';
import { formatBuildLine } from './buildInfo';
import {
  EMPTY_GUIDE_USAGE,
  EMPTY_USAGE_TOTALS,
  GUIDE_STAGES,
  addGuideCall,
  addGuideUnknownAttempt,
  addProviderUsage,
  addUsage,
  describeGuideStageCost,
  peakCostUsd,
  type GuideUsageTotals,
  type LiveUsageTotals,
} from './coach/liveCost';
import { accessModeStatusSuffix, codeLessAccessNotice, type AccessModeName } from './coach/accessNotice';
import { countGuideCalls, guideHideRatio } from './coach/debugTrace';
import { useIntentLoop, describeTrigger, type GuideUsageEvent } from './intent/useIntentLoop';
import { NarrationBar } from './components/NarrationBar';
import { PlanCard } from './components/PlanCard';
import { TalkBar } from './components/TalkBar';
import { VoiceToggle } from './components/VoiceToggle';
import { useSpeechInput } from './voice/useSpeechInput';
import { useSpeechOutput } from './voice/useSpeechOutput';
import { useCameraStream, type CameraStatus } from './camera/useCameraStream';
import { BugReportToggle } from './components/BugReportToggle';
import { useBugReporter } from './debug/useBugReporter';
import { CameraStage } from './components/CameraStage';
import { TrackPanel } from './components/TrackPanel';
import { useObjectTrack, type SelectionBilling } from './tracking/useObjectTrack';
import { loadCameraView, loadDraftInputs, saveCameraView, saveDraftInputs } from './draftStorage';
import { PlanMaterialPanel, type MaterialDraftState } from './planMode/PlanMaterialPanel';
import { PlanRunRecord } from './planMode/PlanRunRecord';
import {
  DEFAULT_PLAN_MODEL,
  PLAN_MODEL_OPTION,
  planModelState,
  readPlanModelAvailability,
  type PlanModel,
  type PlanModelAvailability,
} from './planMode/modelChoice';
import { DEFAULT_CORE_MODE, type CoreMode } from './planMode/coreChoice';
import './styles.css';

/**
 * The module file this code is actually running from, and the scripts the browser actually fetched. A
 * screenshot shows the UI, not which build produced it, so the panel names the running asset itself —
 * the one fact that settles "is the browser on the fixed bundle or an older cached one".
 */
const LOADED_MODULE_FILE = (() => {
  try {
    return new URL(import.meta.url).pathname.split('/').filter(Boolean).pop() ?? '알 수 없음';
  } catch {
    return '알 수 없음';
  }
})();

function loadedScriptFiles(): string[] {
  try {
    return performance
      .getEntriesByType('resource')
      .map((entry) => new URL(entry.name).pathname)
      .filter((path) => path.endsWith('.js'))
      .map((path) => path.split('/').filter(Boolean).pop() ?? path)
      .slice(-4);
  } catch {
    return [];
  }
}

/**
 * The tracking run's fence includes a scene-mode token. The still-photo surface was removed with the
 * legacy guidance lane (2026-10-01), so the camera is the only scene.
 */
const SCENE_MODE = 'camera';

const CAMERA_STATUS_TEXT: Record<CameraStatus, string> = {
  idle: '카메라 대기',
  starting: '카메라 여는 중',
  live: '실시간 연결됨',
  denied: '권한 거부됨',
  unavailable: '카메라 없음',
  error: '카메라 오류',
  insecure: 'HTTPS 필요',
  unsupported: '미지원 브라우저',
  ended: '연결 끊김',
};

/** Coverage shown over the video while no live stream is running. */
const CAMERA_PLACEHOLDER: Record<CameraStatus, { title: string; detail: string }> = {
  idle: {
    title: '카메라가 꺼져 있습니다',
    detail: '“카메라 시작”을 누르면 이 자리에 실시간 영상이 흐르고, 안내는 그 영상 위에 겹쳐 그려집니다.',
  },
  starting: { title: '카메라를 여는 중입니다', detail: '브라우저가 권한을 물으면 “허용”을 선택해주세요.' },
  live: { title: '', detail: '' },
  denied: {
    title: '카메라 권한이 거부되었습니다',
    detail: '주소창의 카메라 아이콘에서 권한을 “허용”으로 바꾼 뒤 다시 시도해주세요.',
  },
  unavailable: { title: '카메라를 찾지 못했습니다', detail: '카메라를 연결한 뒤 다시 시도해주세요.' },
  error: { title: '카메라를 열지 못했습니다', detail: '다른 앱이 카메라를 쓰고 있지 않은지 확인하고 다시 시도해주세요.' },
  insecure: {
    title: 'HTTPS 또는 localhost에서만 카메라를 열 수 있습니다',
    detail:
      '브라우저는 평문 http 원격 접속에서 카메라를 차단합니다(우회 수단 없음). HTTPS로 접속하거나 이 PC에서 http://127.0.0.1로 열어주세요.',
  },
  unsupported: { title: '이 브라우저는 카메라 입력을 지원하지 않습니다', detail: '카메라를 지원하는 브라우저로 열어주세요.' },
  ended: {
    title: '카메라 연결이 끊어졌습니다',
    detail: '장치가 분리되었거나 권한이 회수되었습니다. 재시도하면 다시 연결하고, 그때까지 마지막 화면이 남아 있습니다.',
  },
};

/** Shown when the user confirms a completion the confirmer judged done twice. Never a goal status. */
const GUIDE_COMPLETION_NOTICE = '가이드 완료 확인 · 화면 판정 2회 “완료로 보임” 뒤 사용자가 완료를 확인했습니다.';

export const App: React.FC = () => {
  // Session & consent
  const [accessCode, setAccessCode] = useState('');
  /** `null` until health answers: the code field stays visible and the server decides. */
  const [accessCodeRequired, setAccessCodeRequired] = useState<boolean | null>(null);
  /** The server's session mode (`code` / `local` / `open`); picks the code-less notice's wording. */
  const [accessMode, setAccessMode] = useState<AccessModeName>(null);
  /** The tracker's own readiness, independent of the paid provider's (`health.tracker_ready`). */
  const [trackerReady, setTrackerReady] = useState(false);
  /** Whether the paid provider is configured; the startup analysis needs it, manual drawing does not. */
  const [providerReady, setProviderReady] = useState(false);
  /** The guide lane's readiness and its follower, from `/api/health` (GuideHealthResponse). */
  const [guideReady, setGuideReady] = useState(false);
  const [followProvider, setFollowProvider] = useState<'local' | 'clef' | 'deepseek' | null>(null);
  const [followReady, setFollowReady] = useState(false);
  /** The plan lane: whether any plan model can run (`health.plan_ready`), and each model's key (`plan_models`). */
  const [planReady, setPlanReady] = useState(false);
  const [planModels, setPlanModels] = useState<PlanModelAvailability>(null);
  /** The plan model chosen for the next run; frozen while a run is planning or running. */
  const [planModel, setPlanModel] = useState<PlanModel>(DEFAULT_PLAN_MODEL);
  /** The core chosen for the next run (independent of the plan model); frozen while a run is active. */
  const [coreMode, setCoreMode] = useState<CoreMode>(DEFAULT_CORE_MODE);
  /** Overlay-first by default; disabling drawings reveals the accessible text instructions. */
  const [guideVisuals, setGuideVisuals] = useState(true);
  const [instructionVisible, setInstructionVisible] = useState(false);
  /** Page-lifetime accounting of the guide lane, per stage and per provider. */
  const [guideUsage, setGuideUsage] = useState<GuideUsageTotals>(EMPTY_GUIDE_USAGE);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [aiConsent, setAiConsent] = useState(false);
  const [healthStatus, setHealthStatus] = useState('확인 중');
  /** Which server build answered `/api/health` (null until it does, or from an older server). */
  const [serverBuild, setServerBuild] = useState<BuildInfo | null>(null);
  const [isEndingSession, setIsEndingSession] = useState(false);

  // Live camera (the only scene surface)
  const camera = useCameraStream();
  const [cameraFrameSize, setCameraFrameSize] = useState<{ width: number; height: number } | null>(null);
  /** Mirror preference: it flips the preview and every JPEG that is sent, so the model sees what the user sees. */
  const [mirrorView, setMirrorView] = useState(() => loadCameraView().mirror);
  /**
   * Page-lifetime accounting for the startup target-selection call. Kept in the host, not the tracking
   * card, so paid spend survives the card's own resets.
   */
  const [selectionUsage, setSelectionUsage] = useState<LiveUsageTotals>(EMPTY_USAGE_TOTALS);

  // Inputs owned by the human / calling agent
  const [initialDraft] = useState(() => loadDraftInputs());
  const [goal, setGoal] = useState(initialDraft.user_goal);
  const [context, setContext] = useState(initialDraft.context);
  const [guideMode, setGuideMode] = useState<'basic' | 'reference'>('basic');
  /** The accepted reference materials (page-lifetime only); sent to the plan request when reference mode is on. */
  const [materials, setMaterials] = useState<MaterialInput[]>([]);
  /** Set while the reference panel holds an incomplete edit: Start is blocked rather than sending a partial. */
  const [materialDraftIncomplete, setMaterialDraftIncomplete] = useState(false);
  const [draftSaveFailed, setDraftSaveFailed] = useState(false);
  const draftInputsRef = useRef({ user_goal: initialDraft.user_goal, context: initialDraft.context });

  const updateDraftInputs = useCallback((patch: Partial<{ user_goal: string; context: string }>) => {
    const next = {
      user_goal: patch.user_goal !== undefined ? patch.user_goal : draftInputsRef.current.user_goal,
      context: patch.context !== undefined ? patch.context : draftInputsRef.current.context,
    };
    draftInputsRef.current = next;
    if (patch.user_goal !== undefined) setGoal(patch.user_goal);
    if (patch.context !== undefined) setContext(patch.context);
    const ok = saveDraftInputs(next);
    setDraftSaveFailed(!ok);
  }, []);

  // Status & feedback
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusNotice, setStatusNotice] = useState<string | null>(null);

  /**
   * One app session, shared by every consumer. The guide and the tracker can both need a session at the
   * same moment; a single in-flight creation means they share one cookie instead of racing. The promise
   * is cleared on rejection (never cached), and `sessionFenceRef` lets consent revocation or an explicit
   * end invalidate a creation that is still in flight.
   */
  const sessionIdRef = useRef<string | null>(null);
  const sessionPromiseRef = useRef<Promise<string> | null>(null);
  const sessionFenceRef = useRef(0);
  /**
   * Serializes every cookie-affecting session op (create, end). `endSession` addresses the session by
   * cookie, not by id, so a cleanup that ran concurrently with a creation would release the WRONG
   * session — the newest one. Queueing them means a creation always runs after the end that belongs to
   * the previous session, and a discarded creation's own cleanup can only release that orphan.
   */
  const sessionChainRef = useRef<Promise<void>>(Promise.resolve());

  const enqueueSession = useCallback(<T,>(op: () => Promise<T>): Promise<T> => {
    const run = sessionChainRef.current.then(op, op);
    sessionChainRef.current = run.then(
      () => undefined,
      () => undefined
    );
    return run;
  }, []);

  useEffect(() => {
    let unmounted = false;
    getHealth()
      .then((health) => {
        if (unmounted) return;
        const provider = health.provider ?? 'gemini';
        setAccessCodeRequired(health.access_code_required);
        setTrackerReady(health.tracker_ready === true);
        setProviderReady(health.ready === true);
        // The server answers GuideHealthResponse (HealthResponse plus the guide fields); older servers omit them.
        const guideHealth = health as GuideHealthResponse;
        setServerBuild(guideHealth.build ?? null);
        setGuideReady(guideHealth.guide_ready === true);
        setPlanReady(guideHealth.plan_ready === true);
        setPlanModels(readPlanModelAvailability(guideHealth.plan_models));
        setFollowReady(guideHealth.follow_ready === true);
        setFollowProvider(
          guideHealth.follow_provider === 'local' || guideHealth.follow_provider === 'clef' || guideHealth.follow_provider === 'deepseek'
            ? guideHealth.follow_provider
            : null
        );
        setAccessMode(guideHealth.access_mode ?? null);
        const modeSuffix = accessModeStatusSuffix(health.access_code_required, guideHealth.access_mode);
        setHealthStatus(health.ready ? `준비 완료 (${provider})${modeSuffix}` : '대기 중');
      })
      .catch(() => {
        if (!unmounted) setHealthStatus('오프라인');
      });
    return () => {
      unmounted = true;
    };
  }, []);

  const resetSession = useCallback(() => {
    sessionFenceRef.current += 1;
    sessionPromiseRef.current = null;
    sessionIdRef.current = null;
    setSessionId(null);
  }, []);

  /**
   * The one app session, shared by every consumer.
   *
   * The creation is deliberately NOT tied to a caller's AbortSignal: one consumer's cancellation must not
   * cancel the session another consumer is waiting on. It is fenced by `sessionFenceRef` instead — consent
   * revocation and session end bump it, so a creation that lands afterwards is discarded rather than
   * adopted. A rejected creation is cleared from the ref so the next call retries instead of replaying the
   * failure.
   */
  const ensureSession = useCallback((): Promise<string> => {
    if (sessionIdRef.current) return Promise.resolve(sessionIdRef.current);
    if (sessionPromiseRef.current) return sessionPromiseRef.current;
    // `access_code_required` comes from the server, never from the address the browser happens to use.
    // A code-protected deployment needs the entered code before any provider traffic; a code-less one
    // (false) never sends a code at all. While health is still unknown the request goes out without a
    // code and the server stays the authority — a protected backend answers 401 instead of being guessed at.
    const code = accessCodeRequired === false ? '' : accessCode.trim();
    if (accessCodeRequired === true && !code) {
      return Promise.reject(new Error('데모 접근 코드를 입력해주세요.'));
    }
    const fence = sessionFenceRef.current;
    let shared!: Promise<string>;
    shared = enqueueSession(async () => {
      if (fence !== sessionFenceRef.current) throw new Error('요청이 취소되었습니다.');
      const response = await createSession(code || null);
      if (fence !== sessionFenceRef.current) {
        // Consent went away (or an explicit end landed) while this creation was in flight, so the session
        // it just made must not be adopted. Creations are serialized, so this is still the cookie's session
        // and no newer one exists yet: ending it releases this orphan and nothing else. If the end cannot
        // reach the server the orphan expires on the server's own TTL.
        await endSession().catch(() => {});
        throw new Error('요청이 취소되었습니다.');
      }
      sessionIdRef.current = response.session_id;
      setSessionId(response.session_id);
      return response.session_id;
    }).catch((error: unknown) => {
      if (sessionPromiseRef.current === shared) sessionPromiseRef.current = null;
      throw error;
    });
    sessionPromiseRef.current = shared;
    return shared;
  }, [accessCode, accessCodeRequired, enqueueSession]);

  const handleRevokeConsent = async () => {
    setAiConsent(false);
    // Fence any in-flight session creation so a late one cannot be adopted after consent is gone.
    const previousSessionId = sessionIdRef.current;
    resetSession();
    if (!previousSessionId) return;
    setIsEndingSession(true);
    try {
      // Queued behind any in-flight creation/cleanup: a creation that starts later is queued behind
      // THIS end, so the end can only release the session the revocation is about.
      await enqueueSession(() => endSession());
    } catch {
      setErrorMessage(
        '서버 세션 종료 알림이 네트워크로 지연되었습니다. 서버 세션은 곧 자동 만료되지만 진행 중인 작업의 즉각 취소는 보장되지 않습니다.'
      );
    } finally {
      setIsEndingSession(false);
    }
  };

  const handleToggleMirror = (next: boolean) => {
    // The tracking run and the guide both fence on the mirror flag, so the old orientation's run retires.
    setMirrorView(next);
    saveCameraView({ mirror: next });
  };

  // Goal and context edits are a new task: the tracking run and the guide fence on them.
  const taskTextKey = useMemo(
    () => JSON.stringify([goal, context, guideMode, guideMode === 'reference' ? materials : []]),
    [goal, context, guideMode, materials]
  );

  // The startup analysis is a paid call, reported with the same priced/unpriced/unknown rules and the
  // same page-lifetime scope as the guide lane.
  const handleSelectionBilling = useCallback((billing: SelectionBilling) => {
    setSelectionUsage((previous) =>
      billing.kind === 'unknown'
        ? addUsage(previous, null)
        : addProviderUsage(previous, billing.provider, billing.model, billing.usage)
    );
  }, []);
  // The tracking run is owned here (not by the card) so the camera stage can draw its live box over the
  // video and the guide can read the same run. Its per-frame output goes through `track.liveTrack`
  // outside React; `track.shown` is a throttled diagnostic still.
  const track = useObjectTrack({
    videoRef: camera.videoRef,
    ensureSession,
    sessionId,
    consent: aiConsent,
    cameraLive: camera.status === 'live',
    mirror: mirrorView,
    sourceEpoch: camera.sourceEpoch,
    sceneMode: SCENE_MODE,
    cameraFrameSizeKey: `${cameraFrameSize?.width ?? 0}x${cameraFrameSize?.height ?? 0}`,
    taskKey: taskTextKey,
    userGoal: goal,
    context,
    onSelectionBilling: handleSelectionBilling,
  });
  const handleGuideUsage = useCallback((event: GuideUsageEvent) => {
    setGuideUsage((previous) =>
      event.kind === 'billed'
        ? addGuideCall(previous, event.stage, event.provider, event.model, event.usage)
        : addGuideUnknownAttempt(previous, event.stage, event.lane)
    );
  }, []);
  // Recorded as the user's own confirmation, never written into a goal status: the twice-confirmed
  // judgement and this press are both shown for what they are.
  const handleGuideCompletion = useCallback(() => setStatusNotice(GUIDE_COMPLETION_NOTICE), []);
  // The reference panel reports an edited-but-incomplete source; Start blocks on it instead of sending a partial.
  const handleMaterialDraftState = useCallback(
    (state: MaterialDraftState) => setMaterialDraftIncomplete(state.hasDraft && !state.complete),
    []
  );
  const intent = useIntentLoop({
    videoRef: camera.videoRef,
    mirror: mirrorView,
    liveTrack: track.liveTrack,
    ensureSession,
    sessionId,
    consent: aiConsent,
    cameraLive: camera.status === 'live',
    userGoal: goal,
    context,
    materials: guideMode === 'reference' ? materials : null,
    fenceKey: `${taskTextKey}\u0000${camera.sourceEpoch}\u0000${mirrorView}\u0000${SCENE_MODE}\u0000${planModel}\u0000${coreMode}`,
    planModel,
    coreMode,
    followProvider,
    startTracking: track.startWithTarget,
    stopTracking: track.stop,
    onUsage: handleGuideUsage,
    onSessionExpired: resetSession,
    onUserConfirmedCompletion: handleGuideCompletion,
  });
  // Voice is a second channel for the same words: feedback reads what the narration/talk bar shows, a
  // spoken utterance goes through `intent.talk` exactly like a typed one. The speaker yields while the
  // microphone is open so the recogniser never hears the guide.
  const bugReportSnapshot = () => ({
    goal,
    context,
    session_id: sessionId,
    mirror: mirrorView,
    scene_mode: SCENE_MODE,
    plan_model: planModel,
    core_mode: coreMode,
    follow_provider: followProvider,
    server_build: serverBuild,
    web_commit: __WEB_COMMIT__,
    camera: {
      status: camera.status,
      frame_size: cameraFrameSize,
      device: camera.devices.find((device) => device.deviceId === camera.activeDeviceId)?.label ?? null,
    },
    track_phase: track.phase,
    view: intent.view,
    trace: intent.trace,
    guide_usage: guideUsage,
    error_message: errorMessage,
    status_notice: statusNotice,
  });
  // Debug bug reports: while the checkbox is on, the camera recording and this snapshot stream to the
  // server's report directory; the timeline notes guide events so the video can be lined up with them.
  const bugReporter = useBugReporter({
    videoRef: camera.videoRef,
    live: camera.status === 'live',
    sourceEpoch: camera.sourceEpoch,
    snapshot: bugReportSnapshot,
  });
  const { mark } = bugReporter;
  const view = intent.view;
  useEffect(() => {
    mark('guide', {
      phase: view.phase,
      stepIndex: view.stepIndex,
      planRevision: view.plan?.planRevision ?? null,
      step: view.plan?.steps[view.stepIndex] ?? null,
      notice: view.notice,
      completion: view.completion,
      lastTrigger: view.lastTrigger,
    });
  }, [mark, view.phase, view.stepIndex, view.plan, view.notice, view.completion, view.lastTrigger]);
  useEffect(() => {
    if (view.talk) mark('talk', view.talk);
  }, [mark, view.talk]);
  useEffect(() => {
    if (errorMessage) mark('error', errorMessage);
  }, [mark, errorMessage]);
  const speechOutput = useSpeechOutput(intent.view);
  const speechInput = useSpeechInput({
    onFinal: (transcript) => void intent.talk(transcript),
    onListening: (listening) => (listening ? speechOutput.pause() : speechOutput.resume()),
  });
  const guideCostText = `가이드 · ${GUIDE_STAGES.map(
    (stage) => `${stage === 'plan' ? '계획' : stage === 'follow' ? '단계 확인' : stage === 'confirm' ? '완료 판정' : '대화'} ${describeGuideStageCost(guideUsage, stage)}`
  ).join(' · ')}`;
  const followProviderText =
    followProvider === 'local'
      ? '로컬 모델(API 요금 없음)'
      : followProvider === 'clef'
        ? '로컬 판정 모델 Clef(API 요금 없음)'
        : followProvider === 'deepseek'
          ? 'DeepSeek'
          : '설정되지 않은 추종 모델';
  // The plan lane exists when any plan model can run; guide_ready keeps a server that predates plan_models
  // (and the legacy startup-analysis lane) working.
  const guideLaneReady = guideReady || planReady;
  const planModelReady = planModelState(planModels, planModel) === 'ready';
  const guideTrace = intent.trace;
  const guideCallCounts = countGuideCalls(guideTrace);
  const guideHide = guideHideRatio(guideTrace);
  const selectionCostParts: string[] = [];
  if (selectionUsage.input + selectionUsage.output > 0) {
    selectionCostParts.push(`누적 추정 비용(피크 단가 기준) $${peakCostUsd(selectionUsage).toFixed(6)}`);
  }
  if (selectionUsage.unpricedStages > 0) {
    selectionCostParts.push(
      `단가 미확인 ${selectionUsage.unpricedStages}회(토큰만 기록: 입력 ${selectionUsage.unpricedInput} / 출력 ${selectionUsage.unpricedOutput})`
    );
  }
  if (selectionUsage.unknownAttempts > 0) {
    selectionCostParts.push(`사용량을 못 받은 시도 ${selectionUsage.unknownAttempts}회(비용 불명)`);
  }
  const selectionCostText = `시작 분석 · ${
    selectionCostParts.length > 0 ? selectionCostParts.join(' · ') : '아직 실행되지 않았습니다'
  }`;
  const loadedScripts = loadedScriptFiles();

  // Playback pending is its own visible state: the stream is attached but nothing is advancing, so the
  // badge must not claim a live connection.
  const playbackPending = camera.status === 'live' && camera.playBlocked;
  const cameraStatusText = playbackPending ? '재생 대기 중' : CAMERA_STATUS_TEXT[camera.status];
  const cameraPlaceholderInfo = CAMERA_PLACEHOLDER[camera.status];
  const cameraPlaceholder =
    camera.status === 'live'
      ? null
      : { title: cameraPlaceholderInfo.title, detail: camera.message ?? cameraPlaceholderInfo.detail };
  const cameraCanRetry =
    camera.status === 'denied' || camera.status === 'unavailable' || camera.status === 'error' || camera.status === 'ended';
  const cameraCanStart = camera.status === 'idle' || cameraCanRetry;

  return (
    <div className="app-wrapper">
      <header className="app-header">
        <div className="header-brand">
          <img
            className="header-mark"
            src="/brand-mark-48.png"
            srcSet="/brand-mark-48.png 1x, /brand-mark-96.png 2x"
            width={48}
            height={48}
            alt=""
            aria-hidden="true"
            decoding="async"
          />
          <div>
            <h1 className="header-title">synoptics</h1>
            <p className="header-subtitle">라이브 카메라 가이드 — 지금 보이는 화면 위에 다음 행동을 그려줍니다.</p>
          </div>
        </div>
      </header>

      <main className="main-layout">
        <section className="workspace-column">
          <div className="card">
            <div className="card-header-flex">
              <h2 className="card-title">1. 실시간 카메라 화면</h2>
            </div>

            <CameraStage
              videoRef={camera.videoRef}
              status={camera.status}
              statusText={cameraStatusText}
              onIntrinsicSize={setCameraFrameSize}
              mirrored={mirrorView}
              onResumePlayback={camera.resumePlayback}
              playBlocked={playbackPending}
              placeholder={cameraPlaceholder}
              liveTrack={track.liveTrack}
              liveTrackPhase={track.phase}
              guideOverlay={intent.overlay}
              onInstructionVisibility={setInstructionVisible}
              visualsEnabled={guideVisuals}
              narration={
                intent.view.phase !== 'idle' || intent.view.notice || intent.view.error || intent.view.plan ? (
                  <NarrationBar view={intent.view} instructionVisible={guideVisuals && instructionVisible} visualsEnabled={guideVisuals} onConfirmCompletion={intent.confirmCompletion} />
                ) : null
              }
            />

            <VoiceToggle speechOutput={speechOutput} />
            <TalkBar view={intent.view} onSend={intent.talk} speechInput={speechInput} />

            <TrackPanel
              track={track}
              consent={aiConsent}
              cameraLive={camera.status === 'live'}
              trackerReady={trackerReady}
              analysisReady={providerReady && guideMode === 'basic'}
              analysisCostText={selectionCostText}
              onGrantConsent={() => setAiConsent(true)}
              guide={{
                ready: guideLaneReady,
                followerReady: followReady,
                followProviderText,
                planModel,
                planModels,
                modelReady: planModelReady,
                onSelectPlanModel: setPlanModel,
                coreMode,
                onSelectCoreMode: setCoreMode,
                planning: intent.view.phase === 'planning',
                running: intent.view.phase === 'running',
                startBlockedReason:
                  guideMode === 'reference' && materialDraftIncomplete ? '입력한 참고자료를 확인해 주세요.' : null,
                onStart: () => void intent.start(),
                onStop: intent.stop,
                onManual: intent.manual,
                onReplan: intent.replan,
                replanBlockedReason: intent.view.replanBlockedReason,
                costText: guideCostText,
                visualsEnabled: guideVisuals,
                onToggleVisuals: setGuideVisuals,
              }}
            />

            <PlanCard view={intent.view} />
            <PlanRunRecord view={intent.view} materials={materials} goal={goal} active={guideMode === 'reference'} />

            <div className="consent-box">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={aiConsent}
                  onChange={(event) => {
                    if (event.target.checked) {
                      setAiConsent(true);
                    } else {
                      void handleRevokeConsent();
                    }
                  }}
                />
                <span>
                  그림 안내는 계획·완료 확인·대화에 {PLAN_MODEL_OPTION[planModel].provider}, 단계 확인에 {followProviderText}, 위치 추적에 사용자의
                  GPU 서버로 화면을 전송합니다. 참고자료 기반 안내는 입력한 자료도 계획·완료 확인·대화에 함께 전송합니다.
                  안내는 언제든 정지할 수 있으며, 미리보기만 켜면 전송하지 않고 서버는 이미지를 저장하지 않습니다.
                </span>
              </label>
            </div>

            {/* Status and errors sit beside the camera instead of above it: a banner that lands on top of
                the page pushes the live picture below the fold on phones, which is the one thing that must
                stay visible. */}
            {errorMessage && (
              <div className="alert-card alert-error" role="alert">
                <span className="alert-icon">⚠️</span>
                <span className="alert-message">{errorMessage}</span>
                <button
                  type="button"
                  className="alert-dismiss-btn"
                  onClick={() => setErrorMessage(null)}
                  aria-label="오류 메시지 닫기"
                >
                  ✕
                </button>
              </div>
            )}

            {statusNotice && (
              <div className="alert-card alert-warning" role="status">
                <span className="alert-icon">ℹ️</span>
                <span className="alert-message">{statusNotice}</span>
              </div>
            )}

            <div className="button-group" data-testid="camera-controls">
              <span className="header-badge badge-live">
                <span className="status-dot" aria-hidden="true" />
                <span>서버: {healthStatus}</span>
              </span>
              <span className="build-line" data-testid="build-line">{formatBuildLine(serverBuild, __WEB_COMMIT__)}</span>
              {cameraCanStart && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  data-testid="camera-start"
                  disabled={camera.status === 'starting'}
                  onClick={() => void camera.start()}
                >
                  카메라 시작
                </button>
              )}
              {camera.status === 'live' && (
                <button type="button" className="btn btn-secondary" data-testid="camera-stop" onClick={camera.stop}>
                  카메라 정지
                </button>
              )}
              {cameraCanRetry && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  data-testid="camera-retry"
                  onClick={() => void camera.start(camera.activeDeviceId ?? undefined)}
                >
                  다시 시도
                </button>
              )}
              <label className="mirror-toggle-field">
                <input
                  type="checkbox"
                  data-testid="mirror-toggle"
                  checked={mirrorView}
                  onChange={(event) => handleToggleMirror(event.target.checked)}
                />
                <span>
                  거울 보기 — 미리보기와 전송 사진을 좌우 반전합니다. 안내 좌표와 글자는 뒤집지 않고,
                  방향 안내는 “화면 기준”으로 표시됩니다.
                </span>
              </label>
              <BugReportToggle reporter={bugReporter} />
              {camera.devices.length > 1 && (
                <label className="camera-device-field">
                  <span className="form-label">카메라 선택</span>
                  <select
                    className="input-text"
                    data-testid="camera-device-select"
                    value={camera.activeDeviceId ?? ''}
                    onChange={(event) => camera.selectDevice(event.target.value)}
                  >
                    {camera.devices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>

            <details className="disclosure">
              <summary className="disclosure-summary">표시 방식</summary>
              <p className="card-description">
                그림 안내는 추적 중인 대상에 붙어 움직이며, 위치가 오래됐거나 가려지면 숨깁니다. 영상은 캡처·대기
                중에도 계속 흐르며 재생이 막히면 캡처하지 않습니다.
              </p>
              <p className="card-description">
                “거울 보기”를 켜면 미리보기와 전송 이미지가 함께 좌우 반전되어 AI가 보는 방향이 화면과
                일치합니다. 안내 도형과 글자는 반전하지 않습니다.
              </p>
            </details>
          </div>
        </section>

        <section className="workspace-column">
          <div className="card">
            <h2 className="card-title">2. 목표 · 맥락</h2>
            <fieldset className="mode-picker">
              <legend className="form-label">안내 모드</legend>
              <label>
                <input type="radio" name="guide-mode" value="basic" checked={guideMode === 'basic'}
                  onChange={() => setGuideMode('basic')} />
                기본 · 목표로 안내
              </label>
              <label>
                <input type="radio" name="guide-mode" value="reference" checked={guideMode === 'reference'}
                  onChange={() => setGuideMode('reference')} />
                참고자료 · 문서 기반 안내 (데모)
              </label>
            </fieldset>
            {guideMode === 'reference' && (
              <p className="card-description">
                작업서의 절차와 현재 화면을 비교합니다. AI 안내와 자료 원문을 함께 확인하고, 화면에 보이지 않는 상태는 직접
                확인해 주세요. 계획을 검토하고 승인·수정하는 실제 Plan 흐름은 Live PWA에서 진행합니다.
              </p>
            )}
            <div className="form-group">
              <label className="form-label" htmlFor="goal-input">
                작업 목표
              </label>
              <textarea
                id="goal-input"
                className="input-text"
                rows={2}
                maxLength={300}
                value={goal}
                placeholder={guideMode === 'reference' ? '예: 포장 작업서에 따라 유리컵을 안전하게 포장하기' : '예: 지금 화면에서 다음으로 무엇을 해야 하는지 알려줘'}
                onChange={(event) => updateDraftInputs({ user_goal: event.target.value })}
              />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="context-input">
                작업 맥락 (선택)
              </label>
              <textarea
                id="context-input"
                className="input-text"
                rows={2}
                maxLength={1000}
                value={context}
                placeholder="예: 에이전트가 준 절차·배경 설명을 그대로 붙여넣으세요"
                onChange={(event) => updateDraftInputs({ context: event.target.value })}
              />
            </div>
            {draftSaveFailed ? (
              <p className="form-feedback form-feedback-error" role="alert">
                브라우저 저장 공간이 제한되어 변경사항이 저장되지 않았습니다. 입력 내용은 계속 사용하실 수 있습니다.
              </p>
            ) : (
              <p className="card-description">
                목표와 맥락만 브라우저에 자동 저장됩니다. 참고자료와 작업 기록은 이 페이지에서만 유지됩니다.
              </p>
            )}
            <p className="card-description">
              목표·맥락·모드·참고자료를 바꾸면 진행 중인 가이드와 추적이 멈춥니다. 다시 “가이드 시작”을 누르면 새 기준으로 계획합니다.
            </p>
          </div>
          <div hidden={guideMode !== 'reference'}>
            <PlanMaterialPanel materials={materials} onChange={setMaterials} onDraftStateChange={handleMaterialDraftState}
              ensureSession={ensureSession} disabled={isEndingSession || guideMode !== 'reference'} />
          </div>

          <div className="card">
            <h2 className="card-title">3. 세션 · 접속</h2>
            {accessCodeRequired === false ? (
              <p className="card-description" data-testid="local-mode-notice" data-access-mode={accessMode ?? undefined}>
                {codeLessAccessNotice(accessMode)}
              </p>
            ) : (
              <div className="form-group">
                <label className="form-label" htmlFor="access-code-input">
                  데모 접근 코드
                </label>
                <input
                  id="access-code-input"
                  className="input-text"
                  type="password"
                  value={accessCode}
                  autoComplete="off"
                  onChange={(event) => setAccessCode(event.target.value)}
                />
              </div>
            )}
            <div className="button-group">
              <button
                type="button"
                className="btn btn-secondary"
                disabled={isEndingSession}
                onClick={() => void handleRevokeConsent()}
              >
                전송 중단 · 세션 종료
              </button>
            </div>
            <p className="card-description">
              전송을 중단하면 진행 중인 가이드와 추적이 멈추고 받은 안내가 지워집니다. 카메라 영상은 이 브라우저
              안에서만 처리됩니다. 원격 접속에서 카메라를 쓰려면 HTTPS가 필요하며, 평문 http 원격 접속은 브라우저가
              차단합니다(우회하지 않습니다).
            </p>
          </div>

          <div className="card" data-testid="debug-panel">
            <h2 className="card-title">4. 디버그 · 가이드 추적</h2>
            <p className="card-description">
              이 브라우저 안에서만 보이는 진단 기록입니다. 가이드 레인의 시간·횟수와 최근 호출을 숫자로
              보여줍니다. 이미지와 모델 응답 문장은 기록하지 않고, 페이지를 닫으면 사라집니다. “버그 리포트
              전송”을 켠 동안에만 카메라 녹화와 가이드 상태가 디버깅용으로 서버에 저장됩니다.
            </p>

            <div
              className="debug-wait"
              data-testid="debug-build"
              data-module-file={LOADED_MODULE_FILE}
              data-loaded-scripts={loadedScripts.join(',')}
            >
              <strong>브라우저가 실행 중인 빌드</strong>
              <p className="card-description debug-numbers">
                {`모듈 파일=${LOADED_MODULE_FILE} · 이 페이지가 받아온 스크립트=${loadedScripts.length > 0 ? loadedScripts.join(', ') : '없음'}`}
              </p>
              <p className="text-secondary">
                화면 캡처로는 알 수 없는 값입니다. 지금 보고 있는 창이 어떤 번들 파일을 실제로 실행했는지
                보여줍니다.
              </p>
            </div>

            <div
              className="debug-wait"
              data-testid="debug-guide"
              data-first-say-ms={guideTrace.msStartToFirstSay ?? ''}
              data-final-plan-ms={guideTrace.msStartToFinalPlan ?? ''}
              data-first-box-ms={guideTrace.msStartToFirstBox ?? ''}
              data-target-changed={guideTrace.targetChanged}
              data-target-moved={guideTrace.targetMoved}
              data-follow-skip={guideTrace.followSkip}
              data-confirm-advance={guideTrace.confirmAdvance}
              data-follow-pending-skip={guideTrace.followPendingSkip}
              data-stuck-replan={guideTrace.stuckReplan}
              data-replan-limited={guideTrace.replanLimited}
              data-budget-capped={guideTrace.budgetCapped}
              data-talk={guideTrace.talk}
              data-talk-rejected={guideTrace.talkRejected}
              data-talk-blocked={guideTrace.talkBlocked}
              data-stale-superseded={guideTrace.staleSuperseded}
              data-sequential-nominate={guideTrace.sequentialNominate}
              data-sequential-advance={guideTrace.sequentialAdvance}
              data-sequential-stale={guideTrace.sequentialStale}
            >
              <strong>가이드 레인 · 시간과 횟수(가이드 시작 기준)</strong>
              <p className="card-description debug-numbers">
                {`첫 문구=${guideTrace.msStartToFirstSay === null ? '—' : `${guideTrace.msStartToFirstSay}ms`}` +
                  ` · 최종 계획=${guideTrace.msStartToFinalPlan === null ? '—' : `${guideTrace.msStartToFinalPlan}ms`}` +
                  ` · 첫 상자=${guideTrace.msStartToFirstBox === null ? '—' : `${guideTrace.msStartToFirstBox}ms`}` +
                  ` · 단계 완료(추종)=${guideTrace.followDone} · 불확실(추종)=${guideTrace.followUnsure}` +
                  ` · 확정자 번복=${guideTrace.confirmReversals} · 확정자 단계 이동=${guideTrace.confirmAdvance}` +
                  ` · 대상 이동 트리거=${guideTrace.targetMoved}` +
                  ` · 대상 변화 트리거=${guideTrace.targetChanged} · 건너뛰기=${guideTrace.followSkip}` +
                  ` · 건너뛰기 보류=${guideTrace.followPendingSkip} · 막힘 재계획=${guideTrace.stuckReplan}` +
                  ` · 재계획 한도=${guideTrace.replanLimited}` +
                  ` · AI 호출 한도 대기=${guideTrace.budgetCapped}` +
                  ` · 대화=${guideTrace.talk}(설명 ${guideTrace.talkSay} · 대상 ${guideTrace.talkTarget} · 표시 ${guideTrace.talkMark}` +
                  ` · 되돌림 ${guideTrace.talkGoTo} · 재계획 ${guideTrace.talkReplan} · 폐기 ${guideTrace.talkRejected} · 차단 ${guideTrace.talkBlocked})` +
                  ` · 지난 계획 응답 폐기=${guideTrace.staleSuperseded}` +
                  ` · 순차 코어 후보=${guideTrace.sequentialNominate} · 상위 확인 확정=${guideTrace.sequentialAdvance}` +
                  ` · 폐기/보류=${guideTrace.sequentialStale}` +
                  ` · 300ms 숨김 비율=${guideHide === null ? '—' : `${(guideHide * 100).toFixed(1)}%(표본 ${guideTrace.hideSamples})`}` +
                  ` · 최근 호출 결합=${guideCallCounts.bound} / 문구만=${guideCallCounts.text_only} / 폐기=${guideCallCounts.rejected} / 실패=${guideCallCounts.failed}`}
              </p>
              {guideTrace.calls.length > 0 && (
                <ol className="guide-debug-calls" aria-label="가이드 호출(최신순)">
                  {guideTrace.calls.map((call) => (
                    <li key={call.id} data-testid="debug-guide-call" data-accepted={call.accepted} data-stage={call.stage}>
                      {`${call.stage} · ${describeTrigger(call.trigger) ?? call.trigger} · ${call.provider ?? '응답 없음'} · ${
                        call.msToFinal === null ? '—' : `${call.msToFinal}ms`
                      } · ${call.accepted}${call.errorCode ? `(${call.errorCode})` : ''}`}
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};
