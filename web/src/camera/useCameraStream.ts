/**
 * Owns the live camera stream: permission, device selection, track lifecycle, and teardown.
 *
 * Rules this hook enforces (see /tmp/aisw-camera-contract.md):
 * - getUserMedia runs only in response to an explicit `start()` call, never on mount.
 * - audio is always false, facingMode prefers `environment` (rear) on mobile.
 * - every superseded or stopped stream has all of its tracks stopped; no stream is ever reused.
 * - `sourceEpoch` bumps whenever the live source is replaced or torn down, so callers can fence
 *   late async results against the source they were computed for.
 * - an insecure context is reported instead of attempting (and silently failing) a capture.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraStatus =
  | 'idle'
  | 'starting'
  | 'live'
  | 'denied'
  | 'unavailable'
  | 'error'
  | 'insecure'
  | 'unsupported'
  | 'ended';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export interface CameraController {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  message: string | null;
  devices: CameraDevice[];
  activeDeviceId: string | null;
  /** Bumps whenever the live scene source is replaced or torn down. */
  sourceEpoch: number;
  /** True when the browser refused to autoplay the stream (a tap on the video resumes it). */
  playBlocked: boolean;
  start: (deviceId?: string) => Promise<void>;
  stop: () => void;
  selectDevice: (deviceId: string) => void;
  resumePlayback: () => void;
}

function stopAllTracks(stream: MediaStream | null): void {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
}

function errorName(error: unknown): string {
  return typeof error === 'object' && error !== null && 'name' in error ? String(error.name) : '';
}

function describeCameraFailure(error: unknown): { status: CameraStatus; message: string } {
  switch (errorName(error)) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return {
        status: 'denied',
        message:
          '브라우저에서 카메라 사용이 거부되었습니다. 주소창의 카메라 권한을 "허용"으로 바꾼 뒤 다시 시도해주세요.',
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return { status: 'unavailable', message: '사용할 수 있는 카메라를 찾지 못했습니다. 카메라를 연결한 뒤 다시 시도해주세요.' };
    case 'NotReadableError':
    case 'TrackStartError':
      // NotReadableError does not prove another app owns the camera: OS-level capture blocks, missing
      // system camera permission, and driver/hardware failures report the same way, so the copy names the
      // things the user can actually check instead of asserting a cause.
      return {
        status: 'error',
        message: '카메라를 열 수 없습니다. 다른 앱이 카메라를 쓰고 있는지, 시스템 카메라 권한이 허용되어 있는지 확인한 뒤 다시 시도해주세요.',
      };
    case 'SecurityError':
      return { status: 'insecure', message: '보안 컨텍스트가 아니어서 카메라를 열 수 없습니다. HTTPS 또는 localhost에서 접속해주세요.' };
    case 'OverconstrainedError':
      return {
        status: 'error',
        message: '선택한 카메라를 열 수 없습니다. 다른 카메라를 선택하거나 “다시 시도”를 눌러주세요.',
      };
    default:
      return { status: 'error', message: '카메라를 시작하지 못했습니다. 잠시 후 다시 시도해주세요.' };
  }
}

/**
 * Ask for one specific camera, or for the best default camera.
 *
 * When a device was explicitly selected it stays the user's choice: a failure is reported instead of
 * silently opening a different camera.
 */
async function requestStream(deviceId?: string): Promise<MediaStream> {
  // `facingMode` is only ever an *ideal* preference and never an exact requirement: a MacBook (or any
  // machine) with a single built-in camera — or a desktop with no facing metadata at all — must still
  // start on the default video input, and the device list is never filtered by name.
  const video: MediaTrackConstraints = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
    : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } };
  return navigator.mediaDevices.getUserMedia({ audio: false, video });
}

export function useCameraStream(): CameraController {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  // Fences every in-flight getUserMedia/play against the newest start/stop request.
  const opSeqRef = useRef(0);

  const [status, setStatus] = useState<CameraStatus>('idle');
  const [message, setMessage] = useState<string | null>(null);
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null);
  const [sourceEpoch, setSourceEpoch] = useState(0);
  const [playBlocked, setPlayBlocked] = useState(false);

  const refreshDevices = useCallback(async () => {
    // enumerateDevices is optional (and labels stay empty until permission is granted): a browser that
    // cannot list devices simply offers no picker instead of failing the start.
    if (typeof navigator.mediaDevices?.enumerateDevices !== 'function') return;
    const list = await navigator.mediaDevices.enumerateDevices().catch(() => []);
    setDevices(
      list
        .filter((device) => device.kind === 'videoinput')
        .map((device, index) => ({ deviceId: device.deviceId, label: device.label || `카메라 ${index + 1}` }))
    );
  }, []);

  const handleTrackEnded = useCallback((event: Event) => {
    const track = event.target as MediaStreamTrack;
    const stream = streamRef.current;
    // Only the stream that is still current may declare the camera gone.
    if (!stream || !stream.getTracks().includes(track)) return;
    // Same ownership fence as stop(): bumping the op makes every pending start/play continuation stale,
    // and clearing streamRef means a late continuation can neither resurrect 'live' for a camera that is
    // gone nor release a stream it no longer owns. Without this, a track that ends while start() awaits
    // play() would be overwritten with 'live' the moment that play() settles.
    opSeqRef.current += 1;
    streamRef.current = null;
    for (const owned of stream.getTracks()) owned.removeEventListener('ended', handleTrackEnded);
    stopAllTracks(stream);
    const video = videoRef.current;
    if (video && video.srcObject === stream) video.srcObject = null;
    setActiveDeviceId(null);
    setPlayBlocked(false);
    setStatus('ended');
    setMessage('카메라 연결이 끊어졌습니다(장치 분리 또는 권한 회수). 재시도하면 다시 연결합니다.');
    setSourceEpoch((epoch) => epoch + 1);
  }, []);

  const start = useCallback(
    async (deviceId?: string) => {
      if (!window.isSecureContext) {
        setStatus('insecure');
        setMessage('카메라는 HTTPS 또는 localhost에서만 열 수 있습니다. 현재 주소는 보안 컨텍스트가 아닙니다.');
        return;
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('unsupported');
        setMessage('이 브라우저는 카메라 입력을 지원하지 않습니다. 사진 모드로 진행해주세요.');
        return;
      }

      const op = opSeqRef.current + 1;
      opSeqRef.current = op;

      // Whatever this hook already owns is released BEFORE asking for a replacement. Exclusive cameras
      // (front/back pairs, or one USB device claimed twice) fail to open while the old one is still
      // held, and leaving the old stream attached would let a stopped camera's last frame pass for the
      // freshly selected one.
      const owned = streamRef.current;
      if (owned) {
        streamRef.current = null;
        for (const track of owned.getVideoTracks()) track.removeEventListener('ended', handleTrackEnded);
        stopAllTracks(owned);
        const ownedVideo = videoRef.current;
        if (ownedVideo) ownedVideo.srcObject = null;
        setSourceEpoch((epoch) => epoch + 1);
      }

      setStatus('starting');
      setMessage(null);
      setPlayBlocked(false);

      let stream: MediaStream;
      try {
        stream = await requestStream(deviceId);
      } catch (error) {
        if (op !== opSeqRef.current) return;
        const failure = describeCameraFailure(error);
        // An explicitly chosen device stays chosen: the UI keeps showing it so the next action is a
        // deliberate pick, never a silent switch to another camera.
        if (deviceId) setActiveDeviceId(deviceId);
        setStatus(failure.status);
        setMessage(failure.message);
        return;
      }

      if (op !== opSeqRef.current) {
        // A newer start/stop superseded this one while the permission prompt was open.
        stopAllTracks(stream);
        return;
      }

      streamRef.current = stream;
      for (const track of stream.getTracks()) track.addEventListener('ended', handleTrackEnded);

      const video = videoRef.current;
      if (video) {
        // Safari decides autoplay from the real attributes, not from the IDL properties React sets for
        // `muted`; both are applied here, immediately before play().
        video.muted = true;
        video.defaultMuted = true;
        if (!video.hasAttribute('muted')) video.setAttribute('muted', '');
        if (!video.hasAttribute('playsinline')) video.setAttribute('playsinline', '');
        video.autoplay = true;
        video.srcObject = stream;
        try {
          await video.play();
        } catch {
          // Autoplay policy: the stage offers a tap-to-play affordance instead of failing the start.
          if (op === opSeqRef.current) setPlayBlocked(true);
        }

        // play() is asynchronous: a stop, device switch, or unmount can land while it is pending. This
        // start is then superseded and must not resurrect 'live' state for a stream nobody owns — and it
        // may only release its OWN stream, never the newer one that replaced it.
        if (op !== opSeqRef.current || streamRef.current !== stream) {
          if (streamRef.current === stream) streamRef.current = null;
          for (const track of stream.getVideoTracks()) track.removeEventListener('ended', handleTrackEnded);
          stopAllTracks(stream);
          if (video.srcObject === stream) video.srcObject = null;
          return;
        }
      }

      const track = stream.getVideoTracks()[0];
      // getSettings is optional in older engines; the active device is then simply unknown.
      const reportedDeviceId = typeof track?.getSettings === 'function' ? track.getSettings().deviceId : undefined;
      setActiveDeviceId(reportedDeviceId ?? deviceId ?? null);
      void refreshDevices();
      setStatus('live');
      setSourceEpoch((epoch) => epoch + 1);
    },
    [handleTrackEnded, refreshDevices]
  );

  const stop = useCallback(() => {
    opSeqRef.current += 1;
    const stream = streamRef.current;
    streamRef.current = null;
    if (stream) {
      for (const track of stream.getVideoTracks()) track.removeEventListener('ended', handleTrackEnded);
      stopAllTracks(stream);
    }
    const video = videoRef.current;
    if (video) video.srcObject = null;
    setStatus('idle');
    setMessage(null);
    setPlayBlocked(false);
    setSourceEpoch((epoch) => epoch + 1);
  }, [handleTrackEnded]);

  const selectDevice = useCallback(
    (deviceId: string) => {
      void start(deviceId);
    },
    [start]
  );

  const resumePlayback = useCallback(() => {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream) return;
    // Same fence as start(): this resume belongs to one operation on one owned stream. A late result
    // after a stop or a device switch must never report a playback state for a session that is gone
    // (clearing a fresh session's block, or re-raising a block on a stopped one).
    const op = opSeqRef.current;
    const stillCurrent = () => op === opSeqRef.current && streamRef.current === stream;
    void video
      .play()
      .then(() => {
        if (stillCurrent()) setPlayBlocked(false);
      })
      .catch(() => {
        if (stillCurrent()) setPlayBlocked(true);
      });
  }, []);

  useEffect(() => {
    // Playback state can also change without our affordance (a later autoplay attempt, engine controls,
    // the OS suspending the tab), so the pending flag follows the element itself instead of only our own
    // play() calls. It only ever describes an owned stream.
    if (status !== 'live') return;
    const video = videoRef.current;
    if (!video) return;
    const onPlaying = () => setPlayBlocked(false);
    const onPause = () => {
      if (streamRef.current) setPlayBlocked(true);
    };
    video.addEventListener('playing', onPlaying);
    video.addEventListener('pause', onPause);
    return () => {
      video.removeEventListener('playing', onPlaying);
      video.removeEventListener('pause', onPause);
    };
  }, [status]);

  useEffect(() => {
    // Unmount (and mode switch away from the stage) must release the device: a live indicator light
    // on a torn-down page is a privacy bug, not a cosmetic one.
    return () => {
      opSeqRef.current += 1;
      stopAllTracks(streamRef.current);
      streamRef.current = null;
      const video = videoRef.current;
      if (video) video.srcObject = null;
    };
  }, []);

  return {
    videoRef,
    status,
    message,
    devices,
    activeDeviceId,
    sourceEpoch,
    playBlocked,
    start,
    stop,
    selectDevice,
    resumePlayback,
  };
}
