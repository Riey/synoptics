/**
 * The primary live surface: a continuously playing camera video with the tracking box and the anchored
 * guide overlay drawn on exactly the video's content rectangle.
 *
 * Invariants this component maintains:
 * - the `<video>` element always exists in camera mode, so a stream can be attached without a race;
 * - the overlay box tracks the letterboxed content rect (not the media box) through ResizeObserver and
 *   the video's own `loadedmetadata`/`resize` events, so intrinsic size, orientation and box resizes all
 *   keep coordinates honest;
 * - the video is never frozen, replaced by a still image, or mirrored: capture happens elsewhere and
 *   reads the element while it keeps playing;
 * - the live track box (when a `liveTrack` store is given) is a sibling svg in the same content-rect
 *   overlay, updated outside React per frame; it is never mirrored (its coordinates are already in the
 *   mirrored upload's space, which is what the user sees).
 */

import React, { useEffect, useRef } from 'react';
import { computeContentRect } from '../camera/geometry';
import type { LiveTrackStore } from '../tracking/liveTrackStore';
import type { GuideOverlayStore } from '../intent/guideOverlayStore';
import { AnchoredGuidanceOverlay } from './AnchoredGuidanceOverlay';
import { LiveTrackChip, LiveTrackOverlay, type LiveTrackPhase } from './LiveTrackOverlay';

interface CameraStageProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: string;
  statusText: string;
  /** Reports the stream's intrinsic size whenever it changes (the host fences tracking on it). */
  onIntrinsicSize: (size: { width: number; height: number }) => void;
  /** True when the preview is mirrored left/right; only the video element is flipped, never the overlay. */
  mirrored: boolean;
  onResumePlayback: () => void;
  playBlocked: boolean;
  /** Coverage shown over the video while no live stream is running; null when live. */
  placeholder: { title: string; detail: string } | null;
  /** The tracking run's per-frame live store; when present the live box and status chip are shown. */
  liveTrack?: LiveTrackStore | null;
  /** The tracking hook's phase, used by the status chip before any tracker answer is accepted. */
  liveTrackPhase?: LiveTrackPhase;
  /** The guide lane's anchored commands (focus ring, label, verdict warning), drawn on the live box. */
  guideOverlay?: GuideOverlayStore | null;
  /** Actual action/label visibility, including placement and freshness; used for the text fallback. */
  onInstructionVisibility?: (visible: boolean) => void;
  /**
   * False hides the live box and anchored drawings. Text fallback and tracking status remain available.
   */
  visualsEnabled?: boolean;
  /**
   * Compact status and accessible fallback at the top of the content rect; labels are placed around it.
   */
  narration?: React.ReactNode;
}

export const CameraStage: React.FC<CameraStageProps> = ({
  videoRef,
  status,
  statusText,
  onIntrinsicSize,
  mirrored,
  onResumePlayback,
  playBlocked,
  placeholder,
  liveTrack = null,
  liveTrackPhase = 'idle',
  guideOverlay = null,
  onInstructionVisibility,
  visualsEnabled = true,
  narration = null,
}) => {
  const mediaRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const publishedRef = useRef<{ width: number; height: number } | null>(null);
  const onIntrinsicSizeRef = useRef(onIntrinsicSize);
  onIntrinsicSizeRef.current = onIntrinsicSize;

  // The autoplay-critical attributes are stamped on the element itself: React sets `muted` only as an
  // IDL property, while Safari's autoplay decision reads the attributes, and this must hold before the
  // first start attempt (not only inside the start path).
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = true;
    video.defaultMuted = true;
    video.autoplay = true;
    if (!video.hasAttribute('muted')) video.setAttribute('muted', '');
    if (!video.hasAttribute('playsinline')) video.setAttribute('playsinline', '');
  }, [videoRef]);

  useEffect(() => {
    const media = mediaRef.current;
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!media || !overlay || !video) return undefined;

    const sync = () => {
      const width = video.videoWidth;
      const height = video.videoHeight;
      // The reference box is the video element itself: it is the scene's CSS box (object-fit letterboxes
      // inside it), so the overlay lands on the pixels the video actually paints. Measuring the media
      // wrapper instead would include its border and oversize the overlay by that border on every side.
      const box = video.getBoundingClientRect();
      const rect = computeContentRect({ width: box.width, height: box.height }, { width, height });
      if (rect) {
        overlay.style.left = `${rect.left}px`;
        overlay.style.top = `${rect.top}px`;
        overlay.style.width = `${rect.width}px`;
        overlay.style.height = `${rect.height}px`;
      }
      overlay.dataset.overlayReady = rect ? 'true' : 'false';

      const published = publishedRef.current;
      if (width > 0 && height > 0 && (published?.width !== width || published?.height !== height)) {
        publishedRef.current = { width, height };
        onIntrinsicSizeRef.current({ width, height });
      }
    };

    sync();
    video.addEventListener('loadedmetadata', sync);
    video.addEventListener('loadeddata', sync);
    video.addEventListener('resize', sync);
    video.addEventListener('emptied', sync);
    // Window resize/orientation are always observed too: Safari reports media dimensions later than
    // Chrome does, and older Safari has no ResizeObserver at all.
    window.addEventListener('resize', sync);
    window.addEventListener('orientationchange', sync);
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(sync) : null;
    observer?.observe(media);
    return () => {
      observer?.disconnect();
      video.removeEventListener('loadedmetadata', sync);
      video.removeEventListener('loadeddata', sync);
      video.removeEventListener('resize', sync);
      video.removeEventListener('emptied', sync);
      window.removeEventListener('resize', sync);
      window.removeEventListener('orientationchange', sync);
    };
  }, [videoRef]);

  return (
    <div
      className="camera-stage"
      role="region"
      aria-label="실시간 카메라 화면"
      data-testid="camera-stage"
      data-mirror={mirrored ? 'on' : 'off'}
      data-playback-pending={playBlocked ? 'true' : 'false'}
    >
      <div className="camera-stage-media" ref={mediaRef} data-testid="camera-stage-media">
        <video
          ref={videoRef}
          className="camera-stage-video"
          data-testid="camera-video"
          data-mirror={mirrored ? 'on' : 'off'}
          playsInline
          muted
          autoPlay
          onClick={onResumePlayback}
        />
        <div
          ref={overlayRef}
          className="camera-stage-overlay"
          data-testid="camera-overlay"
          data-overlay-ready="false"
        >
          {liveTrack && visualsEnabled && <LiveTrackOverlay store={liveTrack} />}
          {liveTrack && guideOverlay && (
            <AnchoredGuidanceOverlay liveTrack={liveTrack} guide={guideOverlay} visible={visualsEnabled} onInstructionVisibility={onInstructionVisibility} />
          )}
          {liveTrack && <LiveTrackChip store={liveTrack} phase={liveTrackPhase} />}
          {narration && (
            <div className="camera-stage-top-stack" data-testid="camera-top-stack">
              {narration}
            </div>
          )}
        </div>
        <div
          className={`camera-stage-status${playBlocked ? ' camera-stage-status-pending' : ''}`}
          data-testid="camera-status"
          data-camera-status={status}
          data-playback-pending={playBlocked ? 'true' : 'false'}
        >
          <span className="status-dot" aria-hidden="true" />
          <span>{statusText}</span>
        </div>
        {placeholder && (
          <div className="camera-stage-placeholder">
            <strong>{placeholder.title}</strong>
            <span className="text-secondary">{placeholder.detail}</span>
          </div>
        )}
        {playBlocked && (
          <div
            className="camera-stage-placeholder camera-stage-tap"
            data-testid="camera-playback-pending"
            role="status"
            aria-live="polite"
          >
            <strong>재생이 멈춰 있습니다</strong>
            <span className="text-secondary">
              브라우저가 자동 재생을 막았거나 재생이 멈췄습니다. 재생을 시작하면 실시간 영상이 다시 흐르고, 그때 현재 화면을 분석할 수 있습니다.
            </span>
            <button type="button" className="btn btn-primary" data-testid="camera-resume" onClick={onResumePlayback}>
              재생 시작
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
