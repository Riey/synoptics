/**
 * The "버그 리포트 전송" checkbox: while it is checked (remembered in localStorage) and the camera is live,
 * the camera is recorded in short segments and each one is uploaded into this page's report as soon as it
 * closes. Unchecking closes the report; leaving the page marks it done with what already arrived.
 *
 * The timeline notes guide events with wall-clock times so the video can be lined up with them.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import { AutoReporter, type ReporterStatus } from './autoReporter';
import { bugReportApi, bugReportEnabled } from './bugReport';
import { imageFileTag, setGuideTap, stripImages, type GuideCallEvent } from './guideTap';
import { SessionRecorder, pickRecorderMime, type RecorderLike } from './sessionRecorder';

const STORAGE_KEY = 'synoptics.bugReport.send';
const SEGMENT_MS = 30 * 1000;
const MAX_SEGMENT_BYTES = 40 * 1024 * 1024;
const MAX_PENDING_BYTES = 200 * 1024 * 1024;
const VIDEO_BITS_PER_SECOND = 1_500_000;
const MAX_EVENTS = 2000;

export interface TimelineEvent {
  at: string;
  kind: string;
  detail?: unknown;
}

export interface BugReporter {
  /** The server has a report directory configured. */
  available: boolean;
  /** Null when this browser has no usable MediaRecorder. */
  mime: string | null;
  checked: boolean;
  setChecked: (checked: boolean) => void;
  recording: boolean;
  status: ReporterStatus | null;
  mark: (kind: string, detail?: unknown) => void;
  /** Sends a comment with the recording up to this moment. */
  comment: (text: string) => Promise<void>;
}

function loadChecked(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function saveChecked(checked: boolean): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, checked ? '1' : '0');
  } catch {
    // storage blocked: the choice lasts for this page only
  }
}

export function useBugReporter(options: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  live: boolean;
  sourceEpoch: number;
  snapshot: () => Record<string, unknown>;
}): BugReporter {
  const { videoRef, live, sourceEpoch } = options;
  const [available, setAvailable] = useState(false);
  const [checked, setCheckedState] = useState(loadChecked);
  const [status, setStatus] = useState<ReporterStatus | null>(null);
  const [recording, setRecording] = useState(false);
  const events = useRef<TimelineEvent[]>([]);
  const snapshotRef = useRef(options.snapshot);
  snapshotRef.current = options.snapshot;

  const parts = useRef<{ recorder: SessionRecorder; reporter: AutoReporter; mime: string | null } | null>(null);
  if (parts.current === null) {
    const mime =
      typeof MediaRecorder === 'undefined' ? null : pickRecorderMime((candidate) => MediaRecorder.isTypeSupported(candidate));
    const recorder = new SessionRecorder({
      // BlobEvent is wider than RecorderLike's `{ data }`; the recorder only reads `data`.
      create: (stream) =>
        new MediaRecorder(stream, { mimeType: mime ?? undefined, videoBitsPerSecond: VIDEO_BITS_PER_SECOND }) as unknown as RecorderLike,
      mime: mime ?? 'video/webm',
      segmentMs: SEGMENT_MS,
      maxBytes: MAX_SEGMENT_BYTES,
    });
    const reporter = new AutoReporter({
      api: bugReportApi,
      maxPendingBytes: MAX_PENDING_BYTES,
      snapshot: () => ({
        ...snapshotRef.current(),
        user_agent: navigator.userAgent,
        recorder: { mime, stats: recorder.stats() },
        timeline: events.current,
        updated_at: new Date().toISOString(),
      }),
      onStatus: setStatus,
    });
    recorder.onSegmentClosed = (segment) => reporter.push(segment);
    parts.current = { recorder, reporter, mime };
  }
  const { recorder, reporter, mime } = parts.current;

  const lastDetail = useRef(new Map<string, string>());
  const mark = useCallback((kind: string, detail?: unknown) => {
    // Re-renders hand over equal state under new object identities: only a change is an event.
    const key = JSON.stringify(detail ?? null);
    if (kind !== 'comment' && lastDetail.current.get(kind) === key) return;
    lastDetail.current.set(kind, key);
    const list = events.current;
    list.push({ at: new Date().toISOString(), kind, detail });
    if (list.length > MAX_EVENTS) list.splice(0, list.length - MAX_EVENTS);
  }, []);

  const setChecked = useCallback((next: boolean) => {
    saveChecked(next);
    setCheckedState(next);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void bugReportEnabled().then((enabled) => !cancelled && setAvailable(enabled));
    return () => {
      cancelled = true;
    };
  }, []);

  const active = available && checked && mime !== null;

  // While sending, every guide call (request without its images, the model's answer or error) goes into
  // report.json and the frames it sent (scene, before_scene) go up as JPEG files named by call number.
  const callSeq = useRef(0);
  useEffect(() => {
    if (!active) return;
    setGuideTap((event: GuideCallEvent) => {
      const seq = ++callSeq.current;
      const stage = event.path.split('/').pop() ?? 'call';
      const { request, images } = stripImages(event.payload);
      const files = images.map((image) => ({
        name: `call-${String(seq).padStart(4, '0')}-${stage}-${imageFileTag(image.field)}.${image.mime === 'image/png' ? 'png' : 'jpg'}`,
        blob: base64Blob(image.base64, image.mime),
      }));
      reporter.logCall(
        {
          seq,
          stage,
          started_at: new Date(event.startedAt).toISOString(),
          ms: event.endedAt - event.startedAt,
          frames: files.map((file) => file.name),
          request,
          ...(event.error === undefined ? { response: event.response } : { error: event.error }),
        },
        files
      );
    });
    return () => setGuideTap(null);
  }, [active, reporter]);
  useEffect(() => {
    const source = videoRef.current?.srcObject;
    if (active && live && source instanceof MediaStream) {
      recorder.attach(source);
      setRecording(true);
      mark('camera', { live: true, sourceEpoch });
      return;
    }
    setRecording(false);
    mark('camera', { live, sourceEpoch, recording: false });
    void recorder.detach().then(() => {
      if (!active) return reporter.finish();
    });
  }, [active, live, sourceEpoch, videoRef, recorder, reporter, mark]);

  useEffect(() => {
    // Hidden (app switch, lock screen): ship what is recorded so far. Leaving: close the report.
    const onVisibility = () => {
      if (document.visibilityState === 'hidden') void recorder.flush();
    };
    const onPageHide = () => reporter.abandon();
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('pagehide', onPageHide);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('pagehide', onPageHide);
    };
  }, [recorder, reporter]);

  const comment = useCallback(
    async (text: string) => {
      mark('comment', text);
      // Ship the segment that holds this moment first, so the comment's video is already on the server.
      await recorder.flush();
      await reporter.comment(text);
    },
    [mark, recorder, reporter]
  );

  return { available, mime, checked, setChecked, recording, status, mark, comment };
}

function base64Blob(base64: string, mime: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Blob([bytes], { type: mime });
}
