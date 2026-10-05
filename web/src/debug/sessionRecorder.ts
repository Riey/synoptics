/**
 * Records the live camera stream in short, independently playable segments for debug bug reports.
 *
 * - One MediaRecorder per segment. A segment closes when the stream is replaced or torn down, when it
 *   reaches `segmentMs`, and on `flush()` — each closed segment is a complete file (concatenating two
 *   recorders' output is not), handed to `onSegmentClosed`.
 * - A closed segment is handed out and forgotten; the receiver owns its memory. A single segment that grows
 *   past `maxBytes` is closed early.
 */

export interface RecorderLike {
  readonly state: string;
  readonly mimeType: string;
  ondataavailable: ((event: { data: Blob }) => void) | null;
  onstop: (() => void) | null;
  onerror: (() => void) | null;
  start(timeslice?: number): void;
  stop(): void;
}

export interface SessionRecorderOptions {
  create: (stream: MediaStream) => RecorderLike;
  /** Fallback when the recorder reports no mimeType. */
  mime: string;
  segmentMs: number;
  maxBytes: number;
  now?: () => number;
  /** How long `flush()` waits for a stopping recorder's last chunk. */
  stopTimeoutMs?: number;
}

export interface RecordedSegment {
  index: number;
  /** Epoch ms. */
  startedAt: number;
  endedAt: number;
  mime: string;
  bytes: number;
  blob: Blob;
}

export interface RecorderStats {
  recording: boolean;
  /** Segments started so far. */
  /** Bytes held by the running segment. */
  bytes: number;
  segments: number;
  failures: number;
}

interface Segment {
  index: number;
  startedAt: number;
  endedAt: number | null;
  mime: string;
  chunks: Blob[];
  bytes: number;
}

interface Open {
  segment: Segment;
  recorder: RecorderLike;
  stopped: Promise<void>;
  markStopped: () => void;
  timer: ReturnType<typeof setTimeout>;
}

const MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4;codecs=avc1',
  'video/mp4',
];

export function pickRecorderMime(isSupported: (mime: string) => boolean): string | null {
  return MIME_CANDIDATES.find((mime) => isSupported(mime)) ?? null;
}

export class SessionRecorder {
  onSegmentClosed: ((segment: RecordedSegment) => void) | null = null;
  private current: Open | null = null;
  private stream: MediaStream | null = null;
  private nextIndex = 0;
  private failures = 0;
  private readonly options: SessionRecorderOptions;
  private readonly now: () => number;

  constructor(options: SessionRecorderOptions) {
    this.options = options;
    this.now = options.now ?? Date.now;
  }

  attach(stream: MediaStream): void {
    if (this.stream === stream && this.current) return;
    void this.close();
    this.stream = stream;
    this.open();
  }

  /** Stops recording; the running segment still closes and is handed out. Resolves once it has. */
  detach(): Promise<void> {
    this.stream = null;
    return this.settle(this.close());
  }

  /** Closes the running segment now (recording continues in a fresh one). Resolves once it is handed out. */
  flush(): Promise<void> {
    if (!this.current) return Promise.resolve();
    const closing = this.close();
    this.open();
    return this.settle(closing);
  }

  stats(): RecorderStats {
    return {
      recording: this.current !== null,
      bytes: this.current?.segment.bytes ?? 0,
      segments: this.nextIndex,
      failures: this.failures,
    };
  }

  private settle(closing: Promise<void>): Promise<void> {
    return Promise.race([closing, wait(this.options.stopTimeoutMs ?? 3000)]);
  }

  private open(): void {
    if (!this.stream) return;
    let recorder: RecorderLike;
    try {
      recorder = this.options.create(this.stream);
    } catch {
      this.failures += 1;
      return;
    }
    const segment: Segment = {
      index: this.nextIndex++,
      startedAt: this.now(),
      endedAt: null,
      mime: recorder.mimeType || this.options.mime,
      chunks: [],
      bytes: 0,
    };
    let markStopped = () => {};
    const stopped = new Promise<void>((resolve) => {
      markStopped = () => {
        if (segment.endedAt !== null) return;
        segment.endedAt = this.now();
        if (segment.bytes > 0) {
          this.onSegmentClosed?.({
            index: segment.index,
            startedAt: segment.startedAt,
            endedAt: segment.endedAt,
            mime: segment.mime,
            bytes: segment.bytes,
            blob: new Blob(segment.chunks, { type: segment.mime }),
          });
        }
        resolve();
      };
    });
    recorder.ondataavailable = (event) => {
      if (!event.data || event.data.size === 0) return;
      segment.chunks.push(event.data);
      segment.bytes += event.data.size;
      if (segment.bytes > this.options.maxBytes && this.current?.segment === segment) void this.flush();
    };
    recorder.onstop = markStopped;
    recorder.onerror = () => {
      this.failures += 1;
      markStopped();
    };
    try {
      recorder.start(1000);
    } catch {
      this.failures += 1;
      return;
    }
    const timer = setTimeout(() => {
      if (this.current?.segment !== segment) return;
      void this.close();
      this.open();
    }, this.options.segmentMs);
    this.current = { segment, recorder, stopped, markStopped, timer };
  }

  private close(): Promise<void> {
    const open = this.current;
    if (!open) return Promise.resolve();
    this.current = null;
    clearTimeout(open.timer);
    try {
      if (open.recorder.state !== 'inactive') open.recorder.stop();
      else open.markStopped();
    } catch {
      open.markStopped();
    }
    return open.stopped;
  }
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
