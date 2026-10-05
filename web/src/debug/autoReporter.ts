/**
 * Streams closed recording segments into one server-side bug report, in order, as they arrive.
 *
 * The report is created on the first upload; after each upload pass its `report.json` is rewritten with a
 * fresh snapshot (guide state, timeline, guide calls with their answers, uploaded files), so a tab that dies
 * mid-session still leaves a usable report. Guide calls' frames go up as files beside the video. A failed upload stays queued and is retried; if the queue outgrows `maxPendingBytes`
 * the oldest pending segments are dropped (and counted).
 */
import type { RecordedSegment } from './sessionRecorder';

export function extensionForMime(mime: string): 'webm' | 'mp4' {
  return mime.includes('mp4') ? 'mp4' : 'webm';
}

export interface ReportApi {
  create(meta: Record<string, unknown>): Promise<string>;
  putSegment(reportId: string, position: number, segment: RecordedSegment): Promise<void>;
  putFile(reportId: string, name: string, blob: Blob): Promise<void>;
  putMeta(reportId: string, meta: Record<string, unknown>): Promise<void>;
  finish(reportId: string, keepalive: boolean): Promise<void>;
}

export interface UploadedSegment {
  file: string;
  started_at: string;
  ended_at: string;
  duration_ms: number;
  mime: string;
  bytes: number;
}

export interface ReporterStatus {
  reportId: string | null;
  uploaded: number;
  bytes: number;
  pending: number;
  dropped: number;
  comments: number;
  calls: number;
  error: string | null;
}

export interface ReportComment {
  at: string;
  text: string;
}

/** Most recent guide calls kept in `report.json` (their frames stay as files either way). */
export const MAX_LOGGED_CALLS = 500;

type QueueItem =
  | { kind: 'segment'; segment: RecordedSegment; bytes: number }
  | { kind: 'file'; name: string; blob: Blob; bytes: number };

export class AutoReporter {
  private readonly api: ReportApi;
  private readonly snapshot: () => Record<string, unknown>;
  private readonly onStatus: (status: ReporterStatus) => void;
  private readonly maxPendingBytes: number;
  private readonly retryMs: number;
  private queue: QueueItem[] = [];
  private calls: Record<string, unknown>[] = [];
  private uploaded: UploadedSegment[] = [];
  private comments: ReportComment[] = [];
  private reportId: string | null = null;
  private bytes = 0;
  private dropped = 0;
  private error: string | null = null;
  private running: Promise<void> | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;

  constructor(options: {
    api: ReportApi;
    snapshot: () => Record<string, unknown>;
    onStatus?: (status: ReporterStatus) => void;
    maxPendingBytes: number;
    retryMs?: number;
  }) {
    this.api = options.api;
    this.snapshot = options.snapshot;
    this.onStatus = options.onStatus ?? (() => {});
    this.maxPendingBytes = options.maxPendingBytes;
    this.retryMs = options.retryMs ?? 5000;
  }

  status(): ReporterStatus {
    return {
      reportId: this.reportId,
      uploaded: this.uploaded.length,
      bytes: this.bytes,
      pending: this.queue.length,
      calls: this.calls.length,
      dropped: this.dropped,
      comments: this.comments.length,
      error: this.error,
    };
  }

  push(segment: RecordedSegment): void {
    this.enqueue({ kind: 'segment', segment, bytes: segment.bytes });
  }

  /** One guide call (request without images, answer) for `report.json`, with its frames as files. */
  logCall(call: Record<string, unknown>, files: { name: string; blob: Blob }[]): void {
    this.calls.push(call);
    if (this.calls.length > MAX_LOGGED_CALLS) this.calls.splice(0, this.calls.length - MAX_LOGGED_CALLS);
    for (const file of files) this.queue.push({ kind: 'file', name: file.name, blob: file.blob, bytes: file.blob.size });
    this.enqueue(null);
  }

  private enqueue(item: QueueItem | null): void {
    if (item) this.queue.push(item);
    let pending = this.queue.reduce((sum, queued) => sum + queued.bytes, 0);
    while (pending > this.maxPendingBytes && this.queue.length > 1) {
      pending -= this.queue.shift()!.bytes;
      this.dropped += 1;
    }
    this.emit();
    void this.pump();
  }

  /**
   * Adds a comment to the report and rewrites `report.json` once queued segments are in. With no report
   * yet (camera off), the comment opens one.
   */
  async comment(text: string, at: Date = new Date()): Promise<void> {
    this.comments.push({ at: at.toISOString(), text });
    this.emit();
    await this.pump();
    try {
      this.reportId ??= await this.api.create(this.meta());
      await this.api.putMeta(this.reportId, this.meta());
      this.error = null;
    } catch (error) {
      this.fail(error);
    }
    this.emit();
  }

  /** Uploads what is queued (one attempt), then closes the report. The next segment starts a new one. */
  async finish(): Promise<void> {
    await this.pump();
    const reportId = this.reportId;
    this.reset();
    if (reportId) {
      try {
        await this.api.finish(reportId, false);
      } catch (error) {
        this.fail(error);
      }
    }
    this.emit();
  }

  /** Page is going away: mark the report done with what reached the server, without waiting. */
  abandon(): void {
    const reportId = this.reportId;
    this.reset();
    if (reportId) void this.api.finish(reportId, true).catch(() => {});
  }

  private reset(): void {
    if (this.retry) clearTimeout(this.retry);
    this.retry = null;
    this.reportId = null;
    this.uploaded = [];
    this.comments = [];
    this.calls = [];
    this.queue = [];
    this.bytes = 0;
    this.dropped = 0;
  }

  private meta(): Record<string, unknown> {
    return {
      ...this.snapshot(),
      comments: this.comments,
      calls: this.calls,
      segments: this.uploaded,
      dropped_uploads: this.dropped,
    };
  }

  private pump(): Promise<void> {
    this.running ??= this.drain().finally(() => {
      this.running = null;
    });
    return this.running;
  }

  private async drain(): Promise<void> {
    let sent = false;
    while (this.queue.length > 0) {
      const item = this.queue[0];
      try {
        this.reportId ??= await this.api.create(this.meta());
        if (item.kind === 'segment') {
          const { segment } = item;
          const position = this.uploaded.length;
          await this.api.putSegment(this.reportId, position, segment);
          this.uploaded.push({
            file: `video-${String(position).padStart(4, '0')}.${extensionForMime(segment.mime)}`,
            started_at: new Date(segment.startedAt).toISOString(),
            ended_at: new Date(segment.endedAt).toISOString(),
            duration_ms: segment.endedAt - segment.startedAt,
            mime: segment.mime,
            bytes: segment.bytes,
          });
        } else {
          await this.api.putFile(this.reportId, item.name, item.blob);
        }
        if (this.queue[0] === item) this.queue.shift();
        this.bytes += item.bytes;
        this.error = null;
        sent = true;
      } catch (error) {
        this.fail(error);
        this.retry ??= setTimeout(() => {
          this.retry = null;
          void this.pump();
        }, this.retryMs);
        break;
      } finally {
        this.emit();
      }
    }
    // One report.json rewrite per pass: frames arrive about once a second, the snapshot is not small.
    if (sent && this.reportId) {
      try {
        await this.api.putMeta(this.reportId, this.meta());
      } catch (error) {
        this.fail(error);
      }
      this.emit();
    }
  }

  private fail(error: unknown): void {
    this.error = error instanceof Error ? error.message : String(error);
  }

  private emit(): void {
    this.onStatus(this.status());
  }
}
