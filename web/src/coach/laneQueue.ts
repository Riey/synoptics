/**
 * The job order inside one v2 capture worker (`capture-v2.worker.ts`): one job at a time, kinds in a fixed
 * priority, and for some kinds only the newest job waits.
 *
 * - Jobs run one at a time: they share the worker's scratch canvases, and two encodes on one thread would
 *   only interleave, making both late.
 * - `order` ranks kinds: a queued job of an earlier kind always runs before one of a later kind (FIFO
 *   within a kind). In the guide worker a follow/confirm frame — a request is waiting on it — goes before
 *   an appearance sample.
 * - `latestWins` kinds keep at most one job waiting: a newer job replaces the queued one, which is handed
 *   to `supersede` (the worker answers it as superseded instead of doing stale work). A job already running
 *   is never preempted or superseded.
 *
 * Pure (no worker globals), so `node --test` runs it.
 */
export interface LanePolicy<K extends string> {
  order: readonly K[];
  latestWins: readonly K[];
}

export class LaneQueue<K extends string, J extends { kind: K }> {
  private readonly jobs: J[] = [];
  private running = false;
  private readonly policy: LanePolicy<K>;
  private readonly run: (job: J) => Promise<void>;
  private readonly supersede: (job: J) => void;

  constructor(policy: LanePolicy<K>, run: (job: J) => Promise<void>, supersede: (job: J) => void) {
    this.policy = policy;
    this.run = run;
    this.supersede = supersede;
  }

  /** Jobs waiting (not the one running). */
  get waiting(): number {
    return this.jobs.length;
  }

  get busy(): boolean {
    return this.running;
  }

  push(job: J): void {
    if (this.policy.latestWins.includes(job.kind)) {
      for (let i = this.jobs.length - 1; i >= 0; i -= 1) {
        if (this.jobs[i].kind === job.kind) this.supersede(this.jobs.splice(i, 1)[0]);
      }
    }
    const rank = this.rank(job.kind);
    // After every queued job of the same or an earlier kind, before the first of a later kind.
    const later = this.jobs.findIndex((queued) => this.rank(queued.kind) > rank);
    if (later === -1) this.jobs.push(job);
    else this.jobs.splice(later, 0, job);
    void this.pump();
  }

  private rank(kind: K): number {
    const index = this.policy.order.indexOf(kind);
    return index === -1 ? this.policy.order.length : index;
  }

  private async pump(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      for (let job = this.jobs.shift(); job; job = this.jobs.shift()) {
        // `run` settles every job itself (it replies with an error rather than throwing); a throw here is
        // a bug in the runner and must not wedge the queue for every later job.
        await this.run(job).catch(() => {});
      }
    } finally {
      this.running = false;
    }
  }
}
