/**
 * The capture worker's job order: one job at a time, tracking frames first.
 *
 * Every camera capture shares one worker (tracking frames, guide plan/follow/confirm/talk frames, the
 * appearance samples). Jobs run strictly one at a time — they reuse the worker's scratch canvases, and two
 * full-frame encodes competing for the same cores would only make both late. A tracking frame is the one
 * the live box waits on, so it is never queued behind a guide capture that arrived earlier: the queue
 * takes the oldest `track` job first and only then the oldest of the rest. A job already running is not
 * preempted, so a tracking frame waits at most for the one job in progress.
 *
 * Pure (no worker globals), so `node --test` runs it.
 */
export type CapturePriority = 'track' | 'guide';

export class CaptureJobQueue<J extends { priority: CapturePriority }> {
  private readonly jobs: J[] = [];
  private running = false;
  private readonly run: (job: J) => Promise<void>;

  constructor(run: (job: J) => Promise<void>) {
    this.run = run;
  }

  push(job: J): void {
    if (job.priority === 'track') {
      // After every queued tracking job, before every guide job: FIFO within each priority.
      const firstGuide = this.jobs.findIndex((queued) => queued.priority !== 'track');
      if (firstGuide === -1) this.jobs.push(job);
      else this.jobs.splice(firstGuide, 0, job);
    } else {
      this.jobs.push(job);
    }
    void this.pump();
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
