/**
 * The order in which spoken lines are read, independent of which voice reads them (no DOM, no timers).
 *
 * Each line goes to the server voice (`/api/tts`, the operator's fine-tuned voice) when it is on, and to the
 * browser's `speechSynthesis` otherwise or when the server cannot deliver that line. The rules are the ones the
 * browser-only version had: a `replace` line cuts whatever is being read and drops what was waiting; an `append`
 * line waits its turn; the same sentence already waiting or being read is not queued twice.
 *
 * The server voice needs a round trip per line, so the next line's audio is fetched while the current one plays
 * (one ahead). A line whose fetch fails is read by the browser voice in its place, keeping the order. Failures
 * that say the service is down (not "no session yet") count; two in a row turn the server voice off for the page,
 * so a dead service costs one fallback delay, not one per line.
 */

export interface Playback {
  /** Resolves when the line has been read or stopped; rejects only if it could not start (nothing was heard). */
  done: Promise<void>;
  stop: () => void;
}

/** A failed server line. `countable` is false for a refusal that says nothing about the service (no session). */
export class SpeechLoadError extends Error {
  constructor(message: string, readonly countable: boolean) {
    super(message);
  }
}

export interface SpeechEngines {
  /** Fetch the server audio for `text`; resolves to a function that starts it. */
  load: (text: string, signal: AbortSignal) => Promise<() => Playback>;
  /** The browser voice, or null where the browser cannot speak. */
  local: ((text: string) => Playback) | null;
  onSpeaking: (speaking: boolean) => void;
  /** Called once when repeated failures turn the server voice off. */
  onServerLost?: () => void;
}

export const SERVER_FAILURES_BEFORE_OFF = 2;

interface Item {
  text: string;
  controller: AbortController;
  audio: Promise<(() => Playback) | SpeechLoadError> | null;
}

export class SpeechQueue {
  private queue: Item[] = [];
  private current: { item: Item; playback: Playback } | null = null;
  private generation = 0;
  private pumping = false;
  private failures = 0;

  constructor(private readonly engines: SpeechEngines, private server: boolean) {}

  get serverOn(): boolean {
    return this.server;
  }

  setServer(on: boolean): void {
    this.server = on;
    if (on) this.failures = 0;
  }

  speak(text: string, priority: 'replace' | 'append'): void {
    if (!this.server && !this.engines.local) return;
    if (priority === 'replace') this.cancel();
    else if (this.current?.item.text === text || this.queue.some((item) => item.text === text)) return;
    this.queue.push({ text, controller: new AbortController(), audio: null });
    this.prefetch();
    void this.pump();
  }

  /** Stop the line being read and drop everything waiting (a new step, the microphone opening, speech off). */
  cancel(): void {
    this.generation += 1;
    for (const item of this.queue) item.controller.abort();
    this.queue = [];
    if (this.current) {
      this.current.item.controller.abort();
      this.current.playback.stop();
      this.current = null;
    }
    this.pumping = false;
    this.engines.onSpeaking(false);
  }

  private startLoad(item: Item): void {
    if (item.audio || !this.server) return;
    item.audio = this.engines.load(item.text, item.controller.signal).catch((error: unknown) =>
      error instanceof SpeechLoadError ? error : new SpeechLoadError(String(error), true));
  }

  /** The head of the queue, and one line behind it while something is already playing. */
  private prefetch(): void {
    const ahead = this.current ? 1 : 2;
    for (const item of this.queue.slice(0, ahead)) this.startLoad(item);
  }

  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    const generation = this.generation;
    while (generation === this.generation && this.queue.length > 0) {
      const item = this.queue[0];
      this.startLoad(item);
      const audio = item.audio ? await item.audio : null;
      if (generation !== this.generation) return;
      this.queue.shift();
      let start: (() => Playback) | null = null;
      if (typeof audio === 'function') {
        this.failures = 0;
        start = audio;
      } else if (audio instanceof SpeechLoadError && audio.countable && this.server) {
        this.failures += 1;
        if (this.failures >= SERVER_FAILURES_BEFORE_OFF) {
          this.server = false;
          this.engines.onServerLost?.();
        }
      }
      const local = this.engines.local;
      let played = await this.play(item, start, generation);
      if (!played && local && generation === this.generation) played = await this.play(item, () => local(item.text), generation);
      if (generation !== this.generation) return;
    }
    this.pumping = false;
    this.engines.onSpeaking(false);
  }

  /** Read one line; false when it could not start (the caller then tries the browser voice). */
  private async play(item: Item, start: (() => Playback) | null, generation: number): Promise<boolean> {
    if (!start) return false;
    let playback: Playback;
    try {
      playback = start();
    } catch {
      return false;
    }
    this.current = { item, playback };
    this.engines.onSpeaking(true);
    this.prefetch();
    try {
      await playback.done;
      return true;
    } catch {
      return false;
    } finally {
      if (generation === this.generation) this.current = null;
    }
  }
}
