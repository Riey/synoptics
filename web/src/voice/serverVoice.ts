/**
 * Transport for the server voice: `/api/tts/config` (is it on?) and `POST /api/tts` (one line -> WAV). Same-origin
 * cookie session, like the guide routes. Fetched audio is kept in a small cache by sentence, because the guide
 * repeats its lines (notices, completion) and a repeat should not wait for the network.
 */
import { SpeechLoadError } from './speechQueue';

export interface TtsConfig {
  enabled: boolean;
  ready: boolean;
}

/** Longer than a GPU line (well under a second); a line that takes this long is stale anyway. */
export const TTS_FETCH_TIMEOUT_MS = 12_000;
export const TTS_MAX_CHARS = 300;
const CACHE_ENTRIES = 32;
/**
 * A line asked for before the page has a session (the "계획을 세우는 중" line is queued the moment "가이드 시작"
 * is pressed, while the session request is still out) gets 401. It is asked again a few times instead of
 * going straight to the browser voice, so the first line of a page is also read in the server voice.
 */
export const NO_SESSION_RETRY = { tries: 5, delayMs: 300 };

export async function fetchTtsConfig(fetcher: typeof fetch = fetch): Promise<TtsConfig> {
  try {
    const response = await fetcher('/api/tts/config', { method: 'GET', credentials: 'same-origin' });
    if (!response.ok) return { enabled: false, ready: false };
    const body = await response.json() as Partial<TtsConfig>;
    return { enabled: body.enabled === true, ready: body.ready === true };
  } catch {
    return { enabled: false, ready: false };
  }
}

export class ServerVoice {
  private cache = new Map<string, Blob>();

  // The default must not be the bare `fetch`: called as `this.fetcher(...)` it runs with `this` = this object, and
  // browsers reject that ("Illegal invocation") — every line then failed as "network" and fell back to speechSynthesis.
  constructor(
    private readonly fetcher: typeof fetch = (input, init) => fetch(input, init),
    private readonly timeoutMs = TTS_FETCH_TIMEOUT_MS,
    private readonly noSessionRetry = NO_SESSION_RETRY,
  ) {}

  /** The WAV for `text`, from the cache or the server. Rejects with `SpeechLoadError`. */
  async load(text: string, signal: AbortSignal): Promise<Blob> {
    const cached = this.cache.get(text);
    if (cached) {
      this.cache.delete(text);
      this.cache.set(text, cached);
      return cached;
    }
    if (text.length > TTS_MAX_CHARS) throw new SpeechLoadError('too_long', false);
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(abort, this.timeoutMs);
    try {
      let response: Response;
      for (let attempt = 1; ; attempt += 1) {
        response = await this.fetcher('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
          credentials: 'same-origin',
          signal: controller.signal,
        });
        if (response.status !== 401 || attempt >= this.noSessionRetry.tries) break;
        await abortableDelay(this.noSessionRetry.delayMs, controller.signal);
      }
      if (!response.ok) {
        // No session yet (401) or a line the service refuses (422): the service itself is fine.
        throw new SpeechLoadError(`http_${response.status}`, response.status !== 401 && response.status !== 422);
      }
      const blob = await response.blob();
      this.cache.set(text, blob);
      while (this.cache.size > CACHE_ENTRIES) this.cache.delete(this.cache.keys().next().value as string);
      return blob;
    } catch (error) {
      if (error instanceof SpeechLoadError) throw error;
      // A cancelled line (a newer step, the microphone) is not a service failure; a timeout or a dead network is.
      throw new SpeechLoadError(signal.aborted ? 'aborted' : 'network', !signal.aborted);
    } finally {
      clearTimeout(timer);
      signal.removeEventListener('abort', abort);
    }
  }
}

function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException('aborted', 'AbortError'));
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('aborted', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}
