/**
 * Voice feedback: reads `speechPolicy.spokenLines` aloud. When the server voice is on (`/api/tts/config`: the app
 * proxies a loopback TTS service with the operator's fine-tuned voice), each line is fetched as audio and played;
 * otherwise, or when a line cannot be fetched, the browser's `speechSynthesis` reads it (on-device voices). The
 * order, cutting and de-duplication rules live in `SpeechQueue`. Off until the user turns it on (the choice is kept
 * in localStorage); silent where neither voice exists. `pause()`/`resume()` let the microphone take the speaker's
 * turn so the recogniser does not transcribe the guide's own voice.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { IntentView } from '../intent/useIntentLoop';
import { fetchTtsConfig, ServerVoice } from './serverVoice';
import { spokenLines } from './speechPolicy';
import { SpeechQueue, type Playback } from './speechQueue';

export const VOICE_OUTPUT_STORAGE_KEY = 'synoptics.voice.output';
export const SPEECH_LANG = 'ko-KR';

export interface SpeechOutput {
  supported: boolean;
  enabled: boolean;
  speaking: boolean;
  /** Which voice reads new lines: the server's (operator voice) or the browser's. */
  voice: 'server' | 'browser';
  setEnabled: (enabled: boolean) => void;
  /** Stop reading and hold further lines (the microphone is open). */
  pause: () => void;
  resume: () => void;
}

/** 10 ms of silence (8 kHz 16-bit mono WAV): played inside a user gesture so later, gesture-less playback is allowed. */
const SILENT_WAV = 'data:audio/wav;base64,UklGRsQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YaAAAA'
  + 'A'.repeat(214);

function loadEnabled(): boolean {
  try {
    return window.localStorage.getItem(VOICE_OUTPUT_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

export function useSpeechOutput(view: IntentView): SpeechOutput {
  const localSupported = typeof window !== 'undefined' && 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
  const audioSupported = typeof window !== 'undefined' && typeof Audio !== 'undefined';
  const [server, setServer] = useState(false);
  const supported = localSupported || server;
  const [enabled, setEnabledState] = useState(() => loadEnabled());
  const [speaking, setSpeaking] = useState(false);
  const previousRef = useRef<IntentView | null>(null);
  const heldRef = useRef(false);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const speakLocal = useCallback((text: string): Playback => {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = SPEECH_LANG;
    if (voiceRef.current) utterance.voice = voiceRef.current;
    const done = new Promise<void>((resolve) => {
      utterance.onend = () => resolve();
      utterance.onerror = () => resolve();
      // Chromium sometimes never fires `end`; the queue must not wait on it forever. If this fires early the next
      // utterance only queues behind this one inside speechSynthesis, so the order still holds.
      window.setTimeout(resolve, 4000 + text.length * 250);
    });
    synth.speak(utterance);
    return { done, stop: () => synth.cancel() };
  }, []);

  const queue = useMemo(() => {
    const voice = new ServerVoice();
    return new SpeechQueue({
      load: async (text, signal) => {
        const blob = await voice.load(text, signal);
        return () => playBlob(audioRef.current ?? (audioRef.current = new Audio()), blob);
      },
      local: localSupported ? speakLocal : null,
      onSpeaking: setSpeaking,
      onServerLost: () => setServer(false),
    }, false);
  }, [localSupported, speakLocal]);

  // The server voice is used only when the app says the service is configured and answered its health check.
  useEffect(() => {
    if (!audioSupported) return undefined;
    let live = true;
    void fetchTtsConfig().then((config) => {
      if (!live) return;
      const on = config.enabled && config.ready;
      queue.setServer(on);
      setServer(on);
    });
    return () => {
      live = false;
    };
  }, [audioSupported, queue]);

  // Browsers allow gesture-less playback on an element once it has played inside a gesture. The toggle is one
  // such gesture; the first tap anywhere covers a page that starts with speech already on.
  const unlock = useCallback(() => {
    if (!audioSupported) return;
    const audio = audioRef.current ?? (audioRef.current = new Audio());
    if (audio.src && !audio.paused) return;
    audio.src = SILENT_WAV;
    void audio.play().catch(() => undefined);
  }, [audioSupported]);
  useEffect(() => {
    if (!audioSupported) return undefined;
    window.addEventListener('pointerdown', unlock, { once: true, capture: true });
    return () => window.removeEventListener('pointerdown', unlock, { capture: true });
  }, [audioSupported, unlock]);

  const setEnabled = useCallback((next: boolean) => {
    setEnabledState(next);
    try {
      window.localStorage.setItem(VOICE_OUTPUT_STORAGE_KEY, next ? '1' : '0');
    } catch {
      // Private mode: the choice lasts for this page only.
    }
    if (next) unlock();
    else queue.cancel();
  }, [queue, unlock]);

  // Prefer a Korean voice when the browser lists one; voices load asynchronously in Chromium.
  useEffect(() => {
    if (!localSupported) return undefined;
    const pick = () => {
      const voices = window.speechSynthesis.getVoices();
      voiceRef.current = voices.find((voice) => voice.lang.replace('_', '-').toLowerCase() === SPEECH_LANG.toLowerCase())
        ?? voices.find((voice) => voice.lang.toLowerCase().startsWith('ko')) ?? null;
    };
    pick();
    window.speechSynthesis.addEventListener('voiceschanged', pick);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', pick);
  }, [localSupported]);

  // Every view transition is diffed, even while off, so turning speech on mid-run does not replay history.
  useEffect(() => {
    const lines = spokenLines(previousRef.current, view);
    previousRef.current = view;
    if (!supported || !enabled || heldRef.current) return;
    for (const line of lines) queue.speak(line.text, line.priority);
  }, [view, supported, enabled, queue]);

  useEffect(() => () => queue.cancel(), [queue]);

  const pause = useCallback(() => {
    heldRef.current = true;
    queue.cancel();
  }, [queue]);
  const resume = useCallback(() => {
    heldRef.current = false;
  }, []);

  return { supported, enabled: supported && enabled, speaking, voice: server ? 'server' : 'browser', setEnabled, pause, resume };
}

/** Play `blob` on the shared element; `done` rejects if the browser refuses to start it (autoplay policy). */
function playBlob(audio: HTMLAudioElement, blob: Blob): Playback {
  const url = URL.createObjectURL(blob);
  let finish: () => void = () => undefined;
  const ended = new Promise<void>((resolve) => {
    finish = () => {
      audio.onended = null;
      audio.onerror = null;
      URL.revokeObjectURL(url);
      resolve();
    };
  });
  audio.onended = finish;
  audio.onerror = finish;
  audio.src = url;
  const done = audio.play().then(() => ended, (error: unknown) => {
    finish();
    throw error;
  });
  return {
    done,
    stop: () => {
      audio.pause();
      finish();
    },
  };
}
