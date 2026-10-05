/**
 * Voice commands: one utterance per press of the microphone, transcribed by the browser's Web Speech
 * recogniser (`SpeechRecognition`, Safari/Chromium; absent in Firefox, where the hook reports
 * `supported: false` and the button is not shown) and handed to the talk channel as the typed sentence
 * would be. No audio is sent to this app's server; what the browser vendor does with it is the vendor's
 * policy (Safari: on-device when available; Chromium: Google's service).
 *
 * The transcript is not interpreted here: `/api/guide/talk` classifies the words (step_mark, target, say,
 * go_to, replan), so "이미 했어" by voice is exactly "이미 했어" typed.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { SPEECH_LANG } from './useSpeechOutput';

/** The subset of the Web Speech recogniser this hook uses (TypeScript's DOM lib does not declare it). */
interface Recogniser {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
  onresult: ((event: { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

type RecogniserConstructor = new () => Recogniser;

export function recogniserConstructor(): RecogniserConstructor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as { SpeechRecognition?: RecogniserConstructor; webkitSpeechRecognition?: RecogniserConstructor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export const SPEECH_INPUT_ERRORS: Record<string, string> = {
  'not-allowed': '마이크 사용이 허용되지 않았습니다. 브라우저의 마이크 권한을 확인하세요.',
  'service-not-allowed': '이 브라우저에서는 음성 인식을 쓸 수 없습니다.',
  'audio-capture': '마이크를 찾지 못했습니다.',
  network: '음성 인식 서비스에 연결하지 못했습니다.',
  'no-speech': '말소리를 듣지 못했습니다. 다시 눌러 말해 주세요.',
  aborted: '',
};

export interface SpeechInput {
  supported: boolean;
  listening: boolean;
  /** Words recognised so far in the open utterance (shown in the input while listening). */
  interim: string;
  error: string | null;
  start: () => void;
  stop: () => void;
}

export interface SpeechInputOptions {
  /** A finished utterance; called once per press, with the trimmed transcript. */
  onFinal: (transcript: string) => void;
  /** The microphone is open / closed (the speaker yields its turn). */
  onListening?: (listening: boolean) => void;
}

export function useSpeechInput({ onFinal, onListening }: SpeechInputOptions): SpeechInput {
  const Ctor = recogniserConstructor();
  const supported = Ctor !== null;
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState('');
  const [error, setError] = useState<string | null>(null);
  const activeRef = useRef<Recogniser | null>(null);
  const finalRef = useRef('');
  const callbacksRef = useRef({ onFinal, onListening });
  callbacksRef.current = { onFinal, onListening };

  const stop = useCallback(() => {
    activeRef.current?.stop();
  }, []);

  const start = useCallback(() => {
    if (!Ctor || activeRef.current) return;
    const recogniser = new Ctor();
    recogniser.lang = SPEECH_LANG;
    recogniser.interimResults = true;
    recogniser.continuous = false;
    recogniser.maxAlternatives = 1;
    finalRef.current = '';
    recogniser.onresult = (event) => {
      let finalText = '';
      let interimText = '';
      for (let i = 0; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result.isFinal) finalText += result[0].transcript;
        else interimText += result[0].transcript;
      }
      finalRef.current = finalText;
      setInterim((finalText + interimText).trim());
    };
    recogniser.onerror = (event) => {
      const text = SPEECH_INPUT_ERRORS[event.error];
      setError(text === undefined ? `음성 인식 오류: ${event.error}` : text || null);
    };
    recogniser.onend = () => {
      activeRef.current = null;
      setListening(false);
      setInterim('');
      callbacksRef.current.onListening?.(false);
      const transcript = finalRef.current.trim();
      finalRef.current = '';
      if (transcript) callbacksRef.current.onFinal(transcript);
    };
    activeRef.current = recogniser;
    setError(null);
    setInterim('');
    setListening(true);
    callbacksRef.current.onListening?.(true);
    try {
      recogniser.start();
    } catch (exc) {
      activeRef.current = null;
      setListening(false);
      callbacksRef.current.onListening?.(false);
      setError(`음성 인식을 시작하지 못했습니다: ${exc instanceof Error ? exc.message : String(exc)}`);
    }
  }, [Ctor]);

  useEffect(() => () => {
    activeRef.current?.abort();
    activeRef.current = null;
  }, []);

  return { supported, listening, interim, error, start, stop };
}
