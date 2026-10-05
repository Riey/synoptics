/**
 * The user's words to the running guide (design 2026-10-01 guide-talk §D): one line in, the latest exchange
 * out. Shown only while a guide runs; locked while an answer is awaited; a refusal or failure is one line.
 *
 * Voice: the microphone button transcribes one utterance into the same channel (`useSpeechInput`; hidden
 * where the browser has no recogniser). While the microphone is open the typed input shows the words
 * recognised so far. The spoken feedback toggle lives in `VoiceToggle` (shown in every phase).
 */
import { useEffect, useState, type FormEvent } from 'react';
import type { IntentView } from '../intent/useIntentLoop';
import { TALK_UTTERANCE_MAX } from '../intent/talkPolicy';
import type { SpeechInput } from '../voice/useSpeechInput';
import { formatElapsed } from './NarrationBar';

export function TalkBar({ view, onSend, speechInput }: {
  view: IntentView;
  onSend: (utterance: string) => boolean;
  speechInput: SpeechInput;
}) {
  const [text, setText] = useState('');
  const pendingSince = view.talkPendingSince;
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (pendingSince === null) return undefined;
    setNow(performance.now());
    const timer = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(timer);
  }, [pendingSince]);

  if (view.phase !== 'running') return null;
  const pending = pendingSince !== null;
  const status = view.talkError ?? speechInput.error ?? (pending ? null : view.talkBlockedReason);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (pending || !text.trim()) return;
    if (onSend(text)) setText('');
  };
  const listening = speechInput.listening;

  return (
    <form className="talk-bar" data-testid="talk-bar" data-pending={pending ? 'true' : 'false'} onSubmit={submit}>
      {view.talk && (
        <div className="talk-exchange" aria-live="polite">
          <p className="talk-line" data-testid="talk-last-utterance"><span className="talk-who">나:</span> {view.talk.utterance}</p>
          <p className="talk-line" data-testid="talk-reply"><span className="talk-who">안내:</span> {view.talk.reply}</p>
        </div>
      )}
      <div className="talk-input-row">
        {speechInput.supported && (
          <button
            type="button"
            className={`btn talk-mic${listening ? ' talk-mic-on' : ''}`}
            data-testid="talk-mic"
            aria-pressed={listening}
            aria-label={listening ? '듣는 중 — 누르면 멈춤' : '말로 하기'}
            title={listening ? '듣는 중 — 누르면 멈춤' : '말로 하기 (한 문장)'}
            disabled={pending}
            onClick={() => (listening ? speechInput.stop() : speechInput.start())}
          >
            {listening ? '● 듣는 중' : '🎤 말하기'}
          </button>
        )}
        <label className="visually-hidden" htmlFor="talk-input">안내에게 말하기</label>
        <input
          id="talk-input"
          className="input-text talk-input"
          data-testid="talk-input"
          type="text"
          value={listening ? speechInput.interim : text}
          maxLength={TALK_UTTERANCE_MAX}
          disabled={pending || listening}
          autoComplete="off"
          enterKeyHint="send"
          placeholder={listening ? '말씀하세요…' : '예: 어디를 말하는지 모르겠어 · 더 쉽게 설명해줘 · 이미 했어'}
          onChange={(event) => setText(event.target.value)}
        />
        <button type="submit" className="btn btn-primary talk-send" data-testid="talk-send" disabled={pending || listening || !text.trim()}>
          {pendingSince !== null ? (
            <>답을 정리하는 중 <span data-testid="talk-elapsed">{formatElapsed(now - pendingSince)}</span></>
          ) : (
            '보내기'
          )}
        </button>
      </div>
      {pending && (
        // A question is answered for accuracy, not speed (thinking-mode talk measured 4-9 s, 2026-10-02).
        <p className="talk-status talk-wait" data-testid="talk-wait">
          화면과 계획을 보고 답을 정리하고 있어요. 10초쯤 걸릴 수 있어요.
        </p>
      )}
      {status && (
        <p className="talk-status" data-testid={view.talkError ? 'talk-error' : 'talk-blocked'}>{status}</p>
      )}
    </form>
  );
}
