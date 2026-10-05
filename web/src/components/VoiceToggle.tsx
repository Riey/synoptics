/**
 * The spoken-feedback switch (`useSpeechOutput`): shown whenever the browser can speak, in every phase, so
 * it can be turned on before a guide starts and the first step is read aloud. Nothing is shown when neither the
 * server voice nor `speechSynthesis` is available. `data-voice` says which one reads (`server` = operator voice).
 */
import type { SpeechOutput } from '../voice/useSpeechOutput';

export function VoiceToggle({ speechOutput }: { speechOutput: SpeechOutput }) {
  if (!speechOutput.supported) return null;
  return (
    <label className="voice-toggle" data-testid="voice-toggle" data-speaking={speechOutput.speaking ? 'true' : 'false'}
      data-voice={speechOutput.voice}>
      <input
        type="checkbox"
        data-testid="voice-toggle-input"
        checked={speechOutput.enabled}
        onChange={(event) => speechOutput.setEnabled(event.target.checked)}
      />
      {' '}안내를 음성으로 읽기{speechOutput.speaking ? ' · 읽는 중' : ''}
    </label>
  );
}
