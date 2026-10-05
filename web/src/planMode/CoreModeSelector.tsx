import { CORE_MODES, CORE_MODE_OPTION, coreModeNote, type CoreMode } from './coreChoice';

export interface CoreModeSelectorProps {
  /** The chosen core; frozen into the next run Start creates and sent as `core_mode` on its plan request. */
  value: CoreMode;
  /** True while a guide run is planning or running: the choice is frozen for that run. */
  disabled: boolean;
  onChange: (mode: CoreMode) => void;
}

/**
 * The core picker: one native radio group in a labelled fieldset, both options always visible, and the note
 * spells out what the chosen core will do. It sits beside the plan model picker and is independent of it —
 * either core runs on either model.
 */
export function CoreModeSelector({ value, disabled, onChange }: CoreModeSelectorProps) {
  return (
    <fieldset className="mode-picker plan-core-picker" data-testid="core-mode-picker" disabled={disabled}>
      <legend className="form-label">코어 로직</legend>
      {CORE_MODES.map((mode) => (
        <label key={mode} data-testid="core-mode-option" data-core-mode={mode}>
          <input
            type="radio"
            name="core-mode"
            value={mode}
            checked={value === mode}
            onChange={() => onChange(mode)}
            aria-describedby="core-mode-note"
          />
          <span>{CORE_MODE_OPTION[mode].label}</span>
        </label>
      ))}
      <p className="plan-core-note" id="core-mode-note" data-testid="core-mode-note" role="status">
        {coreModeNote(value)}
      </p>
    </fieldset>
  );
}
