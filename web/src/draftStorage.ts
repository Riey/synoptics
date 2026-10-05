export interface DraftInputs {
  user_goal: string;
  context: string;
}

export const DRAFT_STORAGE_KEY = 'visual-coach:draft-inputs:v1';

export const EMPTY_DRAFT_INPUTS: DraftInputs = Object.freeze({
  user_goal: '',
  context: '',
});

/**
 * Safely loads persisted draft inputs from localStorage.
 * Strictly checks canonical keys (user_goal, context); any other stored key is ignored.
 * Returns EMPTY_DRAFT_INPUTS if storage is unavailable, empty, or corrupted.
 */
export function loadDraftInputs(): DraftInputs {
  if (typeof window === 'undefined') return { ...EMPTY_DRAFT_INPUTS };
  try {
    const raw = window.localStorage.getItem(DRAFT_STORAGE_KEY);
    if (!raw) return { ...EMPTY_DRAFT_INPUTS };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return { ...EMPTY_DRAFT_INPUTS };

    return {
      user_goal: typeof parsed.user_goal === 'string' ? parsed.user_goal : '',
      context: typeof parsed.context === 'string' ? parsed.context : '',
    };
  } catch {
    // SecurityError (cookies/storage blocked), JSON parse error, etc.
    return { ...EMPTY_DRAFT_INPUTS };
  }
}

/**
 * Saves draft inputs to localStorage synchronously.
 * Returns true if saved successfully, false if storage is blocked/exceeded.
 */
export function saveDraftInputs(draft: DraftInputs): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const payload: DraftInputs = {
      user_goal: typeof draft.user_goal === 'string' ? draft.user_goal : '',
      context: typeof draft.context === 'string' ? draft.context : '',
    };
    window.localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(payload));
    return true;
  } catch {
    // QuotaExceededError, SecurityError, etc. Fail closed without crashing.
    return false;
  }
}

export interface CameraView {
  /**
   * Whether the live preview is mirrored left/right. It is a *user choice*, never an inference about
   * the hardware: the app cannot tell whether the operating system already flips the camera. The
   * frames sent to the model are flipped to match the preview exactly, so the model and the human are
   * always looking at the same orientation.
   */
  mirror: boolean;
}

/** Its own key, deliberately separate from the draft inputs: neither migration nor erasure touches them. */
export const CAMERA_VIEW_STORAGE_KEY = 'visual-coach:camera-view:v1';

export const DEFAULT_CAMERA_VIEW: CameraView = Object.freeze({ mirror: false });

/**
 * Safely loads the persisted camera view preference.
 * Returns DEFAULT_CAMERA_VIEW if storage is unavailable, empty, or corrupted.
 */
export function loadCameraView(): CameraView {
  if (typeof window === 'undefined') return { ...DEFAULT_CAMERA_VIEW };
  try {
    const raw = window.localStorage.getItem(CAMERA_VIEW_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_CAMERA_VIEW };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return { ...DEFAULT_CAMERA_VIEW };
    return { mirror: parsed.mirror === true };
  } catch {
    return { ...DEFAULT_CAMERA_VIEW };
  }
}

/** Saves the camera view preference synchronously, failing closed when storage is blocked. */
export function saveCameraView(view: CameraView): boolean {
  if (typeof window === 'undefined') return false;
  try {
    window.localStorage.setItem(CAMERA_VIEW_STORAGE_KEY, JSON.stringify({ mirror: view.mirror === true }));
    return true;
  } catch {
    return false;
  }
}
