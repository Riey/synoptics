/**
 * When the narration bar's notice (replan, revert, lost target, a talk replan…) goes away: as soon as the step
 * changes, or `NOTICE_TTL_MS` after it appeared on the same step, whichever comes first. A notice describes the
 * moment it was raised; left on screen it contradicts the step the guide shows now (operator screenshot
 * 2026-10-02: "새 1단계부터 안내합니다" still shown at step 2/3). Pure; the hook applies it.
 */
export const NOTICE_TTL_MS = 8000;

/** A shown notice, with the step it was raised on and `performance.now()` when it appeared. */
export interface NoticeStamp {
  text: string;
  stepIndex: number;
  at: number;
}

export function noticeExpired(stamp: NoticeStamp | null, stepIndex: number, nowMs: number): boolean {
  return stamp !== null && (stepIndex !== stamp.stepIndex || nowMs - stamp.at >= NOTICE_TTL_MS);
}
