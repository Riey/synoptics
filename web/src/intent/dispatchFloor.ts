/**
 * The client-side floor between two paid guide calls (a DeepSeek follow, a confirm), measured from the
 * moment the previous POST actually left the browser.
 *
 * Moved here from the removed live guidance lane's `camera/livePolicy.ts` (2026-10-01) with its value
 * unchanged. 2.2 s sits above the server's per-session pacing with a margin for the capture/encode and
 * session work between dispatch and arrival, so a well-behaved client is never answered 429.
 */
export const LIVE_MIN_DISPATCH_INTERVAL_MS = 2200;
