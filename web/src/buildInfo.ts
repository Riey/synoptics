/**
 * The build line the demo page shows under the server badge: `v0.1.0 · d2376b6 · 2026-10-02 01:30Z`, with
 * `web <commit>` added when the bundle in the browser came from another commit than the server. Pure.
 */
import type { BuildInfo } from './generated/api.generated';

function utcMinute(iso: string): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso;
  return `${at.toISOString().slice(0, 10)} ${at.toISOString().slice(11, 16)}Z`;
}

export function formatBuildLine(build: BuildInfo | null, webCommit: string): string {
  if (!build) return `web ${webCommit}`;
  const parts = [`v${build.version}`, build.commit];
  if (webCommit !== build.commit) parts.push(`web ${webCommit}`);
  parts.push(utcMinute(build.built_at));
  return parts.join(' · ');
}
