/**
 * The build line under the server badge: server version, server commit, the web bundle's commit when it differs,
 * and the build time in UTC.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { formatBuildLine } from './buildInfo.ts';

const build = { version: '0.1.0', commit: 'd2376b6', built_at: '2026-10-02T01:30:00Z' };

test('server and web from the same commit: one commit is shown', () => {
  assert.equal(formatBuildLine(build, 'd2376b6'), 'v0.1.0 · d2376b6 · 2026-10-02 01:30Z');
});

test('a web bundle from another commit is named next to the server one', () => {
  assert.equal(formatBuildLine(build, 'abc1234'), 'v0.1.0 · d2376b6 · web abc1234 · 2026-10-02 01:30Z');
});

test('an offset time is shown in UTC; an unparsable one as sent; no server build: the web commit only', () => {
  assert.equal(formatBuildLine({ ...build, built_at: '2026-10-02T10:30:00+09:00' }, 'd2376b6'), 'v0.1.0 · d2376b6 · 2026-10-02 01:30Z');
  assert.equal(formatBuildLine({ ...build, built_at: 'soon' }, 'd2376b6'), 'v0.1.0 · d2376b6 · soon');
  assert.equal(formatBuildLine(null, 'abc1234'), 'web abc1234');
});
