/**
 * The code-less notice follows the server's `access_mode`, never the browser's address.
 *
 * Run with `npm run test:web`.
 */
import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const WEB_ROOT = fileURLToPath(new URL('../..', import.meta.url));

const server = await createServer({
  configFile: false,
  root: WEB_ROOT,
  logLevel: 'silent',
  server: { middlewareMode: true, hmr: false, watch: null },
});
after(async () => {
  await server.close();
});

const { OPEN_ACCESS_NOTICE, LOCAL_ACCESS_NOTICE, codeLessAccessNotice, accessModeStatusSuffix } =
  await server.ssrLoadModule('/src/coach/accessNotice.ts');

test('an open-access deployment says the tailnet is why no code is asked', () => {
  assert.equal(codeLessAccessNotice('open'), '테일넷 내부 연결이라 접근 코드 없이 사용합니다.');
  assert.equal(codeLessAccessNotice('open'), OPEN_ACCESS_NOTICE);
  assert.equal(accessModeStatusSuffix(false, 'open'), ' · 테일넷 모드');
});

test('a local-only deployment keeps the local wording', () => {
  assert.equal(codeLessAccessNotice('local'), LOCAL_ACCESS_NOTICE);
  assert.equal(accessModeStatusSuffix(false, 'local'), ' · 로컬 모드');
});

test('an older server without access_mode keeps the local wording', () => {
  assert.equal(codeLessAccessNotice(undefined), LOCAL_ACCESS_NOTICE);
  assert.equal(codeLessAccessNotice(null), LOCAL_ACCESS_NOTICE);
  assert.equal(accessModeStatusSuffix(false, undefined), ' · 로컬 모드');
});

test('a code deployment gets no status suffix', () => {
  assert.equal(accessModeStatusSuffix(true, 'code'), '');
  assert.equal(accessModeStatusSuffix(true, null), '');
});
