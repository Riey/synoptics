/**
 * Tracking-frame transport: raw JPEG body (v2) with the fields in the query string, JSON + base64 otherwise;
 * a server that refuses the binary form (415, older build) gets the same frame again as JSON and JSON from
 * then on; a seeded first frame is always JSON.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (/^\.\.?\//.test(specifier) && !/\.[cm]?[jt]sx?$/.test(specifier)) {
      try {
        return nextResolve(`${specifier}.ts`, context);
      } catch {
        // not a .ts file: resolve as written
      }
    }
    return nextResolve(specifier, context);
  },
});

interface Sent {
  url: string;
  contentType: string;
  body: unknown;
}

const sent: Sent[] = [];
let refuseBinary = false;
const ANSWER = { run_id: 'run-a', frame_id: 'f-1', frame_seq: 1 };

const globals = globalThis as unknown as Record<string, unknown>;
const savedFetch = globals.fetch;
globals.fetch = async (url: string, init: { headers: Record<string, string>; body: unknown }) => {
  const contentType = init.headers['Content-Type'];
  sent.push({ url, contentType, body: init.body });
  if (contentType === 'image/jpeg' && refuseBinary) {
    return new Response(JSON.stringify({ error: 'json_required' }), { status: 415 });
  }
  return new Response(JSON.stringify(ANSWER), { status: 200 });
};
test.after(() => {
  globals.fetch = savedFetch;
});

const mode = await import('../coach/captureMode.ts');
let moduleCase = 0;
async function fresh(trackUpload: 'binary' | 'json' = 'binary') {
  sent.length = 0;
  refuseBinary = false;
  mode.setCaptureSettingsForTest({ mode: 'v2', trackScale: 'off', trackUpload, v2Source: 'auto' });
  moduleCase += 1;
  return import(`./trackClient.ts?case=${moduleCase}`);
}

const QUERY = { session_id: '00000000-0000-4000-8000-000000000000', run_id: 'run-a', frame_id: 'f-1', frame_seq: 1 };
const JPEG = new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0x41])], { type: 'image/jpeg' });

test('a raw JPEG goes as the body with the fields in the query string', async () => {
  const client = await fresh();
  assert.equal(client.trackFrameBinaryAvailable(), true);
  assert.deepEqual(await client.trackFrame(QUERY, { jpeg: JPEG }, undefined), ANSWER);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].contentType, 'image/jpeg');
  assert.equal(sent[0].body, JPEG, 'the Blob itself, no copy and no base64');
  const url = new URL(sent[0].url, 'http://x');
  assert.equal(url.pathname, '/api/track/frame');
  assert.deepEqual(Object.fromEntries(url.searchParams), { ...QUERY, frame_seq: '1' });
});

test('a server that refuses the binary form gets the same frame as JSON, and JSON from then on', async () => {
  const client = await fresh();
  refuseBinary = true;
  assert.deepEqual(await client.trackFrame(QUERY, { jpeg: JPEG }, undefined), ANSWER);
  assert.deepEqual(sent.map((request) => request.contentType), ['image/jpeg', 'application/json']);
  const json = JSON.parse(sent[1].body as string);
  assert.deepEqual(json, { ...QUERY, image_base64: '/9j/QQ==' }, 'the same bytes, base64');
  assert.equal(client.trackFrameBinaryAvailable(), false);
  await client.trackFrame({ ...QUERY, frame_seq: 2 }, { jpeg: JPEG }, undefined);
  assert.equal(sent[2].contentType, 'application/json', 'no second binary attempt');
});

test('a seeded frame and a base64 frame always go as JSON; ?trackupload=json opts out', async () => {
  const client = await fresh();
  const seed = { x: 0.1, y: 0.1, width: 0.2, height: 0.2 };
  await client.trackFrame(QUERY, { jpeg: JPEG }, seed);
  await client.trackFrame(QUERY, { base64: 'QUJD' }, undefined);
  assert.deepEqual(sent.map((request) => request.contentType), ['application/json', 'application/json']);
  assert.deepEqual(JSON.parse(sent[0].body as string).seed_box, seed);
  const optedOut = await fresh('json');
  assert.equal(optedOut.trackFrameBinaryAvailable(), false);
});

test('other failures are not treated as a refusal of the binary form', async () => {
  const client = await fresh();
  globals.fetch = async () => new Response(JSON.stringify({ error: 'stale_frame' }), { status: 409 });
  try {
    await assert.rejects(client.trackFrame(QUERY, { jpeg: JPEG }, undefined), (error: { status: number; code: string }) =>
      error.status === 409 && error.code === 'stale_frame'
    );
    assert.equal(client.trackFrameBinaryAvailable(), true);
  } finally {
    globals.fetch = async (url: string, init: { headers: Record<string, string>; body: unknown }) => {
      sent.push({ url, contentType: init.headers['Content-Type'], body: init.body });
      return new Response(JSON.stringify(ANSWER), { status: 200 });
    };
  }
});
