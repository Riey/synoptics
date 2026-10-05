/**
 * Spoken-line order with two voices: the server voice (fetched audio) and the browser fallback. Loaded through
 * Vite (runtime sibling import), like the other voice and intent policies. Engines are fakes that record calls.
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
const { SpeechQueue, SpeechLoadError, SERVER_FAILURES_BEFORE_OFF } = await server.ssrLoadModule('/src/voice/speechQueue.ts');
const { ServerVoice, fetchTtsConfig } = await server.ssrLoadModule('/src/voice/serverVoice.ts');

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
async function settle() {
  for (let i = 0; i < 10; i += 1) await tick();
}

interface Deferred<T> { promise: Promise<T>; resolve: (value: T) => void; reject: (error: unknown) => void }
function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

/** Records every line each voice was asked to read; each playback finishes when the test says so. */
function harness({ local = true } = {}) {
  const log: string[] = [];
  const loads = new Map<string, Deferred<unknown>>();
  const playing = new Map<string, Deferred<void>>();
  const speaking: boolean[] = [];
  let lost = 0;
  const playback = (label: string) => {
    log.push(label);
    const done = deferred<void>();
    playing.set(label, done);
    return { done: done.promise, stop: () => { log.push(`stop ${label}`); done.resolve(); } };
  };
  const queue = new SpeechQueue({
    load: (text: string, signal: AbortSignal) => {
      log.push(`load ${text}`);
      const pending = deferred<unknown>();
      loads.set(text, pending);
      signal.addEventListener('abort', () => pending.reject(new SpeechLoadError('aborted', false)));
      return pending.promise.then(() => () => playback(`server ${text}`));
    },
    local: local ? (text: string) => playback(`local ${text}`) : null,
    onSpeaking: (on: boolean) => speaking.push(on),
    onServerLost: () => { lost += 1; },
  }, true);
  return {
    queue, log, speaking, lost: () => lost,
    loaded: async (text: string) => { loads.get(text)!.resolve(undefined); await settle(); },
    failed: async (text: string, countable = true) => { loads.get(text)!.reject(new SpeechLoadError('http_503', countable)); await settle(); },
    finish: async (label: string) => { playing.get(label)!.resolve(); await settle(); },
  };
}

test('server lines play in order, the next one fetched while the current one plays', async () => {
  const h = harness();
  h.queue.speak('하나', 'replace');
  h.queue.speak('둘', 'append');
  h.queue.speak('셋', 'append');
  await settle();
  assert.deepEqual(h.log, ['load 하나', 'load 둘']);
  await h.loaded('하나');
  assert.deepEqual(h.log.slice(2), ['server 하나']);
  await h.loaded('둘');
  assert.equal(h.log.includes('server 둘'), false, 'waits for the line being read');
  await h.finish('server 하나');
  assert.deepEqual(h.log.slice(3), ['server 둘', 'load 셋']);
  await h.loaded('셋');
  await h.finish('server 둘');
  await h.finish('server 셋');
  assert.deepEqual(h.log.slice(5), ['server 셋']);
  assert.equal(h.speaking.at(-1), false);
});

test('a replace line cuts the current line and drops the waiting ones', async () => {
  const h = harness();
  h.queue.speak('1단계', 'replace');
  h.queue.speak('알림', 'append');
  await h.loaded('1단계');
  h.queue.speak('2단계', 'replace');
  await settle();
  assert.deepEqual(h.log, ['load 1단계', 'load 알림', 'server 1단계', 'stop server 1단계', 'load 2단계']);
  await h.loaded('2단계');
  await h.finish('server 2단계');
  assert.equal(h.log.includes('server 알림'), false);
});

test('a sentence already waiting or being read is not queued twice', async () => {
  const h = harness();
  h.queue.speak('대상이 다릅니다.', 'append');
  h.queue.speak('대상이 다릅니다.', 'append');
  await h.loaded('대상이 다릅니다.');
  h.queue.speak('대상이 다릅니다.', 'append');
  await h.finish('server 대상이 다릅니다.');
  assert.deepEqual(h.log, ['load 대상이 다릅니다.', 'server 대상이 다릅니다.']);
});

test('a failed line is read by the browser voice in its place, and repeated failures turn the server off', async () => {
  const h = harness();
  h.queue.speak('하나', 'replace');
  h.queue.speak('둘', 'append');
  await h.failed('하나');
  assert.deepEqual(h.log, ['load 하나', 'load 둘', 'local 하나']);
  assert.equal(h.queue.serverOn, true);
  await h.failed('둘');
  await h.finish('local 하나');
  assert.equal(h.log.at(-1), 'local 둘');
  assert.equal(SERVER_FAILURES_BEFORE_OFF, 2);
  assert.equal(h.queue.serverOn, false);
  assert.equal(h.lost(), 1);
  await h.finish('local 둘');
  h.queue.speak('셋', 'append');
  await settle();
  assert.equal(h.log.at(-1), 'local 셋', 'no more server fetches once it is off');
});

test('a refusal that says nothing about the service (no session yet) does not count', async () => {
  const h = harness();
  for (const text of ['하나', '둘', '셋']) {
    h.queue.speak(text, 'replace');
    await h.failed(text, false);
    await h.finish(`local ${text}`);
  }
  assert.equal(h.queue.serverOn, true);
  assert.equal(h.lost(), 0);
});

test('audio the browser refuses to start falls back to the browser voice', async () => {
  const log: string[] = [];
  const queue = new SpeechQueue({
    load: async () => () => ({ done: Promise.reject(new Error('NotAllowedError')), stop: () => undefined }),
    local: (text: string) => { log.push(`local ${text}`); return { done: Promise.resolve(), stop: () => undefined }; },
    onSpeaking: () => undefined,
  }, true);
  queue.speak('안녕하세요', 'replace');
  await settle();
  assert.deepEqual(log, ['local 안녕하세요']);
});

test('cancel (microphone open) stops the line, aborts fetches and reports silence', async () => {
  const h = harness();
  h.queue.speak('하나', 'replace');
  h.queue.speak('둘', 'append');
  await h.loaded('하나');
  h.queue.cancel();
  await settle();
  assert.deepEqual(h.log, ['load 하나', 'load 둘', 'server 하나', 'stop server 하나']);
  assert.equal(h.speaking.at(-1), false);
  assert.equal(h.queue.serverOn, true, 'an aborted fetch is not a service failure');
});

test('without the server voice every line goes straight to the browser voice', async () => {
  const h = harness();
  h.queue.setServer(false);
  h.queue.speak('하나', 'replace');
  h.queue.speak('둘', 'append');
  await settle();
  assert.deepEqual(h.log, ['local 하나']);
  await h.finish('local 하나');
  assert.deepEqual(h.log, ['local 하나', 'local 둘']);
});

test('with neither voice nothing is queued', async () => {
  const h = harness({ local: false });
  h.queue.setServer(false);
  h.queue.speak('하나', 'replace');
  await settle();
  assert.deepEqual(h.log, []);
});

// ------------------------------------------------------------------------------------------- transport

function fakeFetch(answer: (url: string, init: RequestInit) => Response | Promise<Response>) {
  const calls: { url: string; body: unknown }[] = [];
  const fetcher = async (url: string, init: RequestInit) => {
    calls.push({ url, body: init.body ? JSON.parse(String(init.body)) : null });
    return answer(url, init);
  };
  return { fetcher, calls };
}

test('the server voice fetches each sentence once and serves repeats from its cache', async () => {
  const { fetcher, calls } = fakeFetch(() => new Response(new Uint8Array([1, 2, 3]), { headers: { 'content-type': 'audio/wav' } }));
  const voice = new ServerVoice(fetcher);
  const signal = new AbortController().signal;
  const first = await voice.load('완료를 확인했습니다.', signal);
  const again = await voice.load('완료를 확인했습니다.', signal);
  assert.equal(first, again);
  assert.deepEqual(calls, [{ url: '/api/tts', body: { text: '완료를 확인했습니다.' } }]);
});

test('server voice errors say whether the service is at fault', async () => {
  const signal = new AbortController().signal;
  const failure = async (status: number) => {
    const voice = new ServerVoice(fakeFetch(() => new Response('{}', { status })).fetcher, undefined, { tries: 1, delayMs: 0 });
    return voice.load('안녕', signal).then(() => null, (error: InstanceType<typeof SpeechLoadError>) => error.countable);
  };
  assert.equal(await failure(401), false);
  assert.equal(await failure(422), false);
  assert.equal(await failure(503), true);
  assert.equal(await failure(504), true);
  const offline = new ServerVoice(fakeFetch(() => Promise.reject(new TypeError('Failed to fetch'))).fetcher);
  assert.equal(await offline.load('안녕', signal).catch((error) => error.countable), true);
});

test('a line asked before the session exists (401) is asked again, then read in the server voice', async () => {
  let asked = 0;
  const { fetcher } = fakeFetch(() => {
    asked += 1;
    return asked < 3 ? new Response('{}', { status: 401 }) : new Response(new Blob(['wav']), { status: 200 });
  });
  const voice = new ServerVoice(fetcher, undefined, { tries: 5, delayMs: 1 });
  const blob = await voice.load('계획을 세우고 있어요', new AbortController().signal);
  assert.equal(asked, 3);
  assert.equal(await blob.text(), 'wav');
  let refused = 0;
  const never = new ServerVoice(fakeFetch(() => (refused += 1, new Response('{}', { status: 401 }))).fetcher, undefined, { tries: 3, delayMs: 1 });
  assert.equal(await never.load('안녕', new AbortController().signal).catch((error) => error.countable), false);
  assert.equal(refused, 3);
});

test('a slow server line times out as a service failure; a cancelled one does not count', async () => {
  const hang = (_url: string, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
    init.signal!.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
  });
  const slow = new ServerVoice(fakeFetch(hang).fetcher, 5);
  assert.equal(await slow.load('안녕', new AbortController().signal).catch((error) => error.countable), true);
  const cancelled = new AbortController();
  const pending = new ServerVoice(fakeFetch(hang).fetcher, 10_000).load('안녕', cancelled.signal).catch((error) => error.countable);
  cancelled.abort();
  assert.equal(await pending, false);
});

test('the default fetcher calls the global fetch unbound (browsers reject fetch with a foreign this)', async () => {
  const original = globalThis.fetch;
  const receivers: unknown[] = [];
  globalThis.fetch = function (this: unknown) {
    receivers.push(this);
    if (this !== undefined && this !== globalThis) return Promise.reject(new TypeError('Illegal invocation'));
    return Promise.resolve(new Response(new Uint8Array([1]), { headers: { 'content-type': 'audio/wav' } }));
  } as typeof fetch;
  try {
    const blob = await new ServerVoice().load('안녕하세요', new AbortController().signal);
    assert.equal(blob.size, 1);
    assert.equal(receivers.length, 1);
  } finally {
    globalThis.fetch = original;
  }
});

test('config is on only when the app says enabled and ready', async () => {
  const config = (body: unknown, status = 200) => fetchTtsConfig(fakeFetch(() => new Response(JSON.stringify(body), { status })).fetcher);
  assert.deepEqual(await config({ enabled: true, ready: true }), { enabled: true, ready: true });
  assert.deepEqual(await config({ enabled: true, ready: false }), { enabled: true, ready: false });
  assert.deepEqual(await config({}, 404), { enabled: false, ready: false });
  assert.deepEqual(await fetchTtsConfig(fakeFetch(() => Promise.reject(new TypeError('offline'))).fetcher),
    { enabled: false, ready: false });
});
