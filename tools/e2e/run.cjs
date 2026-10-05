// (owner P copy of N's: APP_PORT env, FOLLOW_PAID=1 counts follows as paid, shared budget 40) Real end-to-end run: real page on 127.0.0.1:8043 (real DeepSeek + production tracker), headless Chromium,
// synthetic camera from a y4m clip. Usage: node run.cjs <name> <clip.y4m> <durationSec> <goal>
const fs = require('fs');
const path = require('path');
const puppeteer = require(process.env.E2E_PUPPETEER || '/home/riey/Projects/mouse/node_modules/puppeteer-core');

const [, , NAME, CLIP, DUR_S, GOAL] = process.argv;
const DUR_MS = Number(DUR_S) * 1000;
const E = process.env.E2E_OUT || __dirname;
const OUT = path.join(E, 'runs', NAME);
fs.mkdirSync(OUT, { recursive: true });
const BUDGET_FILE = path.join(E, 'budget.json');
const BUDGET_TOTAL = 40;
const budget = fs.existsSync(BUDGET_FILE) ? JSON.parse(fs.readFileSync(BUDGET_FILE, 'utf8')) : { used: 0, runs: [] };
const paidLeft = BUDGET_TOTAL - budget.used;
if (paidLeft <= 0) { console.log('BUDGET EXHAUSTED', budget.used); process.exit(3); }
const PORT = process.env.APP_PORT || '8043';
const FOLLOW_PAID = process.env.FOLLOW_PAID === '1';
const URL = `http://127.0.0.1:${PORT}/`;
// Utterances to type into the TalkBar: E2E_TALK=<json> with [{ clipMs, text }] (clip time from guide start).
const TALKS = process.env.E2E_TALK ? JSON.parse(fs.readFileSync(process.env.E2E_TALK, 'utf8')) : [];
const log = (...a) => console.log(new Date().toISOString().slice(11, 23), ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Injected before any page script: fetch instrumentation, budget guard, 100 ms DOM sampler.
const INSTRUMENT = (paidLeftInit, followPaid) => {
  const E2E = (window.__e2e = {
    paidLeft: paidLeftInit, followPaid, paidSent: 0, blockGuide: false, net: [], track: [], samples: [], marks: {},
    tGuide: null, clipEndAt: null, firstFrameAt: null, frames: 0,
  });
  const strip = (v, depth = 0) => {
    if (typeof v === 'string') return v.length > 300 ? `<${v.length} chars>` : v;
    if (Array.isArray(v)) return v.map((x) => strip(x, depth + 1));
    if (v && typeof v === 'object') { const o = {}; for (const k of Object.keys(v)) o[k] = strip(v[k], depth + 1); return o; }
    return v;
  };
  const origFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    const isGuide = url.includes('/api/guide/');
    const isTrack = url.includes('/api/track/');
    if (!isGuide && !isTrack) return origFetch(input, init);
    const t0 = performance.now();
    let body = null;
    try { body = init && typeof init.body === 'string' ? JSON.parse(init.body) : null; } catch { body = null; }
    if (isGuide) {
      const isPaid = E2E.followPaid || !url.endsWith('/follow');
      if (E2E.blockGuide || (isPaid && E2E.paidLeft <= 0)) {
        E2E.net.push({ kind: 'guide', url, t0, blocked: E2E.blockGuide ? 'after_clip_end' : 'budget', trigger: body && body.trigger });
        throw new TypeError('e2e harness: guide call blocked (' + (E2E.blockGuide ? 'after clip end' : 'budget') + ')');
      }
      if (isPaid) { E2E.paidLeft -= 1; E2E.paidSent += 1; }
      let imgSig = null;
      if (body && body.scene && typeof body.scene.image_base64 === 'string') {
        const str = body.scene.image_base64; let h = 0x811c9dc5;
        for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
        imgSig = `${str.length}:${h.toString(16)}`;
      }
      const rec = { kind: 'guide', url, t0, trigger: body && body.trigger, utterance: body && body.utterance, imgSig, frameId: body && body.scene && body.scene.frame_id, req: strip(body), events: [] };
      E2E.net.push(rec);
      try {
        const res = await origFetch(input, init);
        rec.status = res.status; rec.tHeaders = performance.now(); rec.ctype = res.headers.get('content-type');
        const clone = res.clone();
        (async () => {
          try {
            if ((rec.ctype || '').includes('text/event-stream')) {
              const reader = clone.body.getReader(); const dec = new TextDecoder(); let buf = '';
              for (;;) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += dec.decode(value, { stream: true });
                let i;
                while ((i = buf.indexOf('\n\n')) >= 0) {
                  const block = buf.slice(0, i); buf = buf.slice(i + 2);
                  const ev = (block.match(/^event: (.*)$/m) || [])[1] || 'comment';
                  const data = (block.match(/^data: (.*)$/m) || [])[1];
                  let parsed = null; try { parsed = data ? JSON.parse(data) : null; } catch { parsed = data; }
                  rec.events.push({ t: performance.now(), event: ev, data: ev === 'final' ? null : strip(parsed) });
                  if (ev === 'final') rec.res = strip(parsed);
                }
              }
            } else {
              rec.res = strip(await clone.json());
            }
          } catch (e) { rec.readError = String(e); }
          rec.tEnd = performance.now();
        })();
        return res;
      } catch (e) { rec.error = String(e); rec.tEnd = performance.now(); throw e; }
    }
    // track lane
    const rec = { kind: url.split('/api/track/')[1], t0, seq: body && body.frame_seq };
    try {
      const res = await origFetch(input, init);
      rec.status = res.status;
      res.clone().json().then((j) => {
        rec.tEnd = performance.now();
        if (j && typeof j === 'object') { rec.state = j.state; rec.transition = j.transition; rec.box = j.box || null; rec.run = j.run_id; rec.gen = j.generation; rec.code = j.code || (j.error && j.error.code) || undefined; }
      }).catch(() => { rec.tEnd = performance.now(); });
      E2E.track.push(rec);
      return res;
    } catch (e) { rec.error = String(e); rec.tEnd = performance.now(); E2E.track.push(rec); throw e; }
  };
  const q = (id) => document.querySelector(`[data-testid="${id}"]`);
  const txt = (id) => { const n = q(id); return n ? n.textContent : null; };
  let canvas = null;
  const luma = (video) => {
    try {
      if (!canvas) { canvas = document.createElement('canvas'); canvas.width = 16; canvas.height = 9; }
      const c = canvas.getContext('2d', { willReadFrequently: true });
      c.drawImage(video, 0, 0, 16, 9);
      const d = c.getImageData(0, 0, 16, 9).data; let s = 0;
      for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      return Math.round((s / (d.length / 4)) * 10) / 10;
    } catch { return null; }
  };
  const hookVideo = () => {
    const v = q('camera-video');
    if (!v || v.__e2eHooked || !v.requestVideoFrameCallback) return;
    v.__e2eHooked = true;
    const cb = (now, meta) => {
      if (E2E.firstFrameAt === null) E2E.firstFrameAt = now;
      E2E.frames = meta.presentedFrames;
      v.requestVideoFrameCallback(cb);
    };
    v.requestVideoFrameCallback(cb);
  };
  setInterval(hookVideo, 50);
  E2E.sample = () => {
    const now = performance.now();
    const bar = q('narration-bar'); const ov = q('anchored-guidance'); const chip = q('live-track-chip');
    const badge = q('track-state-badge'); const dbg = q('debug-guide'); const card = q('plan-card');
    const v = q('camera-video');
    const s = {
      t: Math.round(now), tg: E2E.tGuide === null ? null : Math.round(now - E2E.tGuide),
      clipT: E2E.clip0 == null ? null : Math.round(now - E2E.clip0),
      frames: E2E.frames, luma: v ? luma(v) : null,
      phase: bar && bar.dataset.phase, stepId: bar && bar.dataset.stepId, completion: bar && bar.dataset.completion,
      textOnly: bar && bar.dataset.textOnly,
      say: txt('narration-say'), sayStreaming: q('narration-say') && q('narration-say').dataset.streaming,
      partialTarget: txt('narration-partial-target'), trigger: txt('narration-trigger'),
      provider: q('narration-provider') && q('narration-provider').dataset.provider, elapsed: txt('narration-elapsed'),
      notice: txt('narration-notice'), reselect: txt('narration-reselect'), completionText: txt('narration-completion'),
      planRev: card && card.dataset.planRevision,
      steps: [...document.querySelectorAll('[data-testid="plan-step"]')].map((n) => `${n.dataset.stepId}:${n.dataset.stepState}`).join(','),
      ovVisible: ov && ov.dataset.guideVisible, ovReason: ov && ov.dataset.guideReason, ovWarn: ov && ov.dataset.guideWarn,
      label: txt('anchored-label'),
      chipState: chip && chip.dataset.liveState, chipAge: chip && chip.dataset.liveAgeMs, chipText: chip && chip.textContent,
      badge: badge && badge.dataset.trackState,
      dbgFirstSay: dbg && dbg.dataset.firstSayMs, dbgFinalPlan: dbg && dbg.dataset.finalPlanMs, dbgFirstBox: dbg && dbg.dataset.firstBoxMs,
      dbgText: dbg && (dbg.querySelector('.debug-numbers') || {}).textContent,
      dbgCalls: [...document.querySelectorAll('[data-testid="debug-guide-call"]')].map((n) => `${n.dataset.stage}|${n.dataset.accepted}|${n.textContent}`),
      cost: txt('guide-cost'),
      dbgChanged: dbg && dbg.dataset.targetChanged, dbgMoved: dbg && dbg.dataset.targetMoved,
      dbgOther: dbg && dbg.dataset.followOtherStep, dbgCapped: dbg && dbg.dataset.budgetCapped,
      budget: txt('narration-budget'),
      labelPlacement: q('anchored-label') && q('anchored-label').dataset.placement,
      motion: ov && ov.dataset.motion, motionPhase: ov && ov.dataset.motionPhase, focus: ov && ov.dataset.focus,
      labelAction: q('anchored-label') && q('anchored-label').dataset.action,
      talkPending: q('talk-bar') && q('talk-bar').dataset.pending,
      talkUtterance: txt('talk-last-utterance'), talkReply: txt('talk-reply'),
      talkError: txt('talk-error'), talkBlocked: txt('talk-blocked'),
      dbgTalk: dbg && dbg.dataset.talk,
    };
    E2E.samples.push(s);
    return s;
  };
  E2E.startSampling = () => { if (!E2E.timer) E2E.timer = setInterval(E2E.sample, 100); };
  E2E.stopSampling = () => { clearInterval(E2E.timer); E2E.timer = null; };
};

(async () => {
  const browser = await puppeteer.launch({
    executablePath: process.env.E2E_CHROMIUM || '/usr/bin/chromium',
    headless: 'new',
    args: [
      '--no-sandbox', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
      `--use-file-for-fake-video-capture=${CLIP}`, '--autoplay-policy=no-user-gesture-required', '--window-size=1440,1100',
    ],
  });
  const ctx = browser.defaultBrowserContext();
  await ctx.overridePermissions(`http://127.0.0.1:${PORT}`, ['camera', 'microphone']);
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1100 });
  const consoleLines = [];
  page.on('pageerror', (e) => consoleLines.push('PAGEERROR ' + e.message));
  page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) consoleLines.push(`${m.type()} ${m.text()}`); });
  await page.evaluateOnNewDocument(INSTRUMENT, process.env.DRY ? 0 : paidLeft, FOLLOW_PAID);
  await page.goto(URL, { waitUntil: 'networkidle0' });
  await page.keyboard.press('Escape');
  const clickId = async (id) => { await page.waitForSelector(`[data-testid="${id}"]:not([disabled])`, { timeout: 20000 }); await page.click(`[data-testid="${id}"]`); };

  // goal text
  await page.focus('#goal-input');
  await page.keyboard.sendCharacter(GOAL);
  const goalVal = await page.$eval('#goal-input', (n) => n.value);
  log('goal set:', goalVal);

  // camera
  await clickId('camera-start');
  await page.waitForFunction(() => window.__e2e.firstFrameAt !== null, { timeout: 20000 });
  const cam = await page.evaluate(() => { const v = document.querySelector('[data-testid="camera-video"]'); return { w: v.videoWidth, h: v.videoHeight, firstFrameAt: window.__e2e.firstFrameAt }; });
  log('camera live', cam);
  // consent
  if (await page.$('[data-testid="track-consent-grant"]')) await clickId('track-consent-grant');
  await page.waitForSelector('[data-testid="guide-start"]:not([disabled])', { timeout: 60000 });
  const health = await page.evaluate(() => fetch('/api/health').then((r) => r.json()));
  log('health', JSON.stringify(health));
  await page.evaluate(() => window.__e2e.startSampling());

  // Align guide start with the beginning of the y4m's second loop (the user presses 가이드 시작, then acts).
  const nowP = await page.evaluate(() => performance.now());
  const loopK = Math.max(1, Math.ceil((nowP + 400 - cam.firstFrameAt) / DUR_MS));
  const loopStart = cam.firstFrameAt + loopK * DUR_MS; // page performance.now() of loop k frame 0 (nominal 30 fps)
  await page.evaluate((ls) => { window.__e2e.clip0 = ls; }, loopStart);
  const waitMs = loopStart - nowP + 30;
  if (waitMs < 0) throw new Error(`setup took longer than one loop (${Math.round(-waitMs)} ms late)`);
  await sleep(Math.max(0, waitMs - 150));
  await page.waitForFunction((ls) => performance.now() >= ls + 30, { polling: 5 }, loopStart);
  const tGuide = await page.evaluate((ls) => {
    window.__e2e.clip0 = ls; window.__e2e.tGuide = performance.now();
    document.querySelector('[data-testid="guide-start"]').click();
    return window.__e2e.tGuide;
  }, loopStart);
  log(`guide-start clicked at clip t=${Math.round(tGuide - loopStart)} ms (loop ${loopK + 1})`);
  const clipEnd = loopStart + DUR_MS;

  // E2E_SHOTS_MS=<ms>: from guide start until the clip ends, screenshot the camera stage every <ms> into
  // <OUT>/shots/t<clipMs>.png and record the overlay/callout state at that instant (stageShots in the timeline).
  const SHOTS_MS = Number(process.env.E2E_SHOTS_MS || 0);
  const stageShots = [];
  const stageShotLoop = (async () => {
    if (!(SHOTS_MS > 0)) return;
    const dir = path.join(OUT, 'shots');
    fs.mkdirSync(dir, { recursive: true });
    for (let k = 0; ; k += 1) {
      const due = tGuide + k * SHOTS_MS;
      const now = await page.evaluate(() => performance.now());
      if (due >= clipEnd || now >= clipEnd) break;
      if (now < due) await sleep(due - now);
      try {
        const rec = await page.evaluate((ls) => {
          const ov = document.querySelector('[data-testid="anchored-guidance"]');
          const lab = document.querySelector('[data-testid="anchored-label"]');
          const s = window.__e2e.samples[window.__e2e.samples.length - 1] || {};
          return {
            clipMs: Math.round(performance.now() - ls),
            ovVisible: ov ? ov.dataset.guideVisible : null, ovReason: ov ? ov.dataset.guideReason : null,
            motion: ov ? ov.dataset.motion || null : null, motionPhase: ov ? ov.dataset.motionPhase || null : null,
            calloutAvoid: lab ? lab.dataset.avoid || null : null,
            calloutVisible: lab ? getComputedStyle(lab).visibility : null,
            stepId: s.stepId, label: s.label,
          };
        }, loopStart);
        if (rec.clipMs >= DUR_MS) break;
        const file = path.join(dir, `t${String(rec.clipMs).padStart(5, '0')}.png`);
        const el = await page.$('.camera-stage-media');
        if (el) await el.screenshot({ path: file }); else await page.screenshot({ path: file });
        rec.file = path.relative(OUT, file);
        rec.clipMsAfter = Math.round((await page.evaluate(() => performance.now())) - loopStart);
        stageShots.push(rec);
      } catch (e) { stageShots.push({ error: String(e) }); }
    }
    log(`stage shots: ${stageShots.length} every ${SHOTS_MS} ms`);
  })();

  const shots = {};
  const shotTimes = {};
  const shoot = async (name) => {
    if (shots[name]) return;
    shots[name] = true;
    const tA = await page.evaluate(() => performance.now());
    const el = await page.$('.camera-stage');
    const file = path.join(OUT, `${name}.png`);
    if (el) await el.screenshot({ path: file }); else await page.screenshot({ path: file });
    const wide = path.join(OUT, `${name}-page.png`);
    await page.screenshot({ path: wide });
    const tB = await page.evaluate(() => performance.now());
    shotTimes[name] = { clipMsBefore: Math.round(tA - loopStart), clipMsAfter: Math.round(tB - loopStart) };
    log('screenshot', name, JSON.stringify(shotTimes[name]));
  };
  for (;;) {
    const st = await page.evaluate(() => {
      const s = window.__e2e.samples[window.__e2e.samples.length - 1] || {};
      return { now: performance.now(), say: s.say, ov: s.ovVisible, phase: s.phase };
    });
    if (st.say && st.say.trim()) await shoot('first-text');
    if (st.ov === 'true') await shoot('first-overlay');
    if (st.now >= clipEnd - 450) await shoot('end');
    for (const t of TALKS) {
      if (t.clipMsSent !== undefined || st.now < loopStart + t.clipMs) continue;
      if (!(await page.$('[data-testid="talk-input"]:not([disabled])'))) {
        if (!t.waitLogged) { log('talk waiting for the input', t.text); t.waitLogged = true; }
        continue;
      }
      await page.click('[data-testid="talk-input"]');
      await page.keyboard.sendCharacter(t.text);
      await page.keyboard.press('Enter');
      const sentAt = await page.evaluate(() => performance.now());
      t.clipMsSent = Math.round(sentAt - loopStart);
      log('talk sent', JSON.stringify({ text: t.text, clipMs: t.clipMsSent }));
      await shoot(`talk-${TALKS.indexOf(t)}`);
    }
    if (st.now >= clipEnd) break;
    await sleep(100);
  }
  await stageShotLoop;
  await page.evaluate((ce) => { window.__e2e.blockGuide = true; window.__e2e.clipEndAt = ce; }, clipEnd);
  log('clip end reached; new guide calls blocked; waiting for in-flight calls');
  for (let i = 0; i < 80; i++) {
    const inflight = await page.evaluate(() => window.__e2e.net.filter((r) => r.kind === 'guide' && !r.blocked && r.tEnd === undefined).length);
    if (inflight === 0) break;
    await sleep(100);
  }
  await sleep(300);
  await page.evaluate(() => window.__e2e.sample());
  await shoot('after-end-settled');
  await page.evaluate(() => window.__e2e.stopSampling());
  // stop guide + track
  if (await page.$('[data-testid="track-stop"]:not([disabled])')) await page.click('[data-testid="track-stop"]');
  await sleep(500);
  const data = await page.evaluate(() => {
    const e = window.__e2e;
    return { paidSent: e.paidSent, tGuide: e.tGuide, clip0: e.clip0, clipEndAt: e.clipEndAt, firstFrameAt: e.firstFrameAt, net: e.net, track: e.track, samples: e.samples };
  });
  const finalDbg = await page.$$eval('[data-testid="debug-guide-call"]', (ns) => ns.map((n) => n.textContent));
  const result = { name: NAME, screenshots: shotTimes, clip: CLIP, durationMs: DUR_MS, goal: GOAL, talks: TALKS, camera: cam, health, finalDebugCalls: finalDbg, console: consoleLines, ...data };
  if (SHOTS_MS > 0) Object.assign(result, { stageShotsEveryMs: SHOTS_MS, stageShots });
  fs.writeFileSync(path.join(OUT, 'timeline.json'), JSON.stringify(result, null, 1));
  if (!process.env.DRY) budget.used += data.paidSent;
  budget.runs.push({ name: NAME, paidSent: data.paidSent, at: new Date().toISOString() });
  fs.writeFileSync(BUDGET_FILE, JSON.stringify(budget, null, 1));
  log(`done: paid calls sent this run=${data.paidSent}, budget used=${budget.used}/${BUDGET_TOTAL}, samples=${data.samples.length}, track posts=${data.track.length}`);
  await browser.close();
})().catch((e) => { console.error('HARNESS ERROR', e); process.exit(1); });
