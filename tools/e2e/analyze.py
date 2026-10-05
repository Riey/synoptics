"""Summarise one e2e timeline.json -> summary.json (+ printed). Times: tg = ms since 가이드 시작 click,
clip = ms since the clip's frame 0 of the measured loop (±100 ms, see align.py)."""
import json, re, sys, statistics, collections

RATES = {'cached': 0.006, 'miss': 0.30, 'out': 1.20}  # USD / 1M tokens, web/src/coach/liveCost.ts peak rates


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 1)


def main(path):
    d = json.load(open(path))
    tg0, c0, D = d['tGuide'], d['clip0'], d['durationMs']
    clip_end = c0 + D
    S = d['samples']
    out = {'run': d['name'], 'goal': d['goal'], 'clip_ms': round(D), 'guide_click_clip_ms': round(tg0 - c0)}
    rel = lambda t: None if t is None else round(t - tg0)
    clip = lambda t: None if t is None else round(t - c0)

    # --- plan stream
    plan = next(r for r in d['net'] if r['url'].endswith('/plan'))
    ev = {e['event'] + (':' + list(e['data'].keys())[0] if e['event'] == 'partial' and isinstance(e['data'], dict) else ''): rel(e['t']) for e in plan.get('events', [])}
    last = S[-1]
    first_say_dom = next((s['tg'] for s in S if s['tg'] is not None and s['tg'] >= 0 and s['say'] and s['say'].strip()), None)
    first_box_dom = next((s['tg'] for s in S if s['tg'] is not None and s['ovVisible'] == 'true'), None)
    out['plan'] = {
        'sse_partial_target_ms': ev.get('partial:target'), 'sse_partial_first_say_ms': ev.get('partial:first_say'),
        'sse_final_ms': ev.get('final'), 'request_sent_ms': rel(plan['t0']),
        'target': (plan.get('res') or {}).get('selection', {}).get('target'),
        'steps': [(s['id'], s['say'], s['done_when']) for s in (plan.get('res') or {}).get('steps', [])],
        'goal_when': (plan.get('res') or {}).get('goal_when'),
    }
    out['debug_guide'] = {'msStartToFirstSay': last['dbgFirstSay'], 'msStartToFinalPlan': last['dbgFinalPlan'], 'msStartToFirstBox': last['dbgFirstBox'], 'text': last['dbgText']}
    out['dom_sampled_100ms'] = {'first_say_visible_ms': first_say_dom, 'first_anchored_overlay_visible_ms': first_box_dom}
    track_frames = [r for r in d['track'] if r['kind'] == 'frame']
    first_tracking = next((r for r in track_frames if r.get('state') == 'tracking'), None)
    out['first_tracking_response_ms'] = rel(first_tracking['tEnd']) if first_tracking else None
    out['first_tracking_response_clip_ms'] = clip(first_tracking['tEnd']) if first_tracking else None

    # --- guide calls in order
    dom_calls = list(reversed(d['finalDebugCalls']))  # oldest first
    calls, usage = [], collections.Counter()
    for i, r in enumerate([r for r in d['net'] if r['kind'] == 'guide']):
        res = r.get('res') or {}
        u = res.get('usage') or {}
        for k in ('input_tokens', 'output_tokens', 'cached_input_tokens'):
            usage[k] += u.get(k) or 0
        stage = r['url'].rsplit('/', 1)[1]
        verdict = None
        if stage == 'follow':
            verdict = f"{res.get('step_id')}:{res.get('step_status')} goal_seen={res.get('goal_seen')} anchor={[(a['matches']) for a in res.get('anchor_verdicts') or []]}"
        elif stage == 'confirm':
            verdict = f"goal={(res.get('goal_status') or {}).get('status')} step_check={res.get('step_check')} step_id={res.get('step_id')} replan={'yes' if res.get('replan') else 'no'}"
        elif stage == 'plan':
            verdict = f"target={out['plan']['target']} steps={len(out['plan']['steps'])}"
        calls.append({
            'stage': stage, 'trigger': r.get('trigger') or 'start', 'sent_tg_ms': rel(r['t0']), 'sent_clip_ms': clip(r['t0']),
            'client_ms_to_final': round(r['tEnd'] - r['t0']) if r.get('tEnd') else None, 'server_latency_ms': res.get('latency_ms'),
            'status': r.get('status'), 'blocked': r.get('blocked'), 'provider': res.get('provider'),
            'usage_in_out_cached': [u.get('input_tokens'), u.get('output_tokens'), u.get('cached_input_tokens')],
            'verdict': verdict, 'rationale': ((res.get('goal_status') or {}).get('rationale')),
            'after_clip_end': r['t0'] >= clip_end,
        })
    real = [c for c in calls if not c['blocked']]
    for c, dc in zip(real, dom_calls):
        c['dom'] = dc
        c['accepted'] = dc.rsplit(' · ', 1)[-1]
    out['calls'] = calls
    out['blocked_calls'] = [c for c in calls if c['blocked']]
    cost = (usage['cached_input_tokens'] * RATES['cached'] + (usage['input_tokens'] - usage['cached_input_tokens']) * RATES['miss'] + usage['output_tokens'] * RATES['out']) / 1e6
    out['usage'] = dict(usage, paid_calls=len(real), est_cost_usd_peak=round(cost, 6), app_cost_line=last['cost'])

    # --- counters from the debug block
    m = re.search(r'단계 완료\(추종\)=(\d+) · 불확실\(추종\)=(\d+) · 확정자 번복=(\d+) · 대상 이동 트리거=(\d+)', last['dbgText'] or '')
    out['counters'] = dict(zip(['follow_done', 'follow_unsure', 'confirm_reversals', 'target_moved'], map(int, m.groups()))) if m else None

    # --- step / completion / notice timeline (clip time), within and after the clip
    tl, prev = [], {}
    for s in S:
        if s['clipT'] is None or s['tg'] is None or s['tg'] < 0:
            continue
        for k in ('phase', 'stepId', 'completion', 'notice', 'say', 'chipState', 'ovVisible', 'ovReason', 'ovWarn', 'textOnly', 'reselect'):
            if s[k] != prev.get(k):
                tl.append((s['clipT'], k, s[k]))
                prev[k] = s[k]
    out['ui_changes_clip_ms'] = tl
    in_clip = [s for s in S if s['clipT'] is not None and 0 <= s['clipT'] < D and s['tg'] is not None and s['tg'] >= 0]
    trk = [s for s in in_clip if s['chipState'] == 'tracking']
    vis = [s for s in trk if s['ovVisible'] == 'true']
    out['overlay'] = {
        'samples_in_clip': len(in_clip), 'samples_tracking': len(trk), 'visible_while_tracking': len(vis),
        'fraction_visible_while_tracking': round(len(vis) / len(trk), 3) if trk else None,
        'samples_visible_any_state': sum(1 for s in in_clip if s['ovVisible'] == 'true'),
        'reasons': collections.Counter(s['ovReason'] for s in in_clip), 'chip_states': collections.Counter(s['chipState'] for s in in_clip),
        'warn_samples': sum(1 for s in in_clip if s['ovWarn'] == 'true'),
    }
    ages = [int(s['chipAge']) for s in trk if s['chipAge'] not in (None, '')]
    out['chip_age_ms_while_tracking'] = {'n': len(ages), 'p50': pct(ages, 50), 'p95': pct(ages, 95), 'max': max(ages) if ages else None}
    comp = [(s['clipT'], s['completion']) for s in S if s['completion'] in ('checking', 'confirmed', 'user_confirmed')]
    out['completion_first'] = comp[0] if comp else None
    out['completion_final'] = last['completion']
    out['visually_satisfied_calls'] = [c for c in calls if c['verdict'] and 'goal=visually_satisfied' in c['verdict']]

    # --- tracker loop
    tf = [r for r in track_frames if r['t0'] < clip_end and r.get('tEnd')]
    if tf:
        span = (tf[-1]['t0'] - tf[0]['t0']) / 1000
        rtts = [r['tEnd'] - r['t0'] for r in tf]
        gaps = [b['t0'] - a['t0'] for a, b in zip(tf, tf[1:])]
        segs, cur = [], None
        for r in tf:
            st = r.get('state') or f"http{r.get('status')}"
            if not cur or cur[1] != st:
                cur = [clip(r['t0']), st, 0]
                segs.append(cur)
            cur[2] += 1
        out['tracker'] = {
            'frames_posted_in_clip': len(tf), 'first_post_clip_ms': clip(tf[0]['t0']), 'span_s': round(span, 2),
            'fps': round((len(tf) - 1) / span, 2) if span > 0 else None,
            'rtt_ms_p50': pct(rtts, 50), 'rtt_ms_p95': pct(rtts, 95), 'rtt_ms_max': round(max(rtts), 1),
            'inter_post_ms_p50': pct(gaps, 50), 'inter_post_ms_p95': pct(gaps, 95),
            'states': collections.Counter(r.get('state') for r in tf), 'segments_clip_ms_state_n': segs,
            'control_calls': [(r['kind'], clip(r['t0']), r.get('status')) for r in d['track'] if r['kind'] != 'frame'],
        }
    out['console_errors'] = d['console']
    json.dump(out, open(path.replace('timeline.json', 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
