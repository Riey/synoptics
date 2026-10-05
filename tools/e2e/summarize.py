import json,sys
d=json.load(open(sys.argv[1])); c0=d['clip0']
print('clip',d['name'],'paid',d['paidSent'])
for r in d['net']:
    if r.get('kind')!='guide': continue
    stage=r['url'].split('/api/guide/')[1].split('?')[0]
    t=round(r['t0']-c0); rt=round(r['tEnd']-r['t0']) if r.get('tEnd') else None
    if r.get('blocked'): print(f'  {stage:8} {r.get("trigger")} @{t} BLOCKED {r["blocked"]}'); continue
    res=r.get('res') or {}
    st=r.get('status')
    if st and st>=400: print(f'  {stage:8} {r.get("trigger")} @{t} rt={rt} HTTP {st} {res.get("code") or res.get("error") or res}'); continue
    if stage=='plan':
        ev=[e['event'] for e in r.get('events',[])]
        steps=[(s['id'],s['done_when']) for s in res.get('steps',[])] if res else None
        print(f'  plan @{t} rt={rt} events={ev} target={(res.get("selection") or {}).get("target") if res else None} steps={steps}')
    elif stage=='follow':
        checks=' '.join(f"{c['step_id']}:{c['visible']}" for c in res.get('step_checks',[])) if res.get('step_checks') else f"{res.get('step_id')}:{res.get('step_status')}"
        print(f'  follow   {str(r.get("trigger")):14} @{t} rt={rt} cur={(r.get("req") or {}).get("current_step")} {checks} goal_seen={res.get("goal_seen")} reselect={res.get("needs_reselect")}')
    else:
        fc=(r.get('req') or {}).get('follow_checks')
        print(f'  confirm  {str(r.get("trigger")):14} @{t} rt={rt} step={res.get("step_id")} check={res.get("step_check")} goal={(res.get("goal_status") or {}).get("status")} replan={"yes" if res.get("replan") else "no"}'+(f' fc={[c["step_id"]+":"+c["visible"] for c in fc]}' if fc else ''))
prev=None
for s in d['samples']:
    key=(s.get('stepId'),s.get('completion'),s.get('notice'),json.dumps(s.get('steps'),ensure_ascii=False))
    if key!=prev:
        print(f'  ui @{round(s["clipT"])} step={s.get("stepId")} completion={s.get("completion")} notice={s.get("notice")} steps={s.get("steps")}'); prev=key
