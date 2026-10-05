#!/usr/bin/env python
"""One-off actual Live REST/WS -> backend ASGI -> canned issuer/Clef/tracker proof.

Synthetic images only. No real network, credentials, model inference, deployment or training.
Run with the upstream worktree's Python and --live-source pointing at the preserved isolated Live source.
"""
from __future__ import annotations

import asyncio
import argparse
import base64
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--live-source', required=True, type=Path)
args = parser.parse_args()
LIVE = args.live_source.resolve()
sys.path[:0] = [str(ROOT), str(LIVE / 'src'), str(LIVE / 'tests')]
for key in list(os.environ):
    if key.startswith(('AISW_', 'OPENAI_', 'DEEPSEEK_', 'GEMINI_', 'GOOGLE_')):
        os.environ.pop(key, None)
os.environ.update(PROVIDER='deepseek', DEEPSEEK_API_KEY='test-only-key',
    DEEPSEEK_API_KEY_FILE='/dev/null', OPENAI_API_KEY_FILE='/dev/null',
    DEMO_ACCESS_CODE='graph-smoke', AISW_FOLLOW_PROVIDER='clef')

import httpx
from starlette.testclient import TestClient
from backend.app.main import app as backend_app, store
from backend.app.guide_contracts import GuidePlanToolOutput
from backend.app.tracking import TrackerClient
from backend.tests.test_guide import DeepSeekStub, ClefStub, clef_choice, unpace
from backend.tests.test_guide_stream import sse_lines
from backend.tests.test_track_select import RecordingTracker
from conftest import Live, create_session, fast_settings, frame, jpeg, open_live
from synoptics_live.app import create_app

_original_connect = socket.socket.connect

def deny_network(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        raise AssertionError(f'Unexpected network dispatch: {address}')
    return _original_connect(sock, address)

socket.socket.connect = deny_network

STEPS = [
    dict(id='s1', say='R1을 끼우세요', done_when='R1이 꽂혀 있다', details='R1 한 개의 두 다리를 지정된 서로 다른 구멍에 끼우세요.', commands=[], targets=['R1']),
    dict(id='s2', say='R2를 끼우세요', done_when='R2가 꽂혀 있다', details=None, commands=[], targets=['R2']),
    dict(id='s3', say='R1 연결을 확인하세요', done_when='R1의 연결 검토가 끝났다', details='사용자가 R1의 연결 상태를 검토한 뒤 확인하세요.', commands=[], check='user', required=True, requires=['s1'], targets=['R1']),
]
PLAN = dict(selection=dict(status='selected', target='circuit board', rationale='합성 검증 대상입니다.'),
    steps=STEPS, goal_when='R1과 R2가 꽂히고 연결 검토가 끝났다', evidence_kind='observed_scene', needs_clarification=False)
verdicts = {'s1': 'unsure', 's2': 'yes', 's3': 'unsure'}
QUESTION = '검토할 대상은 R1 연결이 맞나요?'
CLARIFY = dict(selection=dict(status='uncertain', target=None, rationale='검토 대상을 확인합니다.'),
               steps=[], goal_when='R1과 R2가 꽂히고 연결 검토가 끝났다',
               evidence_kind='observed_scene', needs_clarification=True, clarification_prompt=QUESTION)
GuidePlanToolOutput.model_validate(CLARIFY)
GuidePlanToolOutput.model_validate(PLAN)


def confirmation(body):
    text = next(p['text'] for p in body['messages'][1]['content'] if p['type'] == 'text')
    answer = dict(goal_status=dict(status='in_progress', rationale='합성 회로를 검증 중입니다.'),
                  evidence_kind='observed_scene', needs_clarification=False)
    match = re.search(r'확인할 단계: (s\d+)', text)
    if match:
        answer['step_check'] = verdicts.get(match[1], 'unsure')
    return answer


class Issuer(DeepSeekStub):
    async def handle(self, request):
        body = json.loads(request.content)
        if body.get('stream'):
            assert body['tools'][0]['function']['name'] == 'guide_plan'
            self.requests.append(body)
            response = CLARIFY if sum(bool(r.get('stream')) for r in self.requests) == 1 else PLAN
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  content=b''.join(sse_lines(json.dumps(response, ensure_ascii=False))))
        return await super().handle(request)


class BackendTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.inner = httpx.ASGITransport(app=backend_app)
        self.calls = []

    async def handle_async_request(self, request):
        assert str(request.url).startswith('http://127.0.0.1:8000/')
        body = json.loads(request.content) if request.content else {}
        # Accelerated smoke only: remove the backend's paid-call pacing, not its validation/fences.
        for session in store.sessions.values():
            unpace(session.id)
        response = await self.inner.handle_async_request(request)
        self.calls.append((request.url.path, response.status_code, body))
        return response

    async def aclose(self):
        await self.inner.aclose()


def main():
    issuer = Issuer({'guide_plan': PLAN, 'guide_confirm': confirmation})
    clef = ClefStub(lambda body: clef_choice({key: verdicts.get(key, 'no' if key == 'goal' else 'yes')
                    for key in body['questions']})(body), mode='choice')
    tracker = RecordingTracker()
    store.provider = issuer.provider
    store.clef_follower = clef.follower
    store.tracker_client = TrackerClient('http://127.0.0.1:8090', client=httpx.AsyncClient(transport=httpx.MockTransport(tracker.handler)))
    store.sessions.clear()
    store.creations.clear()
    transport = BackendTransport()
    live_app = create_app(fast_settings(engine='real', upstream_url='http://127.0.0.1:8000',
                          upstream_access_code='graph-smoke'), tts=None, upstream_transport=transport)
    report = dict(scope='actual Live REST/WebSocket -> actual backend ASGI -> canned issuer/Clef/tracker HTTP transports',
                  synthetic_images=True, real_model_calls=0, network_dispatch=0, training=False, deployment=False,
                  backend_pacing_disabled_for_smoke=True, states={})
    with TestClient(live_app) as client:
        health = client.get('/v1/health')
        assert health.status_code == 200 and health.json()['ready'], health.text
        session = create_session(client)
        with open_live(client, session['session_id']) as ws:
            live = Live(ws)
            live.hello(session['token'])
            live.send(dict(type='start', goal='R1과 R2를 꽂고 R1 연결을 검토하세요', consent_ai=True,
                           plan_mode=True, core_mode='graph', reference_images=[dict(frame_id='synthetic-ref',
                           label='합성 참고 사진', image_base64=base64.b64encode(jpeg()).decode())]))
            hi = live.until_type('capture_hi')
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi['req_id']))
            question = live.until(lambda m: m['type'] == 'error' or (m['type'] == 'state' and m['phase'] == 'clarifying'))
            assert question['type'] == 'state' and question['clarification'] == QUESTION, question
            assert not clef.requests and not tracker.run_paths()
            report['states']['clarifying_before_dispatch'] = question
            live.send(dict(type='plan_answer', clarification_id='stale-question', answer='늦은 답'))
            assert live.until_type('error')['code'] == 'clarification_stale'
            assert len(issuer.requests) == 1
            live.send(dict(type='plan_answer', clarification_id=question['clarification_id'], answer='R1 연결이 맞습니다'))
            hi = live.until_type('capture_hi')
            ws.send_bytes(frame(501, w=1024, h=576, hi_req=hi['req_id']))
            reviewing = live.until(lambda m: m['type'] == 'error' or (m['type'] == 'state' and m['phase'] == 'reviewing'))
            assert reviewing['type'] == 'state', reviewing
            assert reviewing['plan']['steps'][0]['details'] == STEPS[0]['details']
            assert reviewing['plan']['steps'][1]['details'] is None
            assert not clef.requests and not tracker.run_paths()
            report['states']['reviewing_after_answer'] = reviewing
            plan_calls = [body for path, _, body in transport.calls if path == '/api/guide/plan']
            assert len(plan_calls) == 2
            assert all(p['assisted'] is True and p['core_mode'] == 'graph' for p in plan_calls)
            assert plan_calls[1]['answers'] == [dict(question=QUESTION, answer='R1 연결이 맞습니다')]
            assert plan_calls[1]['reference_images'] == plan_calls[0]['reference_images']
            report['clarification_proof'] = dict(stale_answer_rejected=True, explicit_answer_replanned=True,
                reference_photos_retained=True, assisted_graph_both_turns=True, preapproval_tracking_calls=0)
            live.send(dict(type='plan_edit', edits=[dict(step_id='s1', say='R1을 지정한 곳에 끼우세요')]))
            edited = live.until(lambda m: m['type'] == 'state' and m['plan']['revision'] == reviewing['plan']['revision'] + 1)
            assert edited['plan']['steps'][0]['details'] == STEPS[0]['details']
            live.send(dict(type='plan_approve'))
            live.until(lambda m: m['type'] == 'state' and m['phase'] == 'running')
            second = live.drive(lambda s: 's2' in s['steps_done'], max_frames=1400, pause_s=.01)
            assert 's1' not in second['steps_done']
            report['states']['independent_second_first'] = second
            verdicts['s1'] = 'yes'
            both = live.drive(lambda s: {'s1', 's2'} <= set(s['steps_done']), max_frames=1400, pause_s=.01)
            live.send(dict(type='step_ack', step_id='s3'))
            approved = live.until(lambda m: m['type'] == 'state' and 's3' in m['steps_user_done'])
            report['states']['user_approval'] = approved
            engine = live_app.state.store.sessions[session['session_id']].engine
            facts_before = {key: asdict(value) for key, value in engine.guide.graph_facts.items()}
            verdicts.update(s1='unsure', s2='unsure')
            hidden = live.drive(lambda s: s['step_statuses'].get('s1') == 'unsure' and s['step_statuses'].get('s2') == 'unsure', max_frames=1400, pause_s=.01)
            assert set(hidden['steps_done']) == {'s1', 's2', 's3'}
            assert {key: asdict(value) for key, value in engine.guide.graph_facts.items()} == facts_before
            report['states']['occluded_facts_preserved'] = hidden
            verdicts['s1'] = 'no'
            revoked = live.drive(lambda s: s['step_statuses'].get('s1') == 'no' and 's1' not in s['steps_done'], max_frames=1400, pause_s=.01)
            assert revoked['steps_done'] == ['s2'] and revoked['steps_user_done'] == []
            assert revoked['step_statuses']['s3'] == 'unsure'
            report['states']['contrary_no_withdraws_dependent_approval'] = revoked
            report['accepted_facts_before_revocation'] = facts_before
            assert facts_before['s1']['provider'] == 'deepseek' and facts_before['s1']['accepted_via'] == 'upper.bound'
            assert facts_before['s3']['source'] == 'user' and facts_before['s3']['accepted_via'] == 'step_ack'
            live.send(dict(type='stop'))
            live.until(lambda m: m['type'] == 'state' and m['phase'] == 'idle')
        ended = client.delete('/v1/sessions/' + session['session_id'], headers={'authorization': 'Bearer ' + session['token']})
        assert ended.status_code == 200
    report['routes'] = dict(Counter(path for path, _, _ in transport.calls))
    report['non_2xx'] = [(p, status) for p, status, _ in transport.calls if status >= 300]
    assert not report['non_2xx'], report['non_2xx']
    for route in ('/api/session', '/api/guide/plan', '/api/guide/plan/approve', '/api/guide/follow', '/api/guide/confirm', '/api/track/control', '/api/track/frame'):
        assert report['routes'][route] > 0, route
    report['canned_calls'] = dict(issuer=len(issuer.requests), clef=len(clef.requests), tracker=len(tracker.calls))
    report['source_sha256'] = {(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'live-api/' + str(p.relative_to(LIVE))): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / 'backend/app/intent.py', ROOT / 'backend/app/guide_contracts.py',
                  LIVE / 'src/synoptics_live/real_engine.py', LIVE / 'src/synoptics_live/contracts.py']}
    report['ok'] = True
    target = ROOT / 'verification/actual-asgi-websocket-smoke.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('states', 'accepted_facts_before_revocation')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
