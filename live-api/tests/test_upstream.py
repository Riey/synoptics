"""upstream.py: Origin, SSE parsing at EOF and unknown events."""

from __future__ import annotations

import asyncio
import json

import httpx

from synoptics_live.upstream import UpstreamClient, origin_of


def test_origin_matches_the_host_httpx_sends():
    assert origin_of("http://Example.test:80/x") == "http://example.test"
    assert origin_of("http://127.0.0.1:8045") == "http://127.0.0.1:8045"


def _plan_client(body: str) -> UpstreamClient:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    client = UpstreamClient("http://up.test", transport=httpx.MockTransport(handle))
    client.session_id, client._token = "s", "t"
    return client


FINAL = {"plan_id": "p", "plan_revision": 1, "steps": [], "selection": {"status": "no_target"},
         "needs_clarification": False}


def test_sse_final_without_trailing_blank_line_and_unknown_events():
    body = "event: keepalive\ndata: ping\n\nevent: partial\ndata: {\"target\": \"cup\"}\n\n" \
           f"event: final\ndata: {json.dumps(FINAL)}"
    partials = []
    out = asyncio.run(_plan_client(body).plan({}, lambda k, v: partials.append((k, v))))
    assert out["plan_id"] == "p" and partials == [("target", "cup")]
