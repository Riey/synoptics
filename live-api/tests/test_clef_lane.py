"""The Clef follower is a loopback model: it must be paced as the local lane, not as a paid remote call.

Ports: ``useIntentLoop.ts:640-643`` (``provider === 'local' || provider === 'clef' ? 'local' : 'remote'``)
and ``triggers.ts:212`` (the same local/clef test picks the heartbeat interval). The SPA client follows both;
the Live engine classified only ``local`` as local, so a Clef run was paced at the 2200 ms remote follow
floor, forced the 1200 ms paid-call spacing on every later dispatch, and counted unbilled calls against the
paid budget.
"""

from __future__ import annotations

from synoptics_live import guide_policy as gp
from synoptics_live.contracts import Box

from test_real_engine import H


def test_clef_heartbeat_uses_the_local_interval():
    """triggers.ts:212-215: heartbeatLocalMs (4 s) for both loopback followers, heartbeatRemoteMs (8 s) else."""
    obs = gp.TrackObservation("r", "t", 1, "tracking", Box(x=0.1, y=0.1, width=0.2, height=0.2))
    st, fired = gp.step_triggers(gp.INITIAL_TRIGGER_STATE, observation=obs, now_ms=0, fence_key="k", accepted=None,
                                 follow_provider="clef")
    assert fired == ["acquired"]
    st, fired = gp.step_triggers(st, observation=obs, now_ms=4000, fence_key="k", accepted=None,
                                 follow_provider="clef")
    assert fired == ["heartbeat"]


def test_clef_follow_dispatches_on_the_local_lane():
    """A Clef follow is unbilled: the 1.0 s local floor, no paid spacing, nothing taken from the paid budget."""
    engine = H().engine
    engine.follow_provider = "clef"
    clef_lane = engine._follow_lane()
    assert clef_lane == "local"
    engine.follow_provider = "local"
    assert engine._follow_lane() == "local"
    engine.follow_provider = "deepseek"
    assert engine._follow_lane() == "remote"

    call = gp.PendingCall("follow", "heartbeat", clef_lane)
    state = gp.mark_dispatched(gp.enqueue(gp.INITIAL_GATE_STATE, call), call, 0, gp.GATE_CONFIG)
    assert state.remote_times == () and state.last_remote_at == gp.INITIAL_GATE_STATE.last_remote_at
    state = gp.enqueue(gp.mark_settled(state, "follow"), gp.PendingCall("follow", "heartbeat", "local"))
    assert gp.decide_dispatch(state, "follow", 999, gp.GATE_CONFIG).kind == "wait"        # local floor, not 2200
    assert gp.decide_dispatch(state, "follow", 1000, gp.GATE_CONFIG).kind == "dispatch"
