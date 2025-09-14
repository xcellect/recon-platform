import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.hypothesis import HypothesisManager
from recon_engine.node import ReCoNState


@pytest.fixture
def hm():
    return HypothesisManager()


def test_alpha_valid_delays_request(hm):
    # Two independent actions
    a_low = hm.create_action_hypothesis(10, 0.5, None)  # lower alpha_valid -> more delay
    a_high = hm.create_action_hypothesis(11, 0.5, None) # higher alpha_valid -> less delay

    hm.set_alpha_valid({10: 0.0, 11: 1.0})

    # Request both via root
    hm.request_hypothesis_test('hypothesis_root')

    # Track first step where each action reaches REQUESTED or beyond
    first_high = None
    first_low = None

    for step in range(1, 16):
        hm.propagate_step()
        if first_high is None and hm.action_hypotheses[11].state in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED):
            first_high = step
        if first_low is None and hm.action_hypotheses[10].state in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED):
            first_low = step
        if first_high is not None and first_low is not None:
            break

    assert first_high is not None
    assert first_low is not None
    assert first_high <= first_low  # high-prior action is requested no later than low


def test_alpha_value_orders_alternatives(hm):
    # Create alternatives 20 vs 21, with alpha_value favoring 21
    hm.set_alpha_value({20: 0.1, 21: 0.9})
    alt = hm.create_alternatives_hypothesis([20, 21])

    # Request alternatives parent
    hm.request_hypothesis_test(alt)

    # Run a few steps to see which activates first
    first_active = None
    for _ in range(12):
        hm.propagate_step()
        a20 = hm.action_hypotheses[20].state
        a21 = hm.action_hypotheses[21].state
        if a20 in (ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED) and first_active is None:
            first_active = 20
        if a21 in (ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED) and first_active is None:
            first_active = 21
        if first_active is not None:
            break

    assert first_active == 21
