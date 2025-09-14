import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.hypothesis import HypothesisManager
from recon_engine.node import ReCoNState


def grid(val=0):
    return [[val for _ in range(64)] for _ in range(64)]


@pytest.fixture
def hm():
    return HypothesisManager()


def test_update_hypothesis_result_routes_via_terminal(hm):
    hyp = hm.create_action_hypothesis(2, 0.5, grid(0))
    hm.request_hypothesis_test(hyp.id)
    # Terminal measurement path: set measurement and not directly change state
    hm.set_terminal_measurement(2, True)
    # Allow propagation
    for _ in range(3):
        hm.propagate_step()
    assert hyp.state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)

