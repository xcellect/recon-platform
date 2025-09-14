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


def test_confirmed_state_persists_across_absent_request_briefly(hm):
    hyp = hm.create_action_hypothesis(3, 0.9, grid(0))
    hm.request_hypothesis_test(hyp.id)
    hm.propagate_step()
    hm.set_terminal_measurement(3, True)
    for _ in range(2):
        hm.propagate_step()
    assert hyp.state == ReCoNState.CONFIRMED
    # Remove request by not calling request_hypothesis_test again
    for _ in range(2):
        hm.propagate_step()
    # Should still not reset immediately (persistence)
    assert hyp.state in (ReCoNState.CONFIRMED, ReCoNState.TRUE)

