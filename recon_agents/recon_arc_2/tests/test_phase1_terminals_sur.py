import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.hypothesis import HypothesisManager
from recon_engine.node import ReCoNState


class _GA:
    def __init__(self, value: int):
        self.value = value


def grid(val=0):
    return [[val for _ in range(64)] for _ in range(64)]


@pytest.fixture
def hm():
    return HypothesisManager()


def test_terminal_confirm_flows_via_sur(hm):
    hyp = hm.create_action_hypothesis(1, 0.9, grid(0))
    # Request hypothesis test so terminal is requested via sub
    hm.request_hypothesis_test(hyp.id)
    hm.propagate_step()
    # Set terminal measurement to confirm
    hm.set_terminal_measurement(1, True)
    # Propagate to allow sur confirmation
    for _ in range(3):
        hm.propagate_step()
    assert hyp.state == ReCoNState.CONFIRMED


def test_terminal_fail_flows_via_sur(hm):
    hyp = hm.create_action_hypothesis(2, 0.9, grid(0))
    hm.request_hypothesis_test(hyp.id)
    hm.propagate_step()
    hm.set_terminal_measurement(2, False)
    for _ in range(3):
        hm.propagate_step()
    assert hyp.state in (ReCoNState.FAILED, ReCoNState.WAITING)

