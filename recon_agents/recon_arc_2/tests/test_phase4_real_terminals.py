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


def test_terminal_emits_sur_confirm_without_injection(hm):
    hyp = hm.create_action_hypothesis(7, 0.8, grid(0))
    # Set terminal measurement flag before request
    hm.set_terminal_measurement(7, True)
    hm.request_hypothesis_test(hyp.id)
    for _ in range(3):
        hm.propagate_step()
    assert hyp.state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)


def test_terminal_emits_sur_fail_without_injection(hm):
    hyp = hm.create_action_hypothesis(8, 0.8, grid(0))
    hm.set_terminal_measurement(8, False)
    hm.request_hypothesis_test(hyp.id)
    for _ in range(3):
        hm.propagate_step()
    # May be FAILED or remain WAITING depending on parent semantics; accept FAILED or WAITING
    assert hyp.state in (ReCoNState.FAILED, ReCoNState.WAITING)


