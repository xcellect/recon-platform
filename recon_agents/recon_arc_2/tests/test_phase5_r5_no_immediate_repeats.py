import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.hypothesis import HypothesisManager
from recon_engine.node import ReCoNState


@pytest.fixture
def hm():
    return HypothesisManager()


def test_region_failure_not_immediately_retried(hm):
    r1 = 300
    r2 = 301

    # Create two region actions
    hm.create_action_hypothesis(r1, 0.5, None)
    hm.create_action_hypothesis(r2, 0.5, None)

    # Equal value ordering to avoid por bias; favor r2 by alpha_valid
    hm.set_alpha_value({r1: 0.0, r2: 0.0})
    hm.set_alpha_valid({r1: 0.2, r2: 0.8})

    alt = hm.create_alternatives_hypothesis([r1, r2])

    # Make r1 fail, r2 succeed
    hm.set_terminal_measurement(r1, False)
    hm.set_terminal_measurement(r2, True)

    hm.request_hypothesis_test(alt)

    # Run until r1 has failed at least once
    seen_r1_failed = False
    for _ in range(10):
        hm.propagate_step()
        if hm.action_hypotheses[r1].state == ReCoNState.FAILED:
            seen_r1_failed = True
            break

    assert seen_r1_failed is True

    # During cooldown, r1 should not be retried; r2 should progress
    r2_progressed = False
    for _ in range(max(1, hm.cooldown_steps - 1)):
        hm.propagate_step()
        assert hm.action_hypotheses[r1].state in (ReCoNState.FAILED, ReCoNState.SUPPRESSED, ReCoNState.INACTIVE)
        if hm.action_hypotheses[r2].state in (ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED):
            r2_progressed = True

    assert r2_progressed is True
