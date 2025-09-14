import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.hypothesis import HypothesisManager
from recon_engine.node import ReCoNState


@pytest.fixture
def hm():
    return HypothesisManager()


def test_availability_gates_requests_on_sub(hm):
    # Create a small sequence of actions 0 -> 1
    seq = hm.create_sequence_hypothesis([0, 1])

    # Disallow action 1
    hm.set_available_actions([0])

    # Request the sequence root
    hm.request_hypothesis_test(seq.id)

    # Let requests and gating propagate
    for _ in range(8):
        hm.propagate_step()

    a0 = hm.action_hypotheses[0]
    a1 = hm.action_hypotheses[1]

    # a1 should be por-inhibited due to availability gating
    assert a1.state == ReCoNState.SUPPRESSED
    # a0 should not be suppressed by availability; allow any normal evolution
    assert a0.state != ReCoNState.SUPPRESSED

    # Now allow action 1 and ensure it can be requested/progress
    hm.set_available_actions([0, 1])
    for _ in range(8):
        hm.propagate_step()

    assert a1.state in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED, ReCoNState.FAILED)


def test_noop_suppression_with_decay(hm):
    # Single action hypothesis 5
    ah = hm.create_action_hypothesis(5, 0.5, None)

    # Request the action
    hm.request_hypothesis_test(ah.id)

    # Let it get requested/active
    for _ in range(4):
        hm.propagate_step()

    # First attempt fails (no frame change)
    hm.set_terminal_measurement(5, False)
    for _ in range(3):
        hm.propagate_step()

    # During cooldown, even if we set success, the action should not confirm
    hm.set_terminal_measurement(5, True)
    for _ in range(max(1, hm.cooldown_steps - 1)):
        hm.propagate_step()
        assert ah.state != ReCoNState.CONFIRMED
        assert ah.state != ReCoNState.TRUE

    # After cooldown expires, it should be able to progress and confirm
    for _ in range(hm.cooldown_steps + 3):
        hm.propagate_step()
        if ah.state in (ReCoNState.TRUE, ReCoNState.CONFIRMED):
            break

    assert ah.state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)
