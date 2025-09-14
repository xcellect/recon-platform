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


def test_cooldown_gate_prevents_true_until_decay(hm):
    a = hm.create_action_hypothesis(1, 0.5, grid(0))
    hm.request_hypothesis_test(a.id)
    for _ in range(2):
        hm.propagate_step()
    # Fail once → starts cooldown gate
    hm.set_terminal_measurement(1, False)
    # Even with success set during cooldown, must not reach TRUE/CONFIRMED
    hm.set_terminal_measurement(1, True)
    for _ in range(max(1, hm.cooldown_steps - 1)):
        hm.propagate_step()
        assert hm.action_hypotheses[1].state not in (ReCoNState.TRUE, ReCoNState.CONFIRMED)
    # After cooldown clears and resume, it can progress
    for _ in range(hm.cooldown_steps + 3):
        hm.propagate_step()
        if hm.action_hypotheses[1].state in (ReCoNState.TRUE, ReCoNState.CONFIRMED):
            break
    assert hm.action_hypotheses[1].state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)


def test_alpha_valid_gate_delays_lower_priority(hm):
    hm.create_action_hypothesis(10, 0.5, grid(0))  # low
    hm.create_action_hypothesis(11, 0.5, grid(0))  # high
    hm.set_alpha_valid({10: 0.2, 11: 0.9})
    alt = hm.create_alternatives_hypothesis([10, 11])
    hm.request_hypothesis_test(alt)
    # High should progress first while low is still inactive/suppressed
    observed = False
    for _ in range(8):
        hm.propagate_step()
        hs = hm.action_hypotheses[11].state
        ls = hm.action_hypotheses[10].state
        if hs in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED) and \
           ls in (ReCoNState.INACTIVE, ReCoNState.SUPPRESSED):
            observed = True
            break
    assert observed is True

