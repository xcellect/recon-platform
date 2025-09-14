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


def test_region_priors_select_highest_first(hm):
    r_low, r_high = 100, 101
    hm.create_action_hypothesis(r_low, 0.5, grid(0))
    hm.create_action_hypothesis(r_high, 0.5, grid(0))
    # α_valid favors r_high; α_value equal
    hm.set_alpha_valid({r_low: 0.2, r_high: 0.8})
    hm.set_alpha_value({r_low: 0.0, r_high: 0.0})
    alt = hm.create_alternatives_hypothesis([r_low, r_high])
    # Terminals succeed for both
    hm.set_terminal_measurement(r_low, True)
    hm.set_terminal_measurement(r_high, True)
    hm.request_hypothesis_test(alt)
    progressed_high_first = False
    for _ in range(10):
        hm.propagate_step()
        hs = hm.action_hypotheses[r_high].state
        ls = hm.action_hypotheses[r_low].state
        if hs in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED) and \
           ls in (ReCoNState.INACTIVE, ReCoNState.SUPPRESSED):
            progressed_high_first = True
            break
    assert progressed_high_first is True


def test_region_no_immediate_repeat_on_fail(hm):
    r1, r2 = 110, 111
    hm.cooldown_steps = 3
    hm.create_action_hypothesis(r1, 0.5, grid(0))
    hm.create_action_hypothesis(r2, 0.5, grid(0))
    hm.set_alpha_valid({r1: 0.9, r2: 0.1})
    hm.set_alpha_value({r1: 0.0, r2: 0.0})
    alt = hm.create_alternatives_hypothesis([r1, r2])
    hm.request_hypothesis_test(alt)
    # First, r1 should be chosen; mark it failed which starts cooldown
    hm.set_terminal_measurement(r1, False)
    # Allow r2 to succeed when it gets scheduled
    hm.set_terminal_measurement(r2, True)
    for _ in range(2):
        hm.propagate_step()
    # During cooldown, r2 should progress while r1 stays suppressed/inactive
    avoided_repeat = False
    for _ in range(hm.cooldown_steps + 3):
        hm.propagate_step()
        s1 = hm.action_hypotheses[r1].state
        s2 = hm.action_hypotheses[r2].state
        if s2 in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED) and \
           s1 in (ReCoNState.INACTIVE, ReCoNState.SUPPRESSED, ReCoNState.FAILED):
            avoided_repeat = True
            break
    assert avoided_repeat is True
