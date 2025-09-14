import sys

sys.path.insert(0, '/workspace/recon-platform')

import inspect
import pytest

from recon_agents.recon_arc_2.hypothesis import HypothesisManager, ActionHypothesis, SequenceHypothesis
from recon_engine.node import ReCoNState


def grid(val=0):
    return [[val for _ in range(64)] for _ in range(64)]


@pytest.fixture
def hm():
    return HypothesisManager()


def test_no_custom_process_message_overrides():
    # Ensure classes do not implement custom process_message; must rely on engine FSM
    # If attribute is coming from ReCoNNode, the function object should be the same
    from recon_engine.node import ReCoNNode

    assert "process_message" not in ActionHypothesis.__dict__, "ActionHypothesis must not override process_message"
    assert "process_message" not in SequenceHypothesis.__dict__, "SequenceHypothesis must not override process_message"

    # And inherited attribute should point to base implementation if accessed
    assert getattr(ActionHypothesis, 'process_message', getattr(ReCoNNode, 'process_message', None)) is getattr(ReCoNNode, 'process_message', None)
    assert getattr(SequenceHypothesis, 'process_message', getattr(ReCoNNode, 'process_message', None)) is getattr(ReCoNNode, 'process_message', None)


def test_sequence_confirms_only_on_last_child_via_terminals(hm):
    # Build a simple two-step sequence
    seq = hm.create_sequence_hypothesis([0, 1])

    # Set both terminal measurements ahead of time to avoid early FAIL defaults
    hm.set_terminal_measurement(0, True)
    hm.set_terminal_measurement(1, True)

    # Request the sequence
    hm.request_hypothesis_test(seq.id)

    # Early propagation: parent should not confirm immediately
    for _ in range(2):
        hm.propagate_step()
    assert seq.state != ReCoNState.CONFIRMED

    # Allow sequence to advance to second child and confirm
    for _ in range(10):
        hm.propagate_step()

    assert seq.state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)
