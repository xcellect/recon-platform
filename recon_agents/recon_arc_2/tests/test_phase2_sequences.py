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


def test_sequence_only_last_confirms_parent(hm):
    # Build a sequence hypothesis with indices [0, 1]
    seq = hm.create_sequence_hypothesis([0, 1])

    # Request the sequence
    hm.request_hypothesis_test(seq.id)
    hm.propagate_step()

    # First child confirms; sequence should advance but parent (seq) should NOT confirm yet
    hm.set_action_measurement(0, True)
    for _ in range(2):
        hm.propagate_step()
    assert seq.state != ReCoNState.CONFIRMED

    # Second child confirms; now parent sequence should confirm
    hm.set_action_measurement(1, True)
    for _ in range(2):
        hm.propagate_step()
    assert seq.state == ReCoNState.CONFIRMED

