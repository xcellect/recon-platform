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

    # First child confirms via terminal; parent should not confirm yet
    hm.set_terminal_measurement(0, True)
    for _ in range(3):
        hm.propagate_step()
    assert seq.state != ReCoNState.CONFIRMED

    # Second child confirms via terminal; now parent should reach TRUE/CONFIRMED
    hm.set_terminal_measurement(1, True)
    for _ in range(6):
        hm.propagate_step()
    assert seq.state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)

