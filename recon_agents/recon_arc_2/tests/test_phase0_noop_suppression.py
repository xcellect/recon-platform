import sys

sys.path.insert(0, '/workspace/recon-platform')

import numpy as np
import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


class _GA:
    def __init__(self, value: int):
        self.value = value


class DummyPredictor:
    def __init__(self, logits):
        self._probs = np.array(logits, dtype=float)

    def predict_change_probabilities(self, frame):
        return self._probs


class DummyFrame:
    def __init__(self, frame, available=None):
        self.frame = frame
        self.state = 'NOT_FINISHED'
        self.score = 0
        self.game_id = 'test'
        self.available_actions = available or []


def grid(val=0):
    return [[val for _ in range(64)] for _ in range(64)]


@pytest.fixture
def agent():
    ag = ReCoNArc2Agent(agent_id='recon_arc_2_test', game_id='test')
    # Make ACTION1 highest by default
    ag.change_predictor = DummyPredictor([0.9, 0.1, 0.1, 0.1, 0.1, 0.0])
    # Reduce suppression horizon for test speed
    ag.noop_suppression_steps = 2
    return ag


def test_noop_suppression_avoids_immediate_retry(agent):
    f = DummyFrame(grid(1), available=[_GA(1), _GA(2)])

    # First selection should pick ACTION1 (idx 0)
    a0 = agent.choose_action([], f)
    assert a0 == 0

    # Next call sees same frame (no change) and should suppress idx 0, so pick idx 1
    a1 = agent.choose_action([], f)
    assert a1 == 1


