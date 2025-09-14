import sys

sys.path.insert(0, '/workspace/recon-platform')

import numpy as np
import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent
from recon_agents.recon_arc_2.perception import ChangePredictorTrainer


class CountingPredictor:
    def __init__(self, probs):
        self._probs = np.array(probs, dtype=float)
        self.calls = 0

    def predict_change_probabilities(self, frame):
        self.calls += 1
        return self._probs


def grid(val=0):
    return np.array([[val for _ in range(64)] for _ in range(64)], dtype=np.uint8)


@pytest.fixture
def agent():
    ag = ReCoNArc2Agent(agent_id='recon_arc_2_test', game_id='test')
    return ag


class _GA:
    def __init__(self, value: int):
        self.value = value


class DummyFrame:
    def __init__(self, frame, available=None):
        self.frame = frame
        self.state = 'NOT_FINISHED'
        self.score = 0
        self.game_id = 'test'
        self.available_actions = available or []


def test_prediction_cache_reuses_probs(agent):
    frame = grid(3)
    agent.change_predictor = CountingPredictor([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    f = DummyFrame(frame, available=[_GA(1), _GA(2)])

    # First call computes
    _ = agent.choose_action([], f)
    # Second call with same frame should hit cache
    _ = agent.choose_action([], f)
    assert agent.change_predictor.calls == 1


def test_trainer_experience_dedup():
    from recon_agents.recon_arc_2.perception import ChangePredictor
    model = ChangePredictor()
    trainer = ChangePredictorTrainer(model)
    frame = grid(1)

    # Add same (state, action) twice
    trainer.add_experience(frame, 2, False)
    trainer.add_experience(frame, 2, True)

    # Expect single entry retained (latest outcome ok)
    assert len(trainer.experience_buffer) == 1

