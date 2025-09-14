import sys
import os

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


class _GA:
    """Minimal stand-in for GameAction items (only .value is used)."""
    def __init__(self, value: int):
        self.value = value


class _GS:
    NOT_FINISHED = 'NOT_FINISHED'
    NOT_PLAYED = 'NOT_PLAYED'
    WIN = 'WIN'
    GAME_OVER = 'GAME_OVER'


class DummyFrame:
    def __init__(self, frame, state=_GS.NOT_FINISHED, score=0, available=None):
        self.frame = frame
        self.state = state
        self.score = score
        self.game_id = 'test'
        self.available_actions = available or []


def grid(val=0):
    # 64x64 grid
    return [[val for _ in range(64)] for _ in range(64)]


@pytest.fixture
def agent():
    return ReCoNArc2Agent(agent_id='recon_arc_2_test', game_id='test')


def test_does_not_emit_action6_when_unavailable(agent):
    f = DummyFrame(frame=grid(), available=[_GA(1), _GA(2)])
    action_idx = agent.choose_action([], f)
    assert action_idx in [0, 1, 2, 3, 4, 5]
    # Acceptance: ACTION6 should not be chosen when not available
    assert action_idx != 5


def test_respects_available_actions_subset(agent):
    avail = [_GA(2), _GA(4)]
    f = DummyFrame(frame=grid(), available=avail)
    # Run a few times to account for stochasticity
    for _ in range(10):
        action_idx = agent.choose_action([], f)
        # Must be mapped to one of the allowed simple actions
        assert action_idx in [1, 3]


def test_no_ops_reduction_sanity(agent):
    # Two identical frames simulate no change; ensure feedback path doesn't crash
    f1 = DummyFrame(frame=grid(1), available=[_GA(1), _GA(3)])
    _ = agent.choose_action([], f1)
    f2 = DummyFrame(frame=grid(1), available=[_GA(1), _GA(3)])
    # Simulate next step; should process feedback without errors
    _ = agent.choose_action([], f2)


