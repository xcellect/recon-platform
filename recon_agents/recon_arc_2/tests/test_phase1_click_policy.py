import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


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


def grid_with_two_regions():
    # 64x64; left half zeros, right half ones (region sizes equal)
    g = []
    for y in range(64):
        row = []
        for x in range(64):
            row.append(0 if x < 32 else 1)
        g.append(row)
    return g


@pytest.fixture
def agent():
    return ReCoNArc2Agent(agent_id='recon_arc_2_test', game_id='test')


def test_emits_action6_with_coordinates_when_only_click_available(agent):
    f = DummyFrame(frame=grid_with_two_regions(), available=[_GA(6)])
    action_idx = agent.choose_action([], f)
    # Should choose ACTION6 (index 5) once click policy exists
    assert action_idx == 5


def test_click_policy_respects_available_actions_mixture(agent):
    # When both simple and ACTION6 available, ensure ACTION6 can be selected and policy runs
    f = DummyFrame(frame=grid_with_two_regions(), available=[_GA(2), _GA(6)])
    # Try multiple times due to stochastic elements
    seen_click = False
    seen_simple = False
    for _ in range(20):
        idx = agent.choose_action([], f)
        if idx == 5:
            seen_click = True
        if idx in [0, 1, 2, 3, 4]:
            seen_simple = True
        if seen_click and seen_simple:
            break
    assert seen_click, 'Expected ACTION6 to be possible with click policy'

