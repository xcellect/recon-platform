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


def grid_left_right():
    g = []
    for y in range(64):
        row = []
        for x in range(64):
            row.append(0 if x < 16 else 1)
        g.append(row)
    return g


@pytest.fixture
def agent():
    return ReCoNArc2Agent(agent_id='recon_arc_2_test', game_id='test')


def test_choose_action_with_coordinates_click_only(agent):
    f = DummyFrame(grid_left_right(), available=[_GA(6)])
    action_idx, coords = agent.choose_action_with_coordinates([], f)
    assert action_idx == 5
    assert coords is not None
    x, y = coords
    assert 0 <= x < 64 and 0 <= y < 64


def test_choose_action_with_coordinates_mixed(agent):
    f = DummyFrame(grid_left_right(), available=[_GA(2), _GA(6)])
    seen_click = False
    for _ in range(10):
        action_idx, coords = agent.choose_action_with_coordinates([], f)
        if action_idx == 5:
            assert coords is not None
            x, y = coords
            assert x >= 16
            seen_click = True
            break
    assert seen_click


