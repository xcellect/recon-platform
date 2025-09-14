import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


def grid_left_right():
    # Left half (0), right half (1)
    g = []
    for y in range(64):
        row = []
        for x in range(64):
            row.append(0 if x < 16 else 1)  # Make right region larger
        g.append(row)
    return g


@pytest.fixture
def agent():
    return ReCoNArc2Agent(agent_id='recon_arc_2_test', game_id='test')


def test_propose_click_coordinates_center_of_largest_region(agent):
    frame = grid_left_right()
    x, y = agent.propose_click_coordinates(frame)
    assert 0 <= x < 64 and 0 <= y < 64
    # Largest region is on the right; x should be >= 16
    assert x >= 16


