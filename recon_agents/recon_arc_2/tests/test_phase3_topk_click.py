import sys

sys.path.insert(0, '/workspace/recon-platform')

import numpy as np
import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


def grid_three_regions():
    g = np.zeros((64, 64), dtype=np.uint8)
    # Small left block
    g[:, :8] = 1
    # Medium middle block
    g[:, 24:40] = 2
    # Large right block
    g[:, 48:] = 3
    return g


@pytest.fixture
def agent():
    ag = ReCoNArc2Agent(agent_id='recon_arc_2_test', game_id='test')
    ag.top_k_click_regions = 2
    return ag


def test_topk_click_candidates(agent):
    frame = grid_three_regions()
    # Expect click proposal to be within one of the top-2 largest regions (middle or right)
    x, y = agent.propose_click_coordinates(frame)
    assert 0 <= x < 64 and 0 <= y < 64
    assert (24 <= x < 40) or (48 <= x < 64)


