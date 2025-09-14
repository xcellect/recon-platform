import sys
import os

sys.path.insert(0, '/workspace/recon-platform')

import numpy as np
import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


class DummyFrame:
    def __init__(self, frame, available_actions=None, score=0, game_id="test"):
        self.frame = frame
        self.available_actions = available_actions or []
        self.score = score
        self.game_id = game_id
        self.state = None


def make_frame_with_two_regions() -> np.ndarray:
    # 64x64 with background 0; add small region (value 1) and larger region (value 2)
    f = np.zeros((64, 64), dtype=int)
    # Small region at top-left (3x3)
    f[2:5, 2:5] = 1
    # Large region at bottom-right (8x8)
    f[48:56, 48:56] = 2
    return f


def centroid_of(mask_val: int, frame: np.ndarray) -> tuple:
    ys, xs = np.where(frame == mask_val)
    if len(xs) == 0:
        return (32, 32)
    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    return (cx, cy)


@pytest.mark.parametrize("r6_flag", ["1"])  # ensure R6 path is enabled
def test_agent_proposes_higher_prior_region_coordinates(monkeypatch, r6_flag):
    monkeypatch.setenv("RECON_ARC2_R6", r6_flag)
    agent = ReCoNArc2Agent()
    agent.top_k_click_regions = 2

    frame = make_frame_with_two_regions()
    expected_cx, expected_cy = centroid_of(2, frame)

    # Compose a dummy frame wrapper
    df = DummyFrame(frame=frame, available_actions=[type("A", (), {"value": 6})()])

    coords = agent.propose_click_coordinates(frame)

    assert coords == (expected_cx, expected_cy)
