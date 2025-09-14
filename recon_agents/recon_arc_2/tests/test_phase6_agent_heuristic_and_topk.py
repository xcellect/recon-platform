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


def centroid_of(mask_val: int, frame: np.ndarray) -> tuple:
    ys, xs = np.where(frame == mask_val)
    if len(xs) == 0:
        return (32, 32)
    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    return (cx, cy)


def make_three_regions() -> np.ndarray:
    f = np.zeros((64, 64), dtype=int)
    # Region A small
    f[5:8, 5:8] = 1
    # Region B medium
    f[20:26, 20:26] = 2
    # Region C large (should be chosen)
    f[45:58, 45:58] = 3
    return f


def make_two_regions() -> np.ndarray:
    f = np.zeros((64, 64), dtype=int)
    # Small
    f[2:6, 2:6] = 1
    # Large
    f[40:56, 40:56] = 2
    return f


def test_agent_flag_off_uses_heuristic(monkeypatch):
    monkeypatch.delenv("RECON_ARC2_R6", raising=False)
    agent = ReCoNArc2Agent()
    agent.top_k_click_regions = 2
    frame = make_two_regions()
    expected = centroid_of(2, frame)
    coords = agent.propose_click_coordinates(frame)
    assert coords == expected


def test_agent_r6_topk_prefers_highest_prior(monkeypatch):
    monkeypatch.setenv("RECON_ARC2_R6", "1")
    agent = ReCoNArc2Agent()
    agent.top_k_click_regions = 3
    frame = make_three_regions()
    expected = centroid_of(3, frame)
    coords = agent.propose_click_coordinates(frame)
    assert coords == expected
