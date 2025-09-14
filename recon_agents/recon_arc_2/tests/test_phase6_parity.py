import sys
import os

sys.path.insert(0, '/workspace/recon-platform')

import numpy as np
import pytest

from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


def centroid_of(mask_val: int, frame: np.ndarray) -> tuple:
    ys, xs = np.where(frame == mask_val)
    if len(xs) == 0:
        return (32, 32)
    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    return (cx, cy)


def make_typical_frame() -> np.ndarray:
    f = np.zeros((64, 64), dtype=int)
    # Add a few regions of varying sizes
    f[3:7, 3:7] = 1
    f[20:28, 20:28] = 2
    f[45:59, 45:59] = 3
    return f


def test_recon_path_is_single_and_deterministic():
    agent_a = ReCoNArc2Agent()
    agent_b = ReCoNArc2Agent()
    agent_a.top_k_click_regions = 3
    agent_b.top_k_click_regions = 3
    frame = make_typical_frame()
    expected = agent_a.propose_click_coordinates(frame)
    got = agent_b.propose_click_coordinates(frame)
    assert got == expected
