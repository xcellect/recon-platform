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


def test_r6_on_off_same_coordinates(monkeypatch):
    # R6 off
    monkeypatch.delenv("RECON_ARC2_R6", raising=False)
    agent_off = ReCoNArc2Agent()
    agent_off.top_k_click_regions = 3
    frame = make_typical_frame()
    expected = agent_off.propose_click_coordinates(frame)

    # R6 on
    monkeypatch.setenv("RECON_ARC2_R6", "1")
    agent_on = ReCoNArc2Agent()
    agent_on.top_k_click_regions = 3
    got = agent_on.propose_click_coordinates(frame)

    assert got == expected
