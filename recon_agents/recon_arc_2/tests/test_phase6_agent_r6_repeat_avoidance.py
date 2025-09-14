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


def make_three_regions() -> np.ndarray:
    f = np.zeros((64, 64), dtype=int)
    # Region A small
    f[5:8, 5:8] = 1
    # Region B medium (should be chosen on retry)
    f[20:26, 20:26] = 2
    # Region C large (will fail first)
    f[45:58, 45:58] = 3
    return f


@pytest.mark.parametrize("r6_flag", ["1"])  # enable R6
@pytest.mark.parametrize("fail_first", ["1"])  # fail top-prior region first
def test_agent_r6_avoids_immediate_repeat(monkeypatch, r6_flag, fail_first):
    monkeypatch.setenv("RECON_ARC2_R6", r6_flag)
    monkeypatch.setenv("RECON_ARC2_R6_FAIL_FIRST", fail_first)

    agent = ReCoNArc2Agent()
    agent.top_k_click_regions = 3

    frame = make_three_regions()

    # First proposal will target the large region centroid (but considered failed by flag)
    c_lx, c_ly = centroid_of(3, frame)
    first = agent.propose_click_coordinates(frame)
    assert first == (c_lx, c_ly)

    # Second proposal should avoid immediate repeat and choose the next best (medium)
    # Remove fail flag to allow success on second attempt
    monkeypatch.setenv("RECON_ARC2_R6_FAIL_FIRST", "0")
    c_mx, c_my = centroid_of(2, frame)
    second = agent.propose_click_coordinates(frame)
    assert second == (c_mx, c_my)
