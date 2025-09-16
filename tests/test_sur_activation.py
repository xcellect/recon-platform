import sys
import os

# Ensure the recon_engine package is importable
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNState


def _setup_graph(measurement_value: float, weight: float = 1.0) -> ReCoNGraph:
    g = ReCoNGraph()
    g.add_node("parent", node_type="script")
    g.add_node("term", node_type="terminal")
    # sub link creates reciprocal sur link with same weight
    g.add_link("parent", "term", "sub", weight=weight)

    # Set terminal measurement to a fixed value
    term = g.get_node("term")
    term.measurement_fn = lambda env=None: measurement_value

    return g


def _run_steps(g: ReCoNGraph, steps: int):
    for _ in range(steps):
        g.propagate_step()


def test_sur_activation_uses_measurement_magnitude():
    # measurement above threshold, weight 1.0 → parent should receive ~0.9
    g = _setup_graph(measurement_value=0.9, weight=1.0)
    g.request_root("parent")

    # Step 1: parent ACTIVE, requests child
    # Step 2: child measures and confirms, message queued
    # Step 3: parent receives sur confirm with activation ~= 0.9
    _run_steps(g, 3)

    parent = g.get_node("parent")
    assert parent.state == ReCoNState.TRUE
    assert abs(float(parent.activation) - 0.9) < 1e-6


def test_sur_activation_scaled_by_link_weight():
    # measurement 0.9, weight 0.5 → parent receives 0.45 (< threshold) and stays WAITING
    g = _setup_graph(measurement_value=0.9, weight=0.5)
    g.request_root("parent")

    _run_steps(g, 3)

    parent = g.get_node("parent")
    # Below threshold → not TRUE yet, should be WAITING
    assert parent.state == ReCoNState.WAITING
    assert abs(float(parent.activation) - 0.45) < 1e-6


