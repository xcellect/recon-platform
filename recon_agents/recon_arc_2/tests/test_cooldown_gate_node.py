"""
Test CooldownGateNode - ReCoN-native cooldown with gen persistence.

CooldownGateNode should:
- Use gen self-loop for natural decay
- Emit POR to inhibit action while activation > 0
- Be triggered by setting activation (cooldown steps)
- Decay naturally through gen arithmetic (no Python timers)
"""

import sys
sys.path.insert(0, '/workspace/recon-platform')

import pytest
from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNNode, ReCoNState
from recon_agents.recon_arc_2.hypothesis import CooldownGateNode


def test_cooldown_gate_activation_and_decay():
    """Test that cooldown gate activates on trigger and decays naturally."""
    gate = CooldownGateNode("cooldown_gate")

    # Initial state should be inactive
    assert gate.activation == 0.0
    assert not gate.is_cooling_down()

    # Trigger cooldown for 3 steps
    gate.trigger_cooldown(3)
    assert gate.activation == 1.0  # Should be active
    assert gate.is_cooling_down()

    # Decay over time
    for i in range(5):
        gate.decay_step()
        if not gate.is_cooling_down():
            assert gate.activation == 0.0
            break
    else:
        pytest.fail("Gate did not decay after 5 steps")


def test_cooldown_gate_inhibits_action():
    """Test that active cooldown gate inhibits action via POR."""
    g = ReCoNGraph()

    # Create action node with terminal
    g.add_node("action", "script")
    terminal = ReCoNNode("terminal", "terminal")
    terminal.measurement_fn = lambda env: 1.0  # Always succeeds
    g.add_node(terminal)
    g.add_link("action", "terminal", "sub")

    cooldown_gate = CooldownGateNode("cooldown_gate")
    g.add_node(cooldown_gate)

    # Link gate to action via POR (gate inhibits action)
    g.add_link("cooldown_gate", "action", "por")

    # Without cooldown, action should progress normally
    g.request_root("action")
    for _ in range(4):
        g.propagate_step()
    assert g.get_node("action").state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)

    # Reset and trigger cooldown
    g.get_node("action").state = ReCoNState.INACTIVE
    g.get_node("action").activation = 0.0
    g.get_node("terminal").state = ReCoNState.INACTIVE
    g.get_node("terminal").activation = 0.0
    g.requested_roots.clear()

    cooldown_gate.trigger_cooldown(3)
    g.request_root("cooldown_gate")  # Keep gate active to maintain POR
    g.request_root("action")

    # Action should be suppressed by POR from active gate
    for _ in range(3):
        g.propagate_step()
    assert g.get_node("action").state == ReCoNState.SUPPRESSED


def test_cooldown_gate_releases_after_decay():
    """Test that action can progress after cooldown gate decays to 0."""
    g = ReCoNGraph()

    # Create action node with terminal
    g.add_node("action", "script")
    terminal = ReCoNNode("terminal", "terminal")
    terminal.measurement_fn = lambda env: 1.0  # Always succeeds
    g.add_node(terminal)
    g.add_link("action", "terminal", "sub")

    # Create cooldown gate with 1-step cooldown
    cooldown_gate = CooldownGateNode("cooldown_gate")
    g.add_node(cooldown_gate)

    # Link gate to action via POR
    g.add_link("cooldown_gate", "action", "por")

    # Trigger 1-step cooldown
    cooldown_gate.trigger_cooldown(1)
    g.request_root("cooldown_gate")  # Keep gate active initially
    g.request_root("action")

    # First step: action should be suppressed
    g.propagate_step()
    assert g.get_node("action").state == ReCoNState.SUPPRESSED

    # Manually decay the gate and stop requesting it
    cooldown_gate.decay_step()
    if not cooldown_gate.is_cooling_down():
        g.stop_request("cooldown_gate")

    # Action should now be able to progress
    for _ in range(5):
        g.propagate_step()
        if g.get_node("action").state in (ReCoNState.TRUE, ReCoNState.CONFIRMED):
            break
    assert g.get_node("action").state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)