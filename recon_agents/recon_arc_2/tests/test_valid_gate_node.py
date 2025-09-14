"""
Test ValidGateNode - ReCoN-native α_valid delay with gen persistence.

ValidGateNode should:
- Use gen self-loop for natural decay
- Initial activation = (1 - α_valid) * max_delay
- Emit POR to delay action while activation > 0
- Higher α_valid = shorter delay, lower α_valid = longer delay
"""

import sys
sys.path.insert(0, '/workspace/recon-platform')

import pytest
from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNState
from recon_agents.recon_arc_2.hypothesis import ValidGateNode


def test_valid_gate_delay_from_prior():
    """Test that valid gate delay is proportional to (1 - α_valid)."""
    gate = ValidGateNode("valid_gate", max_delay=6)

    # High α_valid should have short delay
    gate.set_alpha_valid(0.9)  # High confidence
    assert gate.activation == 0.6  # (1 - 0.9) * 6 = 0.6

    # Low α_valid should have long delay
    gate.set_alpha_valid(0.1)  # Low confidence
    assert gate.activation == 5.4  # (1 - 0.1) * 6 = 5.4

    # Medium α_valid should have medium delay
    gate.set_alpha_valid(0.5)  # Medium confidence
    assert gate.activation == 3.0  # (1 - 0.5) * 6 = 3.0


def test_valid_gate_scheduling():
    """Test that higher α_valid actions progress before lower α_valid."""
    g = ReCoNGraph()

    # Create two actions with valid gates
    g.add_node("action_high", "script")
    g.add_node("action_low", "script")

    high_gate = ValidGateNode("gate_high", max_delay=4, decay_rate=1.0)  # Fast decay for testing
    low_gate = ValidGateNode("gate_low", max_delay=4, decay_rate=1.0)

    g.add_node(high_gate)
    g.add_node(low_gate)

    # Add gen self-loops
    g.add_link("gate_high", "gate_high", "gen")
    g.add_link("gate_low", "gate_low", "gen")

    # Link gates to actions
    g.add_link("gate_high", "action_high", "por")
    g.add_link("gate_low", "action_low", "por")

    # Set different α_valid values
    high_gate.set_alpha_valid(0.9)  # Short delay: (1-0.9)*4 = 0.4
    low_gate.set_alpha_valid(0.1)   # Long delay: (1-0.1)*4 = 3.6

    # Request both actions
    g.request_root("action_high")
    g.request_root("action_low")

    # After one propagation step:
    # - High α_valid action should be able to progress (short delay)
    # - Low α_valid action should still be inhibited (long delay)
    g.propagate_step()

    high_state = g.get_node("action_high").state
    low_state = g.get_node("action_low").state

    # High priority should progress, low should be suppressed
    assert high_state in (ReCoNState.ACTIVE, ReCoNState.WAITING)
    assert low_state == ReCoNState.SUPPRESSED


def test_valid_gate_inhibits_during_delay():
    """Test that valid gate emits POR to inhibit action during delay."""
    gate = ValidGateNode("valid_gate", max_delay=3, decay_rate=0.5)

    # Set low α_valid to create delay
    gate.set_alpha_valid(0.2)  # Delay = (1-0.2)*3 = 2.4

    # Should emit POR while delay is active
    result = gate.update_state_compact({"gen": gate.activation})
    assert result["por"] == 1.0
    assert gate.activation > 0

    # After several decay steps, should stop inhibiting
    for _ in range(10):
        result = gate.update_state_compact({"gen": gate.activation})
        gate.activation = result["gen"]
        if gate.activation <= 0.01:
            assert result["por"] == 0.0
            break
    else:
        pytest.fail("Gate did not decay to 0 after 10 steps")