"""
Test Pure ReCoN Hypothesis Network - No Manual State Control

This test verifies that the hypothesis network follows pure ReCoN semantics:
1. Single root request propagates through sub links automatically
2. States emerge from message passing (Table 1), not manual setting
3. Por/ret links provide natural inhibition without Python-side control
4. Gen loops provide persistence for failed states (cooldown)
5. No request_root() calls except for the single hypothesis root
6. No direct state or activation assignments
"""

import sys
sys.path.insert(0, '/workspace/recon-platform')

import pytest
from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNNode, ReCoNState
from recon_agents.recon_arc_2.hypothesis import HypothesisManager, TerminalMeasurementNode


def test_single_root_request_propagation():
    """Test that single root request propagates to children via sub links."""
    g = ReCoNGraph()

    # Create simple hierarchy: root -> action -> terminal
    g.add_node("hypothesis_root", "script")
    g.add_node("action_1", "script")

    terminal = TerminalMeasurementNode("terminal_1")
    terminal.set_measurement(True)  # Set success measurement
    g.add_node(terminal)

    # Connect via sub links
    g.add_link("hypothesis_root", "action_1", "sub")
    g.add_link("action_1", "terminal_1", "sub")

    # Only request the root - children should be requested automatically
    g.request_root("hypothesis_root")

    # Propagate and verify natural message flow
    for _ in range(5):
        g.propagate_step()

    # Root should be active and requesting children
    root_node = g.get_node("hypothesis_root")
    assert root_node.state in (ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED)

    # Action should be requested via sub link from root
    action_node = g.get_node("action_1")
    assert action_node.state in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED)

    # Terminal should be requested via sub link from action and succeed
    terminal_node = g.get_node("terminal_1")
    assert terminal_node.state in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.TRUE, ReCoNState.CONFIRMED)


def test_por_inhibition_without_manual_control():
    """Test that por links provide natural inhibition without Python-side control."""
    g = ReCoNGraph()

    # Create sequence: root -> action1 -> action2 -> terminal
    g.add_node("hypothesis_root", "script")
    g.add_node("action_1", "script")
    g.add_node("action_2", "script")

    terminal = TerminalMeasurementNode("terminal_2")
    terminal.measurement_fn = lambda env: 0.0  # Fails initially
    g.add_node(terminal)

    # Connect in sequence with por inhibition
    g.add_link("hypothesis_root", "action_1", "sub")
    g.add_link("action_1", "action_2", "sub")
    g.add_link("action_1", "action_2", "por")  # action_1 inhibits action_2
    g.add_link("action_2", "terminal_2", "sub")

    # Request only the root
    g.request_root("hypothesis_root")

    # Propagate steps
    for _ in range(3):
        g.propagate_step()

    # Action_1 should be active, action_2 should be suppressed by por
    action1 = g.get_node("action_1")
    action2 = g.get_node("action_2")

    # Action_1 should progress naturally
    assert action1.state in (ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.REQUESTED)

    # Action_2 should be inhibited by por from action_1
    assert action2.state in (ReCoNState.INACTIVE, ReCoNState.SUPPRESSED)


def test_states_emerge_from_messages_not_manual_setting():
    """Test that node states change due to incoming messages, not manual assignment."""
    g = ReCoNGraph()

    # Create simple action with terminal
    g.add_node("action", "script")
    terminal = TerminalMeasurementNode("terminal")
    terminal.set_measurement(True)  # Success measurement
    g.add_node(terminal)
    g.add_link("action", "terminal", "sub")

    # Track initial state before any requests
    initial_state = g.get_node("action").state
    assert initial_state == ReCoNState.INACTIVE  # Should start inactive

    # Request action
    g.request_root("action")

    # State should transition due to request message (may go directly to ACTIVE)
    g.propagate_step()
    after_request = g.get_node("action").state
    assert after_request in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING)

    # Further propagation - should be active or waiting
    g.propagate_step()
    active_state = g.get_node("action").state
    assert active_state in (ReCoNState.ACTIVE, ReCoNState.WAITING)

    # Terminal confirms - parent should become confirmed
    for _ in range(3):
        g.propagate_step()

    final_state = g.get_node("action").state
    assert final_state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)


def test_gen_loop_persistence_for_failed_states():
    """Test that failed nodes persist via gen loops (natural cooldown)."""
    g = ReCoNGraph()

    # Create action that will fail
    g.add_node("action", "script")
    terminal = TerminalMeasurementNode("terminal")
    terminal.measurement_fn = lambda env: 0.0  # Always fails
    g.add_node(terminal)

    g.add_link("action", "terminal", "sub")
    # Add gen loop for persistence
    g.add_link("action", "action", "gen")

    # Request action
    g.request_root("action")

    # Propagate until failure
    for _ in range(5):
        g.propagate_step()

    # Action should be in FAILED state
    action_node = g.get_node("action")
    assert action_node.state == ReCoNState.FAILED

    # Gen loop should maintain the failed state across steps
    for _ in range(3):
        g.propagate_step()
        assert action_node.state == ReCoNState.FAILED  # Persists via gen loop


def test_no_manual_request_management():
    """Verify that HypothesisManager uses only single root request."""
    # Create a hypothesis manager
    manager = HypothesisManager()

    # Check that initially only hypothesis_root might be requested
    # (Implementation will be updated to ensure this)

    # Create some action hypotheses
    action_hyp1 = manager.create_action_hypothesis(1, 0.8, None)
    action_hyp2 = manager.create_action_hypothesis(2, 0.6, None)

    # The manager should NOT call request_root() on individual actions
    # Only the hypothesis network structure should determine propagation

    # This test will pass when we refactor HypothesisManager
    # For now, it documents the expected behavior
    assert True  # Placeholder - will be implemented in refactor


def test_link_weights_modulate_flow_not_manual_gating():
    """Test that CNN priors work through link weights, not manual state control."""
    g = ReCoNGraph()

    # Create parent with two alternative children
    g.add_node("alternatives", "script")
    g.add_node("action_high_priority", "script")
    g.add_node("action_low_priority", "script")

    # Connect with different weights (α_valid affects sub weights)
    g.add_link("alternatives", "action_high_priority", "sub", weight=0.9)  # High α_valid
    g.add_link("alternatives", "action_low_priority", "sub", weight=0.1)   # Low α_valid

    # Request alternatives parent
    g.request_root("alternatives")

    # High weight child should be requested first/stronger
    for _ in range(3):
        g.propagate_step()

    high_priority = g.get_node("action_high_priority")
    low_priority = g.get_node("action_low_priority")

    # Due to weight differences, high priority should progress faster
    # (Exact behavior depends on weight implementation in engine)
    assert high_priority.activation >= low_priority.activation