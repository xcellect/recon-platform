"""
Test setup phase - 0:00-0:30

Verify that the basic skeleton works and tests pass.
"""
import pytest
import sys
import os

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

def test_imports_work():
    """Test that basic imports work"""
    # Should be able to import the core engine components
    from recon_engine.graph import ReCoNGraph
    from recon_engine.node import ReCoNState
    from recon_engine.neural_terminal import CNNValidActionTerminal
    
    assert ReCoNGraph is not None
    assert ReCoNState is not None
    assert CNNValidActionTerminal is not None

def test_continuous_sur_activation_works():
    """Test that continuous sur activation is working (from existing tests)"""
    from recon_engine.graph import ReCoNGraph
    from recon_engine.node import ReCoNState
    
    g = ReCoNGraph()
    g.add_node("parent", node_type="script")
    g.add_node("term", node_type="terminal")
    g.add_link("parent", "term", "sub", weight=1.0)
    
    # Set terminal measurement
    term = g.get_node("term")
    term.measurement_fn = lambda env=None: 0.9
    
    g.request_root("parent")
    
    # Run propagation steps
    for _ in range(3):
        g.propagate_step()
    
    parent = g.get_node("parent")
    assert parent.state == ReCoNState.TRUE
    assert abs(float(parent.activation) - 0.9) < 1e-6

def test_cnn_valid_action_terminal_exists():
    """Test that CNNValidActionTerminal can be instantiated"""
    from recon_engine.neural_terminal import CNNValidActionTerminal
    
    terminal = CNNValidActionTerminal("test_terminal")
    assert terminal.id == "test_terminal"
    assert terminal.model is not None
    assert hasattr(terminal, 'measure')
