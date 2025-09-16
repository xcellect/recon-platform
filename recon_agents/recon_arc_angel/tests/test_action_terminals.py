"""
Test that action nodes can succeed in ReCoN by making them terminal nodes
"""
import pytest
import sys
import os
import torch

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNState

def test_action_as_terminal_nodes():
    """Test that action nodes work better as terminal nodes"""
    g = ReCoNGraph()
    
    # Root
    g.add_node("root", node_type="script")
    
    # Action as terminal node that always confirms
    g.add_node("action_1", node_type="terminal")
    g.add_link("root", "action_1", "sub", weight=1.0)
    
    # Set measurement function to always confirm
    action_terminal = g.get_node("action_1")
    action_terminal.measurement_fn = lambda env=None: 1.0  # Always confirm
    
    # Request and propagate
    g.request_root("root")
    
    for _ in range(3):
        g.propagate_step()
    
    # Action should be CONFIRMED
    assert action_terminal.state == ReCoNState.CONFIRMED
    
    # Root should be TRUE or CONFIRMED (both are successful states)  
    root = g.get_node("root")
    assert root.state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]

def test_mixed_terminal_and_script_actions():
    """Test a mix of terminal actions and script actions with regions"""
    g = ReCoNGraph()
    
    # Root
    g.add_node("root", node_type="script")
    
    # Individual actions as terminals (ACTION1-ACTION5)
    for i in range(1, 6):
        action_id = f"action_{i}"
        g.add_node(action_id, node_type="terminal")
        g.add_link("root", action_id, "sub", weight=1.0)
        
        # Set measurement to confirm
        action_node = g.get_node(action_id)
        action_node.measurement_fn = lambda env=None: 1.0
    
    # Click action as script with region children
    g.add_node("action_click", node_type="script")
    g.add_link("root", "action_click", "sub", weight=1.0)
    
    # Add a few regions as terminals
    for region_y in range(2):
        for region_x in range(2):
            region_id = f"region_{region_y}_{region_x}"
            g.add_node(region_id, node_type="terminal")
            g.add_link("action_click", region_id, "sub", weight=1.0)
            
            # Set measurement to confirm
            region_node = g.get_node(region_id)
            region_node.measurement_fn = lambda env=None: 0.5  # Moderate confirmation
    
    # Request and propagate
    g.request_root("root")
    
    for _ in range(5):
        g.propagate_step()
    
    # Individual actions should be CONFIRMED
    for i in range(1, 6):
        action_id = f"action_{i}"
        action_node = g.get_node(action_id)
        assert action_node.state == ReCoNState.CONFIRMED
    
    # Click action should be FAILED (has no confirming children in this simple test)
    click_node = g.get_node("action_click")
    # In this test, regions don't confirm because they have default 0.5 measurement
    # which is below the default threshold, so click action fails
    assert click_node.state in [ReCoNState.INACTIVE, ReCoNState.FAILED]
    
    # Root should be TRUE or CONFIRMED (both are successful states)  
    root = g.get_node("root")
    assert root.state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
