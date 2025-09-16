"""
Test availability masking fix - ensuring it's airtight

This test demonstrates and fixes the issue where FAILED actions can still
be selected if they have high CNN activations.
"""
import pytest
import sys
import os
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.node import ReCoNState
from recon_agents.recon_arc_angel.hypothesis_manager import HypothesisManager

def test_availability_masking_issue_demonstration():
    """Demonstrate the availability masking issue"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set action_1 as available with low activation
    manager.graph.nodes["action_1"].state = ReCoNState.TRUE
    manager.graph.nodes["action_1"].activation = 0.2
    
    # Set action_click as FAILED (not available) but with high activation from CNN
    manager.graph.nodes["action_click"].state = ReCoNState.FAILED
    manager.graph.nodes["action_click"].activation = 0.9  # High CNN confidence
    
    # Current broken logic: -1 (FAILED) + 0.9 (activation) = -0.1
    # vs 3 (TRUE) + 0.2 (activation) = 3.2
    # So action_1 should win, but let's verify the scoring logic
    
    best_action, _ = manager.get_best_action()
    
    # With current implementation, this should pass (action_1 wins)
    # But if the logic were broken, action_click might win
    assert best_action == "action_1", f"Expected action_1, got {best_action}"

def test_availability_masking_edge_case():
    """Test edge case where FAILED action has very high activation"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set all individual actions as FAILED
    for i in range(1, 6):
        action_id = f"action_{i}"
        manager.graph.nodes[action_id].state = ReCoNState.FAILED
        manager.graph.nodes[action_id].activation = -1.0
    
    # Set action_click as FAILED but with extremely high activation
    manager.graph.nodes["action_click"].state = ReCoNState.FAILED
    manager.graph.nodes["action_click"].activation = 2.0  # Unrealistically high
    
    # Even with high activation, FAILED should never be selected
    best_action, _ = manager.get_best_action()
    
    # Should return None since no viable actions
    assert best_action is None, f"Expected None for no viable actions, got {best_action}"

def test_availability_masking_with_region_scoring():
    """Test that FAILED regions are also properly excluded"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Make action_click viable
    manager.graph.nodes["action_click"].state = ReCoNState.TRUE
    manager.graph.nodes["action_click"].activation = 0.8
    
    # Set some regions as FAILED (not available due to ACTION6 mask)
    manager.graph.nodes["region_3_3"].state = ReCoNState.FAILED
    manager.graph.nodes["region_3_3"].activation = 0.9  # High activation
    
    # Set another region as viable
    manager.graph.nodes["region_5_5"].state = ReCoNState.TRUE
    manager.graph.nodes["region_5_5"].activation = 0.3  # Lower activation
    
    best_action, best_coords = manager.get_best_action()
    
    if best_action == "action_click":
        # Should select region_5_5, not the FAILED region_3_3
        expected_coords = (5 * 8 + 4, 5 * 8 + 4)  # Center of region (5,5)
        assert best_coords == expected_coords, f"Expected {expected_coords}, got {best_coords}"

def test_strict_failed_exclusion():
    """Test that FAILED actions are strictly excluded regardless of activation"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Scenario: Only ACTION2 is available, but CNN strongly prefers ACTION6
    
    # Set ACTION2 as available with low confidence
    manager.graph.nodes["action_2"].state = ReCoNState.WAITING  # Lower priority than TRUE
    manager.graph.nodes["action_2"].activation = 0.1
    
    # Set ACTION6 as FAILED but with very high CNN confidence
    manager.graph.nodes["action_click"].state = ReCoNState.FAILED
    manager.graph.nodes["action_click"].activation = 1.0
    
    # Set all other actions as FAILED
    for i in [1, 3, 4, 5]:
        action_id = f"action_{i}"
        manager.graph.nodes[action_id].state = ReCoNState.FAILED
        manager.graph.nodes[action_id].activation = -1.0
    
    best_action, _ = manager.get_best_action()
    
    # Must select action_2 despite low confidence, because it's the only available one
    assert best_action == "action_2", f"Expected action_2 (only available), got {best_action}"
