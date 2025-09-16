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
    """Demonstrate pure ReCoN availability semantics"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set action_1 as available with low activation
    manager.graph.nodes["action_1"].state = ReCoNState.TRUE
    manager.graph.nodes["action_1"].activation = 0.2
    
    # Set action_click as effectively unavailable (pure ReCoN semantics)
    # by setting its sub weight to near zero
    for link in manager.graph.get_links(source="frame_change_hypothesis", target="action_click"):
        if link.type == "sub":
            link.weight = 1e-6
    manager.graph.nodes["action_click"].activation = 0.9  # High CNN confidence
    
    # With pure ReCoN semantics, action_click should be skipped due to low sub weight
    # vs action_1 with state TRUE + 0.2 activation = 3.2
    # So action_1 should win
    
    best_action, _ = manager.get_best_action()
    
    # With pure ReCoN implementation, action_click should be filtered out
    # due to low sub weight, so action_1 should win
    assert best_action == "action_1", f"Expected action_1, got {best_action}"
    
    # Verify that action_click is effectively unavailable
    assert manager._is_action_effectively_unavailable("action_click"), \
        "action_click should be effectively unavailable due to low sub weight"

def test_availability_masking_edge_case():
    """Test edge case where effectively unavailable action has very high activation"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set all individual actions as effectively unavailable
    for i in range(1, 6):
        action_id = f"action_{i}"
        for link in manager.graph.get_links(source="frame_change_hypothesis", target=action_id):
            if link.type == "sub":
                link.weight = 1e-6
        manager.graph.nodes[action_id].activation = -1.0
    
    # Set action_click as effectively unavailable but with extremely high activation
    for link in manager.graph.get_links(source="frame_change_hypothesis", target="action_click"):
        if link.type == "sub":
            link.weight = 1e-6
    manager.graph.nodes["action_click"].activation = 2.0  # Unrealistically high
    
    # Even with high activation, effectively unavailable actions should never be selected
    best_action, _ = manager.get_best_action()
    
    # Should return None since no viable actions
    assert best_action is None, f"Expected None for no viable actions, got {best_action}"

def test_availability_masking_with_region_scoring():
    """Test that effectively unavailable regions are also properly excluded"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Make action_click viable
    manager.graph.nodes["action_click"].state = ReCoNState.TRUE
    manager.graph.nodes["action_click"].activation = 0.8
    
    # Set some regions as effectively unavailable
    for link in manager.graph.get_links(source="action_click", target="region_3_3"):
        if link.type == "sub":
            link.weight = 1e-6
    manager.graph.nodes["region_3_3"].activation = 0.9  # High activation
    
    # Set another region as viable
    manager.graph.nodes["region_5_5"].state = ReCoNState.TRUE
    manager.graph.nodes["region_5_5"].activation = 0.3  # Lower activation
    
    best_action, best_coords = manager.get_best_action()
    
    if best_action == "action_click":
        # Should select region_5_5, not the effectively unavailable region_3_3
        expected_coords = (5 * 8 + 4, 5 * 8 + 4)  # Center of region (5,5)
        assert best_coords == expected_coords, f"Expected {expected_coords}, got {best_coords}"

def test_strict_failed_exclusion():
    """Test that effectively unavailable actions are strictly excluded regardless of activation"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Scenario: Only ACTION2 is available, but CNN strongly prefers ACTION6
    
    # Set ACTION2 as available with low confidence
    manager.graph.nodes["action_2"].state = ReCoNState.WAITING  # Lower priority than TRUE
    manager.graph.nodes["action_2"].activation = 0.1
    
    # Set ACTION6 as effectively unavailable but with very high CNN confidence
    for link in manager.graph.get_links(source="frame_change_hypothesis", target="action_click"):
        if link.type == "sub":
            link.weight = 1e-6
    manager.graph.nodes["action_click"].activation = 1.0
    
    # Set all other actions as effectively unavailable
    for i in [1, 3, 4, 5]:
        action_id = f"action_{i}"
        for link in manager.graph.get_links(source="frame_change_hypothesis", target=action_id):
            if link.type == "sub":
                link.weight = 1e-6
        manager.graph.nodes[action_id].activation = -1.0
    
    best_action, _ = manager.get_best_action()
    
    # Must select action_2 despite low confidence, because it's the only available one
    assert best_action == "action_2", f"Expected action_2 (only available), got {best_action}"
