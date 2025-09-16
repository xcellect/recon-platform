"""
Test selection and availability phase - 2:30-3:15

Test parent-side selection with state priority + activation,
and respect for available_actions mask.
"""
import pytest
import sys
import os
import torch
import numpy as np
from typing import List

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.node import ReCoNState
from recon_agents.recon_arc_angel.hypothesis_manager import HypothesisManager

def test_hypothesis_manager_build():
    """Test that HypothesisManager can be built successfully"""
    manager = HypothesisManager()
    manager.build_structure()
    
    stats = manager.get_stats()
    assert stats["built"] is True
    assert stats["total_nodes"] == 1 + 6 + 5 + 1 + 64  # root + 6 action scripts + 5 action terminals + cnn + 64 regions
    assert "action_1" in manager.graph.nodes
    assert "action_click" in manager.graph.nodes
    assert "region_0_0" in manager.graph.nodes
    assert "cnn_terminal" in manager.graph.nodes

def test_weight_updates_from_frame():
    """Test that weights are updated correctly from CNN predictions"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Create test frame
    frame = torch.zeros(16, 64, 64)
    frame[1, 20:30, 20:30] = 1.0  # Color 1 in a square
    
    # Update weights
    manager.update_weights_from_frame(frame)
    
    # Check that action weights were updated
    action_1_links = manager.graph.get_links(source="frame_change_hypothesis", target="action_1")
    assert len(action_1_links) > 0
    
    sub_link = next(link for link in action_1_links if link.type == "sub")
    assert 0 <= sub_link.weight <= 1

def test_action_state_tracking():
    """Test that action states are tracked correctly"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Initially all should be inactive
    states = manager.get_action_states()
    for action_id in ["action_1", "action_2", "action_3", "action_4", "action_5", "action_click"]:
        assert states[action_id] == "INACTIVE"

def test_best_action_selection_by_state():
    """Test action selection based on state priority"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Manually set different states to test priority
    manager.graph.nodes["action_1"].state = ReCoNState.CONFIRMED
    manager.graph.nodes["action_1"].activation = 0.5
    
    manager.graph.nodes["action_2"].state = ReCoNState.TRUE  
    manager.graph.nodes["action_2"].activation = 0.8
    
    manager.graph.nodes["action_3"].state = ReCoNState.WAITING
    manager.graph.nodes["action_3"].activation = 0.9
    
    # CONFIRMED should win despite lower activation
    best_action, best_coords = manager.get_best_action()
    assert best_action == "action_1"
    assert best_coords is None

def test_best_action_selection_by_activation():
    """Test action selection by activation when states are equal"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set same state, different activations
    manager.graph.nodes["action_1"].state = ReCoNState.TRUE
    manager.graph.nodes["action_1"].activation = 0.3
    
    manager.graph.nodes["action_2"].state = ReCoNState.TRUE
    manager.graph.nodes["action_2"].activation = 0.8
    
    manager.graph.nodes["action_3"].state = ReCoNState.TRUE
    manager.graph.nodes["action_3"].activation = 0.5
    
    # action_2 should win with highest activation
    best_action, best_coords = manager.get_best_action()
    assert best_action == "action_2"
    assert best_coords is None

def test_click_action_with_region_selection():
    """Test that click action selects best region"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set click action to be competitive
    manager.graph.nodes["action_click"].state = ReCoNState.CONFIRMED
    manager.graph.nodes["action_click"].activation = 0.9
    
    # Set some regions with different states/activations
    manager.graph.nodes["region_2_3"].state = ReCoNState.TRUE
    manager.graph.nodes["region_2_3"].activation = 0.6
    
    manager.graph.nodes["region_5_5"].state = ReCoNState.CONFIRMED
    manager.graph.nodes["region_5_5"].activation = 0.4
    
    manager.graph.nodes["region_7_1"].state = ReCoNState.WAITING
    manager.graph.nodes["region_7_1"].activation = 0.9
    
    # Should select click action with region_5_5 (CONFIRMED state wins)
    best_action, best_coords = manager.get_best_action()
    assert best_action == "action_click"
    assert best_coords is not None
    
    # Should be center of region (5,5) -> pixel (5*8 + 4, 5*8 + 4) = (44, 44)
    expected_y, expected_x = 5 * 8 + 4, 5 * 8 + 4
    assert best_coords == (expected_y, expected_x)

def test_region_coordinate_conversion():
    """Test region to pixel coordinate conversion"""
    manager = HypothesisManager(region_size=8, grid_size=64)
    
    # Test region (0,0) -> center should be (4,4)
    # Test region (3,5) -> center should be (3*8+4, 5*8+4) = (28, 44)
    # Test region (7,7) -> center should be (7*8+4, 7*8+4) = (60, 60)
    
    # We'll test this by setting up regions and checking coordinates
    manager.build_structure()
    
    manager.graph.nodes["action_click"].state = ReCoNState.TRUE
    manager.graph.nodes["action_click"].activation = 1.0
    
    # Test different regions
    test_cases = [
        ("region_0_0", (4, 4)),
        ("region_3_5", (28, 44)), 
        ("region_7_7", (60, 60))
    ]
    
    for region_id, expected_coords in test_cases:
        # Reset all regions
        for node in manager.graph.nodes.values():
            if node.id.startswith("region_"):
                node.state = ReCoNState.INACTIVE
                node.activation = 0.0
        
        # Set target region as best
        manager.graph.nodes[region_id].state = ReCoNState.CONFIRMED
        manager.graph.nodes[region_id].activation = 1.0
        
        best_action, best_coords = manager.get_best_action()
        assert best_action == "action_click"
        assert best_coords == expected_coords

class AvailabilityMask:
    """Helper class to test availability masking functionality"""
    
    def __init__(self, manager: HypothesisManager):
        self.manager = manager
    
    def apply_availability_mask(self, available_actions: List[str]):
        """
        Apply availability mask by setting unavailable actions to FAILED state.
        
        This simulates respecting the available_actions constraint from the harness.
        """
        # Map action names to node IDs
        action_mapping = {
            "ACTION1": "action_1",
            "ACTION2": "action_2", 
            "ACTION3": "action_3",
            "ACTION4": "action_4",
            "ACTION5": "action_5",
            "ACTION6": "action_click"
        }
        
        # Set unavailable actions to FAILED
        for action_name, node_id in action_mapping.items():
            if action_name not in available_actions:
                if node_id in self.manager.graph.nodes:
                    self.manager.graph.nodes[node_id].state = ReCoNState.FAILED
                    self.manager.graph.nodes[node_id].activation = -1.0
                    
                    # Also fail all regions if ACTION6 not available
                    if action_name == "ACTION6":
                        for region_y in range(self.manager.regions_per_dim):
                            for region_x in range(self.manager.regions_per_dim):
                                region_id = f"region_{region_y}_{region_x}"
                                if region_id in self.manager.graph.nodes:
                                    self.manager.graph.nodes[region_id].state = ReCoNState.FAILED
                                    self.manager.graph.nodes[region_id].activation = -1.0

def test_availability_masking():
    """Test that availability masking works correctly"""
    manager = HypothesisManager()
    manager.build_structure()
    
    masker = AvailabilityMask(manager)
    
    # Set all actions to have equal preference initially
    for i in range(1, 6):
        action_id = f"action_{i}"
        manager.graph.nodes[action_id].state = ReCoNState.TRUE
        manager.graph.nodes[action_id].activation = 0.5
    
    manager.graph.nodes["action_click"].state = ReCoNState.TRUE
    manager.graph.nodes["action_click"].activation = 0.5
    
    # Test case 1: Only ACTION1 and ACTION3 available
    masker.apply_availability_mask(["ACTION1", "ACTION3"])
    
    best_action, best_coords = manager.get_best_action()
    assert best_action in ["action_1", "action_3"]
    
    # Test case 2: Only ACTION6 available
    # Reset states first
    for i in range(1, 6):
        action_id = f"action_{i}"
        manager.graph.nodes[action_id].state = ReCoNState.TRUE
        manager.graph.nodes[action_id].activation = 0.5
    
    manager.graph.nodes["action_click"].state = ReCoNState.TRUE
    manager.graph.nodes["action_click"].activation = 0.5
    
    # Set a good region
    manager.graph.nodes["region_3_3"].state = ReCoNState.TRUE
    manager.graph.nodes["region_3_3"].activation = 0.8
    
    masker.apply_availability_mask(["ACTION6"])
    
    best_action, best_coords = manager.get_best_action()
    assert best_action == "action_click"
    assert best_coords is not None

def test_no_available_actions():
    """Test behavior when no actions are available"""
    manager = HypothesisManager()
    manager.build_structure()
    
    masker = AvailabilityMask(manager)
    
    # Set all actions to failed (none available)
    masker.apply_availability_mask([])
    
    best_action, best_coords = manager.get_best_action()
    # Should return None when no actions are available
    assert best_action is None
    assert best_coords is None

def test_integration_with_cnn_and_selection():
    """Integration test: CNN -> weight update -> selection"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Create frame that should favor certain actions/regions
    frame = torch.zeros(16, 64, 64)
    frame[2, 40:48, 40:48] = 1.0  # Color 2 in region (5,5)
    
    # Update weights from CNN
    manager.update_weights_from_frame(frame)
    
    # Request frame change and run a few steps
    manager.request_frame_change()
    
    for _ in range(5):
        manager.propagate_step()
    
    # Get best action
    best_action, best_coords = manager.get_best_action()
    
    # Should select some action (exact choice depends on CNN randomness)
    assert best_action is not None
    
    # If click action chosen, should have coordinates
    if best_action == "action_click":
        assert best_coords is not None
        assert 0 <= best_coords[0] < 64
        assert 0 <= best_coords[1] < 64
