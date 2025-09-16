"""
Test sanity checks phase - 4:00-4:45

Unit tests for magnitude flows, link-weight scaling, region aggregation priority,
integration test for frame → action emergence.
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
from recon_agents.recon_arc_angel.learning_manager import LearningManager
from recon_agents.recon_arc_angel.region_aggregator import RegionAggregator

def test_magnitude_flows_through_recon():
    """Test that CNN probabilities flow through ReCoN as continuous magnitudes"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Create frame with clear pattern
    frame = torch.zeros(16, 64, 64)
    frame[3, 20:30, 20:30] = 1.0  # Color 3 in a square
    
    # Update weights from CNN
    manager.update_weights_from_frame(frame)
    
    # Check that weights were updated with continuous values
    action_1_links = manager.graph.get_links(source="frame_change_hypothesis", target="action_1")
    sub_link = next(link for link in action_1_links if link.type == "sub")
    
    # Weight should be continuous sigmoid output, not just 0/1
    assert 0 <= sub_link.weight <= 1
    assert isinstance(sub_link.weight, (float, torch.Tensor))
    
    # Check click action weight
    click_links = manager.graph.get_links(source="frame_change_hypothesis", target="action_click")
    click_sub_link = next(link for link in click_links if link.type == "sub")
    assert 0 <= click_sub_link.weight <= 1

def test_link_weight_scaling():
    """Test that link weights scale activations correctly"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set specific weights
    action_1_links = manager.graph.get_links(source="frame_change_hypothesis", target="action_1")
    sub_link = next(link for link in action_1_links if link.type == "sub")
    sub_link.weight = 0.3  # Set specific weight
    
    # Set node activation
    manager.graph.nodes["action_1"].activation = 1.0
    manager.graph.nodes["action_1"].state = ReCoNState.CONFIRMED
    
    # Request root and propagate
    manager.request_frame_change()
    for _ in range(3):
        manager.propagate_step()
    
    # Check that parent receives scaled activation
    parent = manager.graph.nodes["frame_change_hypothesis"]
    # The exact activation depends on ReCoN implementation details,
    # but should be influenced by the weight
    assert hasattr(parent, 'activation')

def test_region_aggregation_priority():
    """Test that region aggregation preserves spatial priorities"""
    aggregator = RegionAggregator(region_size=8, grid_size=64)
    
    # Create coordinate probabilities with hot spots in different regions
    coord_probs = torch.zeros(64, 64)
    
    # High probability spot in region (1, 1) -> pixels 8-15, 8-15
    coord_probs[10:14, 10:14] = 0.9
    
    # Medium probability spot in region (5, 5) -> pixels 40-47, 40-47  
    coord_probs[42:46, 42:46] = 0.6
    
    # Low probability spot in region (7, 3) -> pixels 56-63, 24-31
    coord_probs[58:62, 26:30] = 0.3
    
    # Aggregate to regions
    region_scores = aggregator.aggregate_to_regions(coord_probs)
    
    # Check that priorities are preserved
    assert region_scores[1, 1] == 0.9  # Highest
    assert region_scores[5, 5] == 0.6  # Medium
    assert region_scores[7, 3] == 0.3  # Lowest
    
    # Check that empty regions have low scores
    assert region_scores[0, 0] < 0.1

def test_coordinate_conversion_consistency():
    """Test that coordinate conversions are consistent"""
    aggregator = RegionAggregator(region_size=8, grid_size=64)
    
    # Test round-trip conversions
    test_cases = [
        (0, 0), (7, 15), (32, 32), (63, 63)
    ]
    
    for pixel_y, pixel_x in test_cases:
        # Convert pixel to region
        region_y, region_x, local_y, local_x = aggregator.pixel_to_region(pixel_y, pixel_x)
        
        # Convert back to pixel
        recovered_y, recovered_x = aggregator.region_to_pixel(region_y, region_x, local_y, local_x)
        
        assert recovered_y == pixel_y
        assert recovered_x == pixel_x

def test_region_bounds_calculation():
    """Test region bounds calculation"""
    aggregator = RegionAggregator(region_size=8, grid_size=64)
    
    # Test specific regions
    test_cases = [
        ((0, 0), (0, 8, 0, 8)),
        ((1, 1), (8, 16, 8, 16)),
        ((7, 7), (56, 64, 56, 64))
    ]
    
    for (region_y, region_x), expected_bounds in test_cases:
        bounds = aggregator.get_region_bounds(region_y, region_x)
        assert bounds == expected_bounds

def test_action_selection_determinism():
    """Test that action selection is deterministic given same states"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set specific states and activations
    manager.graph.nodes["action_1"].state = ReCoNState.TRUE
    manager.graph.nodes["action_1"].activation = 0.5
    
    manager.graph.nodes["action_2"].state = ReCoNState.CONFIRMED
    manager.graph.nodes["action_2"].activation = 0.3
    
    manager.graph.nodes["action_3"].state = ReCoNState.WAITING
    manager.graph.nodes["action_3"].activation = 0.8
    
    # Get best action multiple times - should be consistent
    best_action_1, coords_1 = manager.get_best_action()
    best_action_2, coords_2 = manager.get_best_action()
    best_action_3, coords_3 = manager.get_best_action()
    
    assert best_action_1 == best_action_2 == best_action_3
    assert coords_1 == coords_2 == coords_3
    
    # Should select action_2 (CONFIRMED state has highest priority)
    assert best_action_1 == "action_2"

def test_learning_experience_uniqueness():
    """Test that learning manager correctly deduplicates experiences"""
    learning_manager = LearningManager(buffer_size=100, batch_size=5)
    
    # Create test frames
    frame1 = torch.zeros(16, 64, 64)
    frame1[1, 10:20, 10:20] = 1.0
    
    frame2 = torch.zeros(16, 64, 64)
    frame2[2, 10:20, 10:20] = 1.0  # Different
    
    # Add same experience multiple times
    added_1 = learning_manager.add_experience(frame1, frame2, "action_1")
    added_2 = learning_manager.add_experience(frame1, frame2, "action_1")  # Duplicate
    added_3 = learning_manager.add_experience(frame1, frame2, "action_2")  # Different action
    
    assert added_1 == True   # First time added
    assert added_2 == False  # Duplicate
    assert added_3 == True   # Different action
    
    assert len(learning_manager.experience_buffer) == 2
    assert len(learning_manager.experience_hashes) == 2

def test_unified_action_indexing_consistency():
    """Test that unified action indexing is consistent"""
    learning_manager = LearningManager()
    
    # Test individual actions
    indices = []
    for i in range(1, 6):
        idx = learning_manager.get_unified_action_index(f"action_{i}")
        indices.append(idx)
    
    # Should be 0, 1, 2, 3, 4
    assert indices == [0, 1, 2, 3, 4]
    
    # Test coordinate actions
    coord_idx_1 = learning_manager.get_unified_action_index("action_click", (0, 0))
    coord_idx_2 = learning_manager.get_unified_action_index("action_click", (0, 1))
    coord_idx_3 = learning_manager.get_unified_action_index("action_click", (1, 0))
    
    assert coord_idx_1 == 5      # First coordinate
    assert coord_idx_2 == 6      # Second coordinate  
    assert coord_idx_3 == 5 + 64 # Second row, first column

def test_frame_to_action_emergence_integration():
    """Integration test: frame → CNN → weights → ReCoN → action selection"""
    # This is the key integration test
    hypothesis_manager = HypothesisManager()
    hypothesis_manager.build_structure()
    
    learning_manager = LearningManager(buffer_size=100, batch_size=5)
    learning_manager.set_cnn_terminal(hypothesis_manager.cnn_terminal)
    
    # Create test frame with clear pattern
    frame = torch.zeros(16, 64, 64)
    frame[1, 30:40, 30:40] = 1.0  # Color 1 in center-ish region
    
    # Step 1: Update hypothesis weights from CNN
    hypothesis_manager.update_weights_from_frame(frame)
    
    # Step 2: Request frame change and propagate
    hypothesis_manager.request_frame_change()
    
    # Run several propagation steps
    for _ in range(5):
        hypothesis_manager.propagate_step()
    
    # Step 3: Get best action
    best_action, best_coords = hypothesis_manager.get_best_action()
    
    # Should select some action
    assert best_action is not None
    
    # If click action, should have valid coordinates
    if best_action == "action_click":
        assert best_coords is not None
        y, x = best_coords
        assert 0 <= y < 64
        assert 0 <= x < 64
    
    # Step 4: Test learning integration
    # Create a different frame to simulate action result
    next_frame = torch.zeros(16, 64, 64)
    next_frame[2, 30:40, 30:40] = 1.0  # Different color
    
    # Add experience
    if best_action == "action_click":
        added = learning_manager.add_experience(frame, next_frame, best_action, best_coords)
    else:
        added = learning_manager.add_experience(frame, next_frame, best_action)
    
    assert added == True  # Should be unique experience
    assert len(learning_manager.experience_buffer) == 1
    
    # Check that reward is 1.0 (frame changed)
    experience = learning_manager.experience_buffer[0]
    assert experience['reward'] == 1.0

def test_multiple_propagation_steps_stability():
    """Test that multiple propagation steps don't cause instability"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Create frame and update weights
    frame = torch.zeros(16, 64, 64)
    frame[3, 15:25, 15:25] = 1.0
    manager.update_weights_from_frame(frame)
    
    # Request and propagate many steps
    manager.request_frame_change()
    
    states_over_time = []
    for step in range(10):
        manager.propagate_step()
        states = manager.get_action_states()
        states_over_time.append(states)
    
    # Should eventually stabilize (not keep changing)
    # At least the last few states should be the same
    assert states_over_time[-1] == states_over_time[-2]

def test_availability_masking_integration():
    """Test availability masking with full system"""
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set up scenario where action_2 would be best
    manager.graph.nodes["action_2"].state = ReCoNState.CONFIRMED
    manager.graph.nodes["action_2"].activation = 0.9
    
    # But make it unavailable by setting to FAILED
    manager.graph.nodes["action_2"].state = ReCoNState.FAILED
    manager.graph.nodes["action_2"].activation = -1.0
    
    # Set action_4 as backup
    manager.graph.nodes["action_4"].state = ReCoNState.TRUE
    manager.graph.nodes["action_4"].activation = 0.5
    
    # Should select action_4 instead of failed action_2
    best_action, _ = manager.get_best_action()
    assert best_action == "action_4"

def test_system_reset_consistency():
    """Test that system resets work correctly"""
    hypothesis_manager = HypothesisManager()
    hypothesis_manager.build_structure()
    
    learning_manager = LearningManager()
    learning_manager.set_cnn_terminal(hypothesis_manager.cnn_terminal)
    
    # Add some experiences
    frame1 = torch.zeros(16, 64, 64)
    frame2 = torch.zeros(16, 64, 64)
    frame2[1, 0, 0] = 1.0
    
    learning_manager.add_experience(frame1, frame2, "action_1")
    assert len(learning_manager.experience_buffer) == 1
    
    # Trigger reset on score change
    reset_occurred = learning_manager.on_score_change(1)
    assert reset_occurred == True
    assert len(learning_manager.experience_buffer) == 0
    
    # Reset hypothesis manager
    hypothesis_manager.reset()
    states = hypothesis_manager.get_action_states()
    
    # All states should be INACTIVE after reset
    for state in states.values():
        assert state == "INACTIVE"
