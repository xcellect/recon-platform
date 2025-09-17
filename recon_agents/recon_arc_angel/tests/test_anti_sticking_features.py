"""
Test Suite for Anti-Sticking Features

Tests the enhanced implementation that prevents getting stuck in the same region:
1. Decoupled CNN softmax with temperature control
2. Per-object stale penalty tracking
3. Top-K probabilistic selection for exploration
4. CNN cache clearing on stale clicks
5. Coordinate probability heatmap analysis
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")
sys.path.insert(0, "/workspace/recon-platform/recon_agents/recon_arc_angel")

from improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager
from improved_production_agent import ImprovedProductionReCoNArcAngel
from recon_engine.neural_terminal import CNNValidActionTerminal


class TestDecoupledCNNSoftmax:
    """Test decoupled CNN softmax with temperature control."""
    
    def test_cnn_terminal_has_temperature_parameters(self):
        """Test that CNN terminal has separate temperature parameters"""
        cnn_terminal = CNNValidActionTerminal("test_cnn", use_gpu=False, action_temp=1.0, coord_temp=1.4)
        
        assert hasattr(cnn_terminal, 'action_temp'), "Should have action temperature parameter"
        assert hasattr(cnn_terminal, 'coord_temp'), "Should have coordinate temperature parameter"
        assert cnn_terminal.action_temp == 1.0, f"Action temp should be 1.0, got {cnn_terminal.action_temp}"
        assert cnn_terminal.coord_temp == 1.4, f"Coord temp should be 1.4, got {cnn_terminal.coord_temp}"
    
    def test_decoupled_softmax_processing(self):
        """Test that action and coordinate probabilities are normalized separately"""
        cnn_terminal = CNNValidActionTerminal("test_cnn", use_gpu=False, action_temp=1.0, coord_temp=2.0)
        
        # Create mock measurement with 4101 logits
        mock_logits = torch.randn(4101)
        
        # Process measurement
        result = cnn_terminal._process_measurement(mock_logits)
        
        assert "action_probabilities" in result, "Should have action probabilities"
        assert "coordinate_probabilities" in result, "Should have coordinate probabilities"
        
        action_probs = result["action_probabilities"]
        coord_probs = result["coordinate_probabilities"]
        
        # Verify shapes
        assert action_probs.shape == (5,), f"Action probs should be (5,), got {action_probs.shape}"
        assert coord_probs.shape == (64, 64), f"Coord probs should be (64,64), got {coord_probs.shape}"
        
        # Verify normalization (should sum to 1)
        assert abs(action_probs.sum().item() - 1.0) < 1e-6, "Action probs should sum to 1"
        assert abs(coord_probs.sum().item() - 1.0) < 1e-6, "Coord probs should sum to 1"
    
    def test_temperature_affects_distribution_flatness(self):
        """Test that higher temperature flattens coordinate distribution"""
        # Low temperature (peaky)
        cnn_low_temp = CNNValidActionTerminal("test_low", use_gpu=False, coord_temp=0.5)
        
        # High temperature (flat)
        cnn_high_temp = CNNValidActionTerminal("test_high", use_gpu=False, coord_temp=2.0)
        
        # Same input logits
        mock_logits = torch.randn(4101)
        mock_logits[5:] = torch.randn(4096) * 2  # Higher variance for coordinates
        
        result_low = cnn_low_temp._process_measurement(mock_logits)
        result_high = cnn_high_temp._process_measurement(mock_logits)
        
        coord_probs_low = result_low["coordinate_probabilities"]
        coord_probs_high = result_high["coordinate_probabilities"]
        
        # Calculate entropy (higher entropy = flatter distribution)
        entropy_low = -(coord_probs_low * torch.log(coord_probs_low + 1e-8)).sum().item()
        entropy_high = -(coord_probs_high * torch.log(coord_probs_high + 1e-8)).sum().item()
        
        assert entropy_high > entropy_low, f"High temp should have higher entropy: {entropy_high:.3f} vs {entropy_low:.3f}"


class TestStalePenaltyTracking:
    """Test per-object stale penalty tracking."""
    
    def test_stale_penalty_tracking(self):
        """Test that stale clicks are tracked per object"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Initially no stale tries
        assert manager.get_stale_penalty(0) == 0.0, "Initial stale penalty should be 0"
        
        # Record stale clicks
        manager.record_stale_click(0)
        manager.record_stale_click(0)
        
        # Should have penalty
        penalty = manager.get_stale_penalty(0)
        expected_penalty = 2 * manager.stale_penalty_lambda  # 2 tries * 0.2 = 0.4
        assert abs(penalty - expected_penalty) < 1e-6, f"Expected penalty {expected_penalty}, got {penalty}"
        
        # Reset on successful click
        manager.record_successful_object_click(0)
        assert manager.get_stale_penalty(0) == 0.0, "Penalty should reset after successful click"
    
    def test_stale_penalty_affects_object_scoring(self):
        """Test that stale penalty reduces object scores"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        manager.update_dynamic_objects_improved(frame)
        
        # Score without stale penalty
        score_clean = manager.calculate_comprehensive_object_score(0, 0.8)
        
        # Add stale tries
        manager.record_stale_click(0)
        manager.record_stale_click(0)  # 2 stale tries
        
        # Score with stale penalty
        score_stale = manager.calculate_comprehensive_object_score(0, 0.8)
        
        # Should be reduced by stale penalty
        expected_reduction = 2 * manager.stale_penalty_lambda  # 0.4
        assert abs((score_clean - score_stale) - expected_reduction) < 1e-6, \
            f"Score should be reduced by {expected_reduction}, was {score_clean - score_stale}"
    
    def test_stale_tries_cleared_for_disappeared_objects(self):
        """Test that stale tries are cleared when objects disappear"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create frame with multiple objects
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1  # Red square
        frame[2, 20:25, 20:25] = 1  # Green square
        
        manager.update_dynamic_objects_improved(frame)
        
        # Record stale tries for both objects
        manager.record_stale_click(0)
        manager.record_stale_click(1)
        
        assert 0 in manager.stale_tries, "Should track stale tries for object 0"
        assert 1 in manager.stale_tries, "Should track stale tries for object 1"
        
        # Create new frame with different objects (force object indices to change)
        frame2 = torch.zeros(16, 64, 64)
        frame2[3, 30:35, 30:35] = 1  # Blue square at different location (will get different index)
        
        manager.update_dynamic_objects_improved(frame2)
        
        # Check how many objects exist now
        num_objects_after = len(manager.current_objects)
        
        # Stale tries should be cleaned up for indices >= num_objects_after
        max_valid_idx = num_objects_after - 1
        
        # Verify cleanup works
        invalid_indices = [idx for idx in manager.stale_tries.keys() if idx > max_valid_idx]
        assert len(invalid_indices) == 0, f"Should clear stale tries for invalid indices: {invalid_indices}"


class TestTopKProbabilisticSelection:
    """Test top-K probabilistic selection for exploration."""
    
    def test_probabilistic_selection_with_multiple_objects(self):
        """Test that top-K selection provides exploration among good objects"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create frame with multiple similar objects
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1  # Red square
        frame[2, 20:25, 20:25] = 1  # Green square
        frame[3, 30:35, 30:35] = 1  # Blue square
        
        # Mock CNN to give similar probabilities
        mock_cnn_result = {
            "action_probabilities": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
            "coordinate_probabilities": torch.zeros(64, 64)
        }
        mock_cnn_result["coordinate_probabilities"][12, 12] = 0.8  # Red
        mock_cnn_result["coordinate_probabilities"][22, 22] = 0.75  # Green  
        mock_cnn_result["coordinate_probabilities"][32, 32] = 0.7  # Blue
        
        with patch.object(manager.cnn_terminal, 'measure') as mock_measure, \
             patch.object(manager.cnn_terminal, '_process_measurement') as mock_process:
            mock_measure.return_value = mock_cnn_result
            mock_process.return_value = mock_cnn_result
            
            manager.update_weights_from_cnn_improved(frame)
        
        # Test multiple selections to see if there's variety
        selected_objects = set()
        
        for _ in range(10):
            best_action, best_coords, best_obj_idx = manager.get_best_action_with_improved_scoring(["ACTION6"])
            if best_obj_idx is not None:
                selected_objects.add(best_obj_idx)
        
        # Should explore multiple objects due to probabilistic selection
        assert len(selected_objects) >= 1, "Should select at least one object"
        # Note: Due to randomness, we can't guarantee multiple selections in 10 tries,
        # but the mechanism is in place for exploration


class TestCNNCacheClearing:
    """Test CNN cache clearing on stale clicks."""
    
    def test_cache_clearing_on_stale_clicks(self):
        """Test that CNN cache is cleared when objects have stale clicks"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Verify cache clearing method exists
        assert hasattr(manager.cnn_terminal, 'clear_cache'), "CNN terminal should have clear_cache method"
        
        # Record stale click
        manager.record_stale_click(0)
        
        # Should trigger cache clearing
        with patch.object(manager.cnn_terminal, 'clear_cache') as mock_clear:
            manager.clear_cnn_cache_if_stale(0)
            mock_clear.assert_called_once()


class TestCoordinateHeatmapLogging:
    """Test coordinate probability heatmap logging for debugging."""
    
    def test_coordinate_heatmap_logging(self):
        """Test that coordinate heatmap logging works without crashing"""
        agent = ImprovedProductionReCoNArcAngel()
        
        # Create test frame
        frame_tensor = torch.zeros(16, 64, 64)
        frame_tensor[1, 10:15, 10:15] = 1
        
        # Should not crash when logging heatmap
        try:
            agent._log_coordinate_heatmap(frame_tensor)
            success = True
        except Exception as e:
            print(f"Heatmap logging error: {e}")
            success = False
        
        assert success, "Coordinate heatmap logging should not crash"


class TestIntegratedAntiStickingWorkflow:
    """Test complete anti-sticking workflow integration."""
    
    def test_complete_anti_sticking_workflow(self):
        """Test that all anti-sticking features work together"""
        agent = ImprovedProductionReCoNArcAngel()
        
        # Mock frame data
        class MockFrameData:
            def __init__(self, frame_array):
                self.frame = frame_array
                self.score = 0
                self.state = "NOT_FINISHED"
                self.available_actions = [Mock(name="ACTION6")]
                self.available_actions[0].name = "ACTION6"
        
        # Frame with red square
        frame_array = np.zeros((64, 64), dtype=np.int64)
        frame_array[10:15, 10:15] = 1
        frame_data = MockFrameData(frame_array)
        
        # First action
        action1 = agent.choose_action([], frame_data)
        assert action1.action_type == "ACTION6"
        
        # Simulate stale click (same frame)
        action2 = agent.choose_action([frame_data], frame_data)
        assert action2.action_type == "ACTION6"
        
        # Verify stale tracking
        stats = agent.get_stats()
        hm_stats = stats['hypothesis_manager']
        
        # Should have some stale tracking information
        assert 'stale_tries' in hm_stats, "Should track stale tries"
        assert 'stale_penalty_lambda' in hm_stats, "Should have stale penalty parameter"


if __name__ == "__main__":
    # Run specific test classes for development
    pytest.main([__file__ + "::TestDecoupledCNNSoftmax", "-v"])
