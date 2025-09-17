"""
Test Suite for Improved ReCoN ARC Angel Implementation

Tests the complete improved implementation based on the agent's suggestions:
1. Proper ReCoN graph structure with por/ret sequences
2. Mask-aware CNN coupling 
3. Background suppression
4. Improved selection scoring
5. Stickiness mechanism
6. Comprehensive integration

Following TDD approach - these tests define the desired behavior before implementation.
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

from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNState
from improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager


class TestImprovedReCoNGraphStructure:
    """Test proper ReCoN graph structure for ACTION6 with por/ret sequences."""
    
    def test_action_click_has_proper_sequence_structure(self):
        """Test that action_click has proper por/ret sequence: click_cnn -> click_objects"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Verify action_click has two script children in sequence
        assert "action_click" in manager.graph.nodes
        assert "click_cnn" in manager.graph.nodes
        assert "click_objects" in manager.graph.nodes
        
        # Verify hierarchical structure
        click_cnn_links = manager.graph.get_links(source="action_click", target="click_cnn", link_type="sub")
        assert len(click_cnn_links) == 1
        
        click_objects_links = manager.graph.get_links(source="action_click", target="click_objects", link_type="sub")
        assert len(click_objects_links) == 1
        
        # Verify por/ret sequence between click_cnn and click_objects
        por_links = manager.graph.get_links(source="click_cnn", target="click_objects", link_type="por")
        assert len(por_links) == 1
        
        ret_links = manager.graph.get_links(source="click_objects", target="click_cnn", link_type="ret")
        assert len(ret_links) == 1
    
    def test_cnn_terminal_under_click_cnn_script(self):
        """Test that cnn_terminal is properly placed under click_cnn script node"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Verify cnn_terminal is child of click_cnn, not root
        cnn_links = manager.graph.get_links(source="click_cnn", target="cnn_terminal", link_type="sub")
        assert len(cnn_links) == 1
        
        # Verify cnn_terminal is NOT directly under root
        root_cnn_links = manager.graph.get_links(source="frame_change_hypothesis", target="cnn_terminal", link_type="sub")
        assert len(root_cnn_links) == 0
    
    def test_object_terminals_under_click_objects_script(self):
        """Test that object terminals are properly placed under click_objects script node"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create some test objects
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1  # Red square
        frame[2, 20:25, 20:25] = 1  # Green square
        
        manager.update_dynamic_objects_improved(frame)
        
        # Verify object terminals are children of click_objects, not action_click
        for obj_idx in range(len(manager.current_objects)):
            object_id = f"object_{obj_idx}"
            if object_id in manager.graph.nodes:
                # Should be under click_objects
                obj_links = manager.graph.get_links(source="click_objects", target=object_id, link_type="sub")
                assert len(obj_links) == 1
                
                # Should NOT be under action_click directly
                direct_links = manager.graph.get_links(source="action_click", target=object_id, link_type="sub")
                assert len(direct_links) == 0


class TestMaskAwareCNNCoupling:
    """Test mask-aware CNN coupling using masked max instead of bounding-box max."""
    
    def test_masked_max_cnn_probability_calculation(self):
        """Test that object probabilities use masked max instead of bounding-box max"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create test frame with distinct objects
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1  # Red square at (10-15, 10-15)
        
        # Mock CNN to return high probability at specific location
        mock_cnn_result = {
            "action_probabilities": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
            "coordinate_probabilities": torch.zeros(64, 64)
        }
        # High probability at edge of bounding box but outside mask
        mock_cnn_result["coordinate_probabilities"][9, 12] = 0.9  # Outside object
        mock_cnn_result["coordinate_probabilities"][12, 12] = 0.6  # Inside object
        
        with patch.object(manager.cnn_terminal, 'measure') as mock_measure, \
             patch.object(manager.cnn_terminal, '_process_measurement') as mock_process:
            mock_measure.return_value = mock_cnn_result
            mock_process.return_value = mock_cnn_result
            
            manager.update_weights_from_cnn_improved(frame)
            
            # Verify that masked max (0.6) is used, not bounding-box max (0.9)
            obj = manager.current_objects[0]
            expected_masked_max = 0.6  # Only consider pixels inside the actual mask
            
            # Check that object probability reflects masked max
            object_links = manager.graph.get_links(source="click_objects", target="object_0", link_type="sub")
            assert len(object_links) > 0
            assert abs(object_links[0].weight - expected_masked_max) < 0.01
    
    def test_background_object_filtering(self):
        """Test that background-like objects are properly filtered or downweighted"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create frame with background strip and small object
        frame = torch.zeros(16, 64, 64)
        frame[0, :, 0] = 1  # Full-height background strip (should be filtered)
        frame[1, 10:15, 10:15] = 1  # Small red square (should be kept)
        
        manager.update_dynamic_objects_improved(frame)
        
        # Background strip should be filtered out or heavily penalized
        background_objects = [obj for obj in manager.current_objects 
                            if obj.get("area_frac", 0) >= 0.20 or obj.get("border_penalty", 0) > 0.5]
        
        # Small object should remain with good confidence
        small_objects = [obj for obj in manager.current_objects 
                        if obj.get("area_frac", 0) < 0.20 and obj.get("regularity", 0) > 0.8]
        
        assert len(small_objects) > 0, "Small regular objects should be preserved"
        
        # If background objects exist, they should have low confidence
        for bg_obj in background_objects:
            assert bg_obj.get("confidence", 1.0) < 0.5, "Background objects should have low confidence"


class TestImprovedSelectionScoring:
    """Test comprehensive object evaluation for selection."""
    
    def test_comprehensive_object_scoring(self):
        """Test that objects are scored using masked_max_cnn_prob + regularity - penalties"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create test frame
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1  # Regular square
        frame[2, 20:30, 20:22] = 1  # Irregular rectangle
        
        # Mock CNN probabilities
        mock_cnn_result = {
            "action_probabilities": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
            "coordinate_probabilities": torch.zeros(64, 64)
        }
        mock_cnn_result["coordinate_probabilities"][12, 12] = 0.8  # Regular object
        mock_cnn_result["coordinate_probabilities"][25, 21] = 0.7  # Irregular object
        
        with patch.object(manager.cnn_terminal, 'measure') as mock_measure, \
             patch.object(manager.cnn_terminal, '_process_measurement') as mock_process:
            mock_measure.return_value = mock_cnn_result
            mock_process.return_value = mock_cnn_result
            
            manager.update_weights_from_cnn_improved(frame)
            
            # Get object scores using improved scoring
            best_action, best_coords = manager.get_best_action_with_improved_scoring(["ACTION6"])
            
            # Regular object should be preferred despite slightly lower CNN probability
            # because it has better regularity and lower penalties
            assert best_action == "action_click"
            assert best_coords is not None
            
            # Verify the selected coordinate is from the regular object (10-15, 10-15)
            y, x = best_coords
            assert 10 <= y <= 14 and 10 <= x <= 14, f"Selected coord {best_coords} should be in regular object bounds"
    
    def test_coordinate_selection_within_object_mask(self):
        """Test that selected coordinates are strictly within the object mask"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create L-shaped object to test mask adherence
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1  # Square part
        frame[1, 15:20, 10:12] = 1  # Extension part (L-shape)
        
        manager.update_dynamic_objects_improved(frame)
        
        # Select coordinate multiple times to test randomness
        valid_selections = 0
        for _ in range(10):
            coord = manager._get_object_coordinate_improved(0)  # Method we'll implement
            if coord is not None:
                y, x = coord
                # Verify coordinate is within the actual object mask
                if ((10 <= y <= 14 and 10 <= x <= 14) or  # Square part
                    (15 <= y <= 19 and 10 <= x <= 11)):   # Extension part
                    valid_selections += 1
        
        assert valid_selections >= 8, "Most coordinate selections should be within object mask"


class TestStickinessAfterFrameChange:
    """Test stickiness mechanism for successful clicks that cause frame changes."""
    
    def test_stickiness_after_successful_click(self):
        """Test that agent re-clicks same coordinate after frame change"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Initial frame
        frame1 = torch.zeros(16, 64, 64)
        frame1[1, 10:15, 10:15] = 1
        
        # Simulate successful click at (12, 12)
        successful_coord = (12, 12)
        manager.record_successful_click(successful_coord)  # Method we'll implement
        
        # New frame (different from frame1 - indicates frame change)
        frame2 = torch.zeros(16, 64, 64)
        frame2[2, 10:15, 10:15] = 1  # Same position, different color
        
        manager.update_weights_from_cnn_improved(frame2)
        
        # Should prefer the same coordinate due to stickiness
        best_action, best_coords = manager.get_best_action_with_improved_scoring(["ACTION6"])
        
        assert best_action == "action_click"
        # Due to stickiness, should select from the object containing the successful coord
        # (may not be exact due to random selection within mask)
        assert best_coords is not None
        y, x = best_coords
        # Should be within the same object bounds (10-15, 10-15)
        assert 10 <= y <= 14 and 10 <= x <= 14, f"Should select from sticky object, got {best_coords}"
    
    def test_stickiness_decay_after_no_change(self):
        """Test that stickiness decays if repeated clicks don't cause changes"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Record successful click
        successful_coord = (12, 12)
        manager.record_successful_click(successful_coord)
        
        # Same frame repeated (no change)
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        # Multiple updates with same frame should decay stickiness
        for _ in range(5):
            manager.update_weights_from_cnn_improved(frame)
            manager.decay_stickiness()  # Method we'll implement
        
        # Create different object with higher CNN probability
        frame[2, 20:25, 20:25] = 1  # New object
        mock_cnn_result = {
            "action_probabilities": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
            "coordinate_probabilities": torch.zeros(64, 64)
        }
        mock_cnn_result["coordinate_probabilities"][12, 12] = 0.3  # Low at sticky coord
        mock_cnn_result["coordinate_probabilities"][22, 22] = 0.9  # High at new object
        
        with patch.object(manager.cnn_terminal, 'measure') as mock_measure, \
             patch.object(manager.cnn_terminal, '_process_measurement') as mock_process:
            mock_measure.return_value = mock_cnn_result
            mock_process.return_value = mock_cnn_result
            
            manager.update_weights_from_cnn_improved(frame)
            best_action, best_coords = manager.get_best_action_with_improved_scoring(["ACTION6"])
            
            # Should now prefer new object due to decayed stickiness
            assert best_coords != successful_coord, "Stickiness should decay and prefer new high-probability object"


class TestReCoNMessagePropagation:
    """Test proper ReCoN message propagation in the improved structure."""
    
    def test_action6_sequence_propagation(self):
        """Test that ACTION6 follows proper sequence: action_click -> click_cnn -> click_objects"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create test frame and update weights
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        with patch.object(manager.cnn_terminal, 'measure') as mock_measure, \
             patch.object(manager.cnn_terminal, '_process_measurement') as mock_process:
            mock_result = {
                "action_probabilities": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
                "coordinate_probabilities": torch.ones(64, 64) * 0.8
            }
            mock_measure.return_value = mock_result
            mock_process.return_value = mock_result
            
            manager.update_weights_from_cnn_improved(frame)
        
        # Reset and request frame change
        manager.reset()
        manager.apply_availability_mask(["ACTION6"])
        manager.request_frame_change()
        
        # Step 1-2: Root should eventually request action_click
        manager.propagate_step()  # Root: INACTIVE -> ACTIVE
        manager.propagate_step()  # Root: ACTIVE -> WAITING, action_click: INACTIVE -> REQUESTED
        assert manager.graph.nodes["action_click"].state in [ReCoNState.REQUESTED, ReCoNState.ACTIVE]
        
        # Step 3-4: action_click should request both click_cnn and click_objects, but por inhibits click_objects
        manager.propagate_step()  # action_click: REQUESTED -> ACTIVE
        manager.propagate_step()  # action_click: ACTIVE -> WAITING, children get requested
        assert manager.graph.nodes["click_cnn"].state in [ReCoNState.REQUESTED, ReCoNState.ACTIVE]
        assert manager.graph.nodes["click_objects"].state == ReCoNState.REQUESTED  # Both requested initially
        
        # Step 5: click_cnn goes active, click_objects gets suppressed by por inhibition
        manager.propagate_step()
        assert manager.graph.nodes["click_cnn"].state in [ReCoNState.ACTIVE, ReCoNState.WAITING]
        assert manager.graph.nodes["click_objects"].state == ReCoNState.SUPPRESSED  # Inhibited by por from click_cnn
        
        # Additional steps: click_cnn should request cnn_terminal and eventually complete
        for _ in range(3):
            manager.propagate_step()
        
        # Verify sequence completion
        if manager.graph.nodes["cnn_terminal"].state == ReCoNState.CONFIRMED:
            assert manager.graph.nodes["click_cnn"].state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
    
    def test_availability_mask_blocks_action6_subtree(self):
        """Test that availability mask properly blocks entire ACTION6 subtree"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Apply mask without ACTION6
        manager.apply_availability_mask(["ACTION1", "ACTION2"])
        
        # Verify action_click and its subtree are blocked
        manager.request_frame_change()
        
        for _ in range(10):  # Multiple steps
            manager.propagate_step()
        
        # action_click subtree should remain inactive/failed
        assert manager.graph.nodes["action_click"].state in [ReCoNState.FAILED, ReCoNState.INACTIVE]
        assert manager.graph.nodes["click_cnn"].state in [ReCoNState.FAILED, ReCoNState.INACTIVE]  
        assert manager.graph.nodes["click_objects"].state in [ReCoNState.FAILED, ReCoNState.INACTIVE]


class TestIntegrationWithProductionAgent:
    """Test integration with ProductionReCoNArcAngel."""
    
    def test_production_agent_uses_improved_manager(self):
        """Test that ImprovedProductionReCoNArcAngel works properly"""
        from improved_production_agent import ImprovedProductionReCoNArcAngel
        
        # Mock frame data
        class MockFrameData:
            def __init__(self):
                self.frame = np.zeros((64, 64), dtype=np.int64)
                self.frame[10:15, 10:15] = 1  # Red square
                self.score = 0
                self.state = "PLAYING"
                self.available_actions = [Mock(name="ACTION6")]
                # Fix mock to have proper name attribute
                self.available_actions[0].name = "ACTION6"
        
        agent = ImprovedProductionReCoNArcAngel()
        frame_data = MockFrameData()
        
        # Should not crash and should return valid action
        action = agent.choose_action([], frame_data)
        assert action is not None
        assert hasattr(action, 'action_type') or hasattr(action, 'name')
        
        # Verify improved features are working
        stats = agent.get_stats()
        assert 'improvements' in stats
        assert len(stats['improvements']) == 6  # All 6 improvements implemented
    
    def test_debug_visualization_shows_proper_selection(self):
        """Test that debug visualization shows coordinate selection within object masks"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create test frame
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1  # Red square
        
        # Mock environment variable for debug
        with patch.dict(os.environ, {'RECON_DEBUG': '1'}):
            with patch.object(manager, '_save_debug_visualization') as mock_save:
                best_action, best_coords = manager.get_best_action_with_improved_scoring(["ACTION6"])
                
                if best_action == "action_click" and best_coords is not None:
                    # Should call debug visualization
                    mock_save.assert_called_once()
                    
                    # Verify coordinate is within expected bounds
                    y, x = best_coords
                    assert 10 <= y <= 14 and 10 <= x <= 14, f"Debug coord {best_coords} should be within object"


if __name__ == "__main__":
    # Run specific test classes for development
    pytest.main([__file__ + "::TestImprovedReCoNGraphStructure", "-v"])
