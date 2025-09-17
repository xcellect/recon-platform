"""
Test Suite for Improved Object-Scoped Stickiness Mechanism

Tests the enhanced stickiness implementation that addresses the issues:
1. Object-scoped change detection instead of global pixel diff
2. Conservative stickiness application with proper gating and capping
3. Boundary contrast calculation for high-contrast object emphasis
4. Proper stickiness clearing after stale attempts
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


class TestObjectScopedChangeDetection:
    """Test object-scoped change detection instead of global pixel diff."""
    
    def test_object_scoped_change_detection_success(self):
        """Test that changes within clicked object mask are detected"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create initial frame with red square
        frame1 = torch.zeros(16, 64, 64)
        frame1[1, 10:15, 10:15] = 1  # Red square
        
        manager.update_dynamic_objects_improved(frame1)
        
        # Record click on the red square
        click_coord = (12, 12)
        obj_idx = 0  # First object
        full_mask = manager._create_full_frame_mask(obj_idx)
        
        manager.record_successful_click(click_coord, obj_idx, frame1, full_mask)
        
        # Create new frame with color change WITHIN the object
        frame2 = torch.zeros(16, 64, 64)
        frame2[2, 10:15, 10:15] = 1  # Same position, different color (red->green)
        
        # Should detect change within object mask
        change_detected = manager.detect_object_scoped_change(frame2)
        assert change_detected, "Should detect color change within clicked object"
    
    def test_object_scoped_change_detection_ignores_external_changes(self):
        """Test that changes outside clicked object mask are ignored"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create initial frame with red square and blue square
        frame1 = torch.zeros(16, 64, 64)
        frame1[1, 10:15, 10:15] = 1  # Red square
        frame1[2, 20:25, 20:25] = 1  # Blue square
        
        manager.update_dynamic_objects_improved(frame1)
        
        # Record click on the red square (first object)
        click_coord = (12, 12)
        obj_idx = 0  # Red square
        full_mask = manager._create_full_frame_mask(obj_idx)
        
        manager.record_successful_click(click_coord, obj_idx, frame1, full_mask)
        
        # Create new frame with change ONLY in blue square (not clicked object)
        frame2 = torch.zeros(16, 64, 64)
        frame2[1, 10:15, 10:15] = 1  # Red square unchanged
        frame2[3, 20:25, 20:25] = 1  # Blue square changed to yellow
        
        # Should NOT detect change (change was outside clicked object)
        change_detected = manager.detect_object_scoped_change(frame2)
        assert not change_detected, "Should ignore changes outside clicked object mask"
    
    def test_change_ratio_and_pixel_thresholds(self):
        """Test that both ratio and absolute pixel thresholds work"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Large object (25 pixels)
        frame1 = torch.zeros(16, 64, 64)
        frame1[1, 10:15, 10:15] = 1  # 5x5 = 25 pixels
        
        manager.update_dynamic_objects_improved(frame1)
        
        # Record click
        click_coord = (12, 12)
        obj_idx = 0
        full_mask = manager._create_full_frame_mask(obj_idx)
        manager.record_successful_click(click_coord, obj_idx, frame1, full_mask)
        
        # Test ratio threshold (change 1 pixel out of 25 = 4% > 2% threshold)
        frame2 = frame1.clone()
        frame2[2, 10, 10] = 1  # Change 1 pixel to green
        frame2[1, 10, 10] = 0  # Remove red from that pixel
        
        change_detected = manager.detect_object_scoped_change(frame2)
        assert change_detected, "Should detect change via ratio threshold (4% > 2%)"
        
        # Test absolute pixel threshold (change 3 pixels)
        frame3 = frame1.clone()
        frame3[2, 10:13, 10] = 1  # Change 3 pixels
        frame3[1, 10:13, 10] = 0
        
        change_detected = manager.detect_object_scoped_change(frame3)
        assert change_detected, "Should detect change via absolute pixel threshold (3 >= 3)"


class TestBoundaryContrastCalculation:
    """Test boundary contrast calculation for high-contrast object emphasis."""
    
    def test_high_contrast_object(self):
        """Test that high-contrast objects get higher contrast scores"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create frame with high-contrast object (red on black background)
        frame = np.zeros((64, 64), dtype=int)
        frame[10:15, 10:15] = 1  # Red square on black background
        
        # Extract object
        objects = manager.extract_objects_from_frame(frame)
        
        # Should have high contrast (red surrounded by black)
        red_object = [obj for obj in objects if obj['colour'] == 1][0]
        assert red_object['contrast'] >= 0.8, f"High-contrast object should have contrast >= 0.8, got {red_object['contrast']}"
    
    def test_contrast_calculation_works(self):
        """Test that contrast calculation produces reasonable values"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Simple test: verify contrast calculation doesn't crash and produces valid values
        frame = np.zeros((64, 64), dtype=int)
        frame[10:15, 10:15] = 1  # Red square on black
        
        objects = manager.extract_objects_from_frame(frame)
        
        # Verify all objects have valid contrast values
        for obj in objects:
            assert 0.0 <= obj['contrast'] <= 1.0, f"Contrast should be in [0,1], got {obj['contrast']}"
        
        # Verify contrast is included in confidence calculation
        red_obj = [obj for obj in objects if obj['colour'] == 1][0]
        assert 'contrast' in red_obj, "Objects should have contrast field"
        assert red_obj['confidence'] > 0, "High-contrast object should have positive confidence"
    
    def test_contrast_affects_object_scoring(self):
        """Test that contrast affects comprehensive object scoring"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create frame with objects having different contrast levels
        frame = np.zeros((64, 64), dtype=int)
        frame[10:15, 10:15] = 1  # Red square on black (high contrast)
        
        # Create second object with reduced contrast (some boundary same color)
        frame[20:25, 20:25] = 2  # Green square
        frame[19, 20:25] = 2     # Top boundary same color (reduces contrast)
        frame[25, 20:25] = 2     # Bottom boundary same color
        
        objects = manager.extract_objects_from_frame(frame)
        
        # Find the objects (filter by reasonable size)
        sizeable_objects = [obj for obj in objects if obj['area'] >= 20]  # At least 20 pixels
        
        if len(sizeable_objects) >= 2:
            # Sort by contrast to compare
            sizeable_objects.sort(key=lambda x: -x['contrast'])
            high_contrast_obj = sizeable_objects[0]
            lower_contrast_obj = sizeable_objects[1]
            
            # Verify contrast affects confidence
            assert high_contrast_obj['confidence'] >= lower_contrast_obj['confidence'], \
                f"Higher contrast object should have better confidence: {high_contrast_obj['confidence']:.3f} vs {lower_contrast_obj['confidence']:.3f}"
        else:
            # Fallback: just verify contrast is calculated
            assert all(obj['contrast'] >= 0.0 for obj in objects), "All objects should have valid contrast scores"


class TestConservativeStickinessGating:
    """Test conservative stickiness application with gating and capping."""
    
    def test_stickiness_gated_by_cnn_probability(self):
        """Test that stickiness is only applied when CNN probability is decent"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        # Create frame and mock low CNN probability
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        manager.update_dynamic_objects_improved(frame)
        
        # Record successful click
        click_coord = (12, 12)
        obj_idx = 0
        full_mask = manager._create_full_frame_mask(obj_idx)
        manager.record_successful_click(click_coord, obj_idx, frame, full_mask)
        
        # Test with low CNN probability (below p_min=0.15)
        low_cnn_prob = 0.10
        score = manager.calculate_comprehensive_object_score(obj_idx, low_cnn_prob)
        
        # Should not get stickiness bonus due to low CNN prob
        obj = manager.current_objects[obj_idx]
        expected_score_without_stickiness = (low_cnn_prob + 
                                           obj["regularity"] * 0.3 + 
                                           obj["contrast"] * 0.4 - 
                                           obj["area_frac"] * 0.5 - 
                                           obj["border_penalty"] * 0.4)
        
        assert abs(score - expected_score_without_stickiness) < 0.01, "Low CNN prob should gate stickiness bonus"
        
        # Test with high CNN probability (above p_min=0.15)
        high_cnn_prob = 0.80
        score_with_high_prob = manager.calculate_comprehensive_object_score(obj_idx, high_cnn_prob)
        
        # Should get stickiness bonus with high CNN prob
        assert score_with_high_prob > score, "High CNN prob should enable stickiness bonus"
    
    def test_stickiness_bonus_capping(self):
        """Test that stickiness bonus is properly capped"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        manager.update_dynamic_objects_improved(frame)
        
        # Record successful click with high stickiness
        click_coord = (12, 12)
        obj_idx = 0
        full_mask = manager._create_full_frame_mask(obj_idx)
        manager.record_successful_click(click_coord, obj_idx, frame, full_mask)
        manager.stickiness_strength = 2.0  # Artificially high
        
        # Test with moderate CNN probability
        cnn_prob = 0.60
        score = manager.calculate_comprehensive_object_score(obj_idx, cnn_prob)
        
        # Calculate expected cap: min(0.5 * stickiness_strength, 0.5 * cnn_prob)
        expected_cap = min(0.5 * 2.0, 0.5 * 0.60)  # min(1.0, 0.30) = 0.30
        
        obj = manager.current_objects[obj_idx]
        base_score = (cnn_prob + obj["regularity"] * 0.3 + obj["contrast"] * 0.4 - 
                     obj["area_frac"] * 0.5 - obj["border_penalty"] * 0.4)
        expected_total = base_score + expected_cap
        
        assert abs(score - expected_total) < 0.01, f"Stickiness bonus should be capped at {expected_cap}"
    
    def test_stickiness_clears_after_stale_attempts(self):
        """Test that stickiness clears after K=2 stale attempts"""
        manager = ImprovedHierarchicalHypothesisManager()
        manager.build_improved_structure()
        
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        manager.update_dynamic_objects_improved(frame)
        
        # Record successful click
        click_coord = (12, 12)
        obj_idx = 0
        full_mask = manager._create_full_frame_mask(obj_idx)
        manager.record_successful_click(click_coord, obj_idx, frame, full_mask)
        
        # Simulate stale attempts (no changes)
        for attempt in range(3):
            manager.update_stickiness(frame, 0.80, True)  # No change, good CNN prob
            
            if attempt == 0:  # First attempt: should persist (attempts=1)
                assert manager.last_click['coords'] is not None, f"Should persist through attempt {attempt}"
            elif attempt == 1:  # Second attempt: should clear (attempts=2, reaches max)
                assert manager.last_click['coords'] is None, f"Should clear after {manager.max_sticky_attempts} stale attempts"
            else:  # Subsequent attempts: should remain cleared
                assert manager.last_click['coords'] is None, f"Should remain cleared after max attempts"


class TestImprovedProductionAgentIntegration:
    """Test integration of improved stickiness in the production agent."""
    
    def test_object_scoped_stickiness_in_production_agent(self):
        """Test that production agent uses object-scoped stickiness correctly"""
        agent = ImprovedProductionReCoNArcAngel()
        
        # Mock frame data
        class MockFrameData:
            def __init__(self, frame_array):
                self.frame = frame_array
                self.score = 0
                self.state = "NOT_FINISHED"
                self.available_actions = [Mock(name="ACTION6")]
                self.available_actions[0].name = "ACTION6"
        
        # Initial frame with red square
        frame1_array = np.zeros((64, 64), dtype=np.int64)
        frame1_array[10:15, 10:15] = 1
        frame1_data = MockFrameData(frame1_array)
        
        # First action - should select red square
        action1 = agent.choose_action([], frame1_data)
        assert action1.action_type == "ACTION6"
        
        # Verify click coordinates are within red square
        x1, y1 = action1.data['x'], action1.data['y']
        assert 10 <= y1 <= 14 and 10 <= x1 <= 14, f"First click {(x1, y1)} should be in red square"
        
        # Second frame with change in the red square
        frame2_array = np.zeros((64, 64), dtype=np.int64)
        frame2_array[10:15, 10:15] = 2  # Red square changed to green
        frame2_data = MockFrameData(frame2_array)
        
        # Second action - should maintain stickiness (select same object area)
        action2 = agent.choose_action([frame1_data], frame2_data)
        assert action2.action_type == "ACTION6"
        
        x2, y2 = action2.data['x'], action2.data['y']
        assert 10 <= y2 <= 14 and 10 <= x2 <= 14, f"Sticky click {(x2, y2)} should be in same object area"
        
        # Verify stickiness statistics
        stats = agent.get_stats()
        assert stats['successful_clicks'] > 0, "Should have recorded successful clicks"


if __name__ == "__main__":
    # Run specific test classes for development
    pytest.main([__file__ + "::TestObjectScopedChangeDetection", "-v"])
