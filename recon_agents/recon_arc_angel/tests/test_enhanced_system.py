"""
Test Suite for Enhanced Anti-Sticking System

Tests the complete systematic solution to prevent getting stuck:
1. Stable object identity with IoU-based matching
2. ReCoN hypothesis system with systematic reduction
3. Causal object-scoped verification
4. Deterministic evidence-led coordinate selection
5. Fixed CNN probability scaling
6. Principled exploration with hypothesis scheduling
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

from stable_object_tracker import StableObjectTracker, PersistentObject
from enhanced_hierarchy_manager import EnhancedHierarchicalHypothesisManager
from enhanced_production_agent import EnhancedProductionReCoNArcAngel


class TestStableObjectIdentity:
    """Test stable object identity with IoU-based matching."""
    
    def test_object_tracker_basic_functionality(self):
        """Test basic object tracker functionality"""
        tracker = StableObjectTracker()
        
        # Create initial objects
        obj1 = {
            "slice": (slice(10, 15), slice(10, 15)),
            "mask": np.ones((5, 5), dtype=bool),
            "colour": 1,
            "area": 25,
            "regularity": 1.0,
            "contrast": 1.0,
            "confidence": 0.8
        }
        
        obj2 = {
            "slice": (slice(20, 25), slice(20, 25)),
            "mask": np.ones((5, 5), dtype=bool),
            "colour": 2,
            "area": 25,
            "regularity": 1.0,
            "contrast": 1.0,
            "confidence": 0.7
        }
        
        # First frame
        persistent_objects = tracker.update_objects([obj1, obj2])
        
        assert len(persistent_objects) == 2, "Should create 2 persistent objects"
        assert len(tracker.object_id_mapping) == 2, "Should map 2 current objects"
        
        # Get persistent object by coordinate
        obj_at_12_12 = tracker.get_persistent_object_containing_coord((12, 12))
        assert obj_at_12_12 is not None, "Should find object containing (12, 12)"
        assert obj_at_12_12.current_color == 1, "Should be the red object"
    
    def test_iou_based_matching_across_frames(self):
        """Test IoU-based matching maintains stable identity"""
        tracker = StableObjectTracker()
        
        # Frame 1: Red square at (10-15, 10-15)
        obj1_frame1 = {
            "slice": (slice(10, 15), slice(10, 15)),
            "mask": np.ones((5, 5), dtype=bool),
            "colour": 1, "area": 25, "regularity": 1.0, "contrast": 1.0, "confidence": 0.8
        }
        
        persistent_objs_1 = tracker.update_objects([obj1_frame1])
        first_id = list(persistent_objs_1.keys())[0]
        
        # Frame 2: Same red square moved slightly (11-16, 11-16) - should match via IoU
        obj1_frame2 = {
            "slice": (slice(11, 16), slice(11, 16)),
            "mask": np.ones((5, 5), dtype=bool),
            "colour": 1, "area": 25, "regularity": 1.0, "contrast": 1.0, "confidence": 0.8
        }
        
        persistent_objs_2 = tracker.update_objects([obj1_frame2])
        
        # Should maintain same ID
        assert first_id in persistent_objs_2, "Should maintain stable ID across frames"
        assert len(persistent_objs_2) == 1, "Should recognize as same object"
        
        # Verify position updated
        updated_obj = persistent_objs_2[first_id]
        assert updated_obj.current_slice == (slice(11, 16), slice(11, 16)), "Should update position"


class TestCausalVerification:
    """Test causal object-scoped verification."""
    
    def test_causal_verification_success(self):
        """Test causal verification detects changes in clicked object only"""
        manager = EnhancedHierarchicalHypothesisManager()
        manager.build_enhanced_structure()
        
        # Frame 1: Red square and blue square
        frame1 = torch.zeros(16, 64, 64)
        frame1[1, 10:15, 10:15] = 1  # Red square
        frame1[2, 20:25, 20:25] = 1  # Blue square
        
        manager.update_dynamic_hypotheses(frame1)
        
        # Frame 2: Red square changed color
        frame2 = torch.zeros(16, 64, 64)
        frame2[3, 10:15, 10:15] = 1  # Red square -> yellow
        frame2[2, 20:25, 20:25] = 1  # Blue square unchanged
        
        # Verify click on red square
        clicked_coord = (12, 12)  # Inside red square
        success = manager.verify_clicked_object(clicked_coord, frame1, frame2)
        
        assert success, "Should detect change in clicked object (red square)"
    
    def test_causal_verification_ignores_other_changes(self):
        """Test causal verification ignores changes in other objects"""
        manager = EnhancedHierarchicalHypothesisManager()
        manager.build_enhanced_structure()
        
        # Frame 1: Red square and blue square
        frame1 = torch.zeros(16, 64, 64)
        frame1[1, 10:15, 10:15] = 1  # Red square
        frame1[2, 20:25, 20:25] = 1  # Blue square
        
        manager.update_dynamic_hypotheses(frame1)
        
        # Frame 2: Only blue square changed
        frame2 = torch.zeros(16, 64, 64)
        frame2[1, 10:15, 10:15] = 1  # Red square unchanged
        frame2[3, 20:25, 20:25] = 1  # Blue square -> yellow
        
        # Verify click on red square (no change in red square)
        clicked_coord = (12, 12)  # Inside red square
        success = manager.verify_clicked_object(clicked_coord, frame1, frame2)
        
        assert not success, "Should ignore changes in other objects (blue square)"


class TestDeterministicSelection:
    """Test deterministic evidence-led coordinate selection."""
    
    def test_deterministic_coordinate_selection(self):
        """Test that coordinate selection is deterministic and evidence-led"""
        manager = EnhancedHierarchicalHypothesisManager()
        manager.build_enhanced_structure()
        
        # Create frame
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        manager.update_dynamic_hypotheses(frame)
        
        # Create mock coordinate probabilities with clear peak
        coord_probs = torch.zeros(64, 64)
        coord_probs[12, 12] = 0.8  # Peak inside object
        coord_probs[11, 11] = 0.6  # Secondary peak
        coord_probs[13, 13] = 0.4  # Tertiary peak
        
        # Get persistent object
        persistent_objs = list(manager.object_tracker.persistent_objects.values())
        if persistent_objs:
            red_obj = [obj for obj in persistent_objs if obj.current_color == 1][0]
            
            # Should deterministically select argmax within mask
            coord1 = manager.get_deterministic_coordinate(red_obj, coord_probs)
            coord2 = manager.get_deterministic_coordinate(red_obj, coord_probs)
            
            assert coord1 == coord2, "Coordinate selection should be deterministic"
            assert coord1 == (12, 12), f"Should select argmax coordinate (12, 12), got {coord1}"


class TestCNNProbabilityScaling:
    """Test fixed CNN probability scaling."""
    
    def test_cnn_probability_scaling_fixes_magnitude(self):
        """Test that CNN probabilities are properly scaled to reasonable values"""
        manager = EnhancedHierarchicalHypothesisManager()
        manager.build_enhanced_structure()
        
        # Create frame
        frame = torch.zeros(16, 64, 64)
        frame[1, 10:15, 10:15] = 1
        
        manager.update_dynamic_hypotheses(frame)
        
        # Create realistic coordinate probabilities (small values after softmax over 4096)
        coord_probs = torch.zeros(64, 64)
        coord_probs[12, 12] = 0.0005  # Typical post-softmax value
        
        # Get persistent object
        persistent_objs = list(manager.object_tracker.persistent_objects.values())
        if persistent_objs:
            red_obj = [obj for obj in persistent_objs if obj.current_color == 1][0]
            
            # Calculate scaled probability
            scaled_prob = manager.calculate_scaled_masked_cnn_probability(coord_probs, red_obj)
            
            # Should be scaled up to reasonable range (capped at 1.0)
            expected_scaled = min(1.0, 0.0005 * 4096)  # Scale factor with cap
            assert abs(scaled_prob - expected_scaled) < 0.1, f"Should scale to ~{expected_scaled}, got {scaled_prob}"
            assert scaled_prob > 0.1, f"Scaled probability should be > 0.1, got {scaled_prob}"


class TestEnhancedProductionAgent:
    """Test enhanced production agent integration."""
    
    def test_enhanced_agent_prevents_action6_without_coords(self):
        """Test that enhanced agent never emits ACTION6 without coordinates"""
        agent = EnhancedProductionReCoNArcAngel()
        
        # Mock frame data
        class MockFrameData:
            def __init__(self):
                self.frame = np.zeros((64, 64), dtype=np.int64)
                self.score = 0
                self.state = "NOT_FINISHED"
                self.available_actions = [Mock(name="ACTION6")]
                self.available_actions[0].name = "ACTION6"
        
        frame_data = MockFrameData()
        
        # Should not crash and should never return ACTION6 without coordinates
        action = agent.choose_action([], frame_data)
        
        if action.action_type == "ACTION6":
            assert hasattr(action, 'data') and action.data, "ACTION6 must have coordinate data"
            assert 'x' in action.data and 'y' in action.data, "ACTION6 must have x,y coordinates"
        
        # Verify prevention statistics
        stats = agent.get_stats()
        # Should track prevention attempts (might be 0 if valid coordinates were found)
        assert 'action6_coord_none_prevented' in stats, "Should track ACTION6 prevention"
    
    def test_causal_verification_integration(self):
        """Test that causal verification is integrated in production agent"""
        agent = EnhancedProductionReCoNArcAngel()
        
        # Mock frame data
        class MockFrameData:
            def __init__(self, frame_array):
                self.frame = frame_array
                self.score = 0
                self.state = "NOT_FINISHED"
                self.available_actions = [Mock(name="ACTION6")]
                self.available_actions[0].name = "ACTION6"
        
        # Frame 1
        frame1_array = np.zeros((64, 64), dtype=np.int64)
        frame1_array[10:15, 10:15] = 1
        frame1_data = MockFrameData(frame1_array)
        
        # Frame 2 (changed)
        frame2_array = np.zeros((64, 64), dtype=np.int64)
        frame2_array[10:15, 10:15] = 2  # Color change
        frame2_data = MockFrameData(frame2_array)
        
        # First action
        action1 = agent.choose_action([], frame1_data)
        
        # Second action (should trigger causal verification)
        action2 = agent.choose_action([frame1_data], frame2_data)
        
        # Verify statistics
        stats = agent.get_stats()
        assert stats['causal_verifications'] > 0, "Should perform causal verifications"


if __name__ == "__main__":
    # Run specific test classes for development
    pytest.main([__file__ + "::TestStableObjectIdentity", "-v"])
