"""
Test suite for ReCoN click-object arbiter.

Tests the minimal ReCoN integration for click target selection using
bottom-up confirmation with link weights for penalty-based object ranking.
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, '/workspace/recon-platform')

# Mock torchvision to avoid import issues
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.models'] = MagicMock()

from recon_agents.blindsquirrel.state_graph import (
    BlindSquirrelState, BlindSquirrelStateGraph,
    compute_object_penalties, create_recon_click_arbiter, execute_recon_click_arbiter
)
from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNNode


class TestReCoNClickArbiter(unittest.TestCase):
    """Test ReCoN-based click object arbitration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock frame with simple 8x8 grid for testing
        self.test_frame_8x8 = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 2, 2, 0],
            [0, 1, 1, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        
        # Mock frame object
        self.mock_frame = MagicMock()
        self.mock_frame.game_id = "test"
        self.mock_frame.score = 0
        self.mock_frame.frame = [self.test_frame_8x8]
        self.mock_frame.available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        
        # Create state with object analysis
        self.state = BlindSquirrelState(self.mock_frame)
        
    def test_object_analysis_creates_correct_objects(self):
        """Test that object analysis correctly identifies connected components."""
        # Should find 3 objects: 2x2 red (color 1), 2x2 green (color 2), 2x3 blue (color 3)
        self.assertEqual(len(self.state.object_data), 3)
        
        # Check object properties
        objects_by_color = {obj['colour']: obj for obj in self.state.object_data}
        
        # Red object (color 1)
        red_obj = objects_by_color[1]
        self.assertEqual(red_obj['area'], 4)  # 2x2
        self.assertEqual(red_obj['regularity'], 1.0)  # Perfect square
        
        # Green object (color 2)  
        green_obj = objects_by_color[2]
        self.assertEqual(green_obj['area'], 4)  # 2x2
        self.assertEqual(green_obj['regularity'], 1.0)  # Perfect square
        
        # Blue object (color 3)
        blue_obj = objects_by_color[3]
        self.assertEqual(blue_obj['area'], 6)  # 2x3
        self.assertAlmostEqual(blue_obj['regularity'], 1.0)  # Perfect rectangle
        
    def test_compute_object_penalties_basic(self):
        """Test basic penalty computation for objects."""
        # Mock Pxy (click heatmap) - uniform for now
        pxy = np.ones((8, 8)) * 0.5
        
        penalties = compute_object_penalties(self.state.object_data, pxy, grid_size=8)
        
        # Should have penalties for all 3 objects
        self.assertEqual(len(penalties), 3)
        
        # All objects should have positive penalties
        for penalty in penalties:
            self.assertGreater(penalty, 0.0)
            self.assertLessEqual(penalty, 1.0)
            
    def test_compute_object_penalties_with_area_cutoff(self):
        """Test penalty computation with area fraction cutoff."""
        # Create state with tiny objects
        tiny_frame = [[0, 1, 0], [0, 0, 0], [2, 2, 2]]
        mock_tiny_frame = MagicMock()
        mock_tiny_frame.game_id = "test"
        mock_tiny_frame.score = 0
        mock_tiny_frame.frame = [tiny_frame]
        mock_tiny_frame.available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        
        tiny_state = BlindSquirrelState(mock_tiny_frame)
        pxy = np.ones((3, 3)) * 0.5
        
        penalties = compute_object_penalties(
            tiny_state.object_data, pxy, grid_size=3, area_frac_cutoff=0.2
        )
        
        # Object with area 1 should be filtered out (1/9 < 0.2)
        # Object with area 3 should remain (3/9 > 0.2)
        filtered_objects = [p for p in penalties if p > 0]
        self.assertEqual(len(filtered_objects), 1)
        
    def test_recon_click_arbiter_creation(self):
        """Test creation of ReCoN graph for click arbitration."""
        # Mock Pxy
        pxy = np.ones((8, 8)) * 0.5
        
        graph, object_weights = create_recon_click_arbiter(self.state.object_data, pxy, grid_size=8)
        
        # Should create ReCoN graph
        self.assertIsInstance(graph, ReCoNGraph)
        
        # Should have root node
        self.assertIn("action_click", graph.nodes)
        
        # Should have object terminals for each valid object
        num_valid_objects = len([w for w in object_weights if w > 0])
        expected_object_nodes = [f"object_{i}" for i in range(len(self.state.object_data))]
        
        for node_id in expected_object_nodes:
            if object_weights[int(node_id.split('_')[1])] > 0:
                self.assertIn(node_id, graph.nodes)
                
        # Should have weights for objects
        self.assertEqual(len(object_weights), len(self.state.object_data))
        
    def test_recon_click_arbiter_execution(self):
        """Test execution of ReCoN click arbiter."""
        # Mock Pxy with bias toward blue object (index 2)
        pxy = np.ones((8, 8)) * 0.1
        pxy[4:6, 2:5] = 0.9  # High activation over blue object area
        
        graph, object_weights = create_recon_click_arbiter(self.state.object_data, pxy, grid_size=8)
        selected_object_idx = execute_recon_click_arbiter(graph, object_weights)
        
        # Should select an object index
        self.assertIsInstance(selected_object_idx, int)
        self.assertGreaterEqual(selected_object_idx, 0)
        self.assertLess(selected_object_idx, len(self.state.object_data))
        
    def test_recon_click_arbiter_with_exploration(self):
        """Test ReCoN click arbiter with exploration flag."""
        pxy = np.ones((8, 8)) * 0.5
        graph, object_weights = create_recon_click_arbiter(self.state.object_data, pxy, grid_size=8)
        
        # Test with exploration
        selected_idx_explore = execute_recon_click_arbiter(graph, object_weights, exploration_rate=1.0)
        
        # Should still return valid index
        self.assertIsInstance(selected_idx_explore, int)
        self.assertGreaterEqual(selected_idx_explore, 0)
        self.assertLess(selected_idx_explore, len(self.state.object_data))
        
    def test_integration_with_blindsquirrel_state(self):
        """Test integration with BlindSquirrelState._get_click_action_obj."""
        # Test that the ReCoN arbiter can be called from the existing method
        
        # Mock the ReCoN arbiter functions
        with patch('recon_agents.blindsquirrel.state_graph.create_recon_click_arbiter') as mock_create, \
             patch('recon_agents.blindsquirrel.state_graph.execute_recon_click_arbiter') as mock_execute:
            
            mock_graph = MagicMock()
            mock_weights = [0.3, 0.7, 0.5]  # Weights for 3 objects
            mock_create.return_value = (mock_graph, mock_weights)
            mock_execute.return_value = 1  # Select second object
            
            # Test with ReCoN enabled
            action_obj = self.state._get_click_action_obj_with_recon(7, use_recon=True)  # action 7 = object index 2
            
            # Should return click action
            self.assertEqual(action_obj['type'], 'click')
            self.assertIn('x', action_obj)
            self.assertIn('y', action_obj)
            
    def test_fallback_to_original_behavior(self):
        """Test fallback to original BlindSquirrel behavior when ReCoN disabled."""
        # Test original behavior is preserved
        original_action = self.state._get_click_action_obj(6)  # action 6 = object index 1
        
        self.assertEqual(original_action['type'], 'click')
        self.assertIn('x', original_action)
        self.assertIn('y', original_action)
        
    def test_edge_cases(self):
        """Test edge cases for ReCoN click arbiter."""
        # Empty object list
        empty_penalties = compute_object_penalties([], np.ones((8, 8)), grid_size=8)
        self.assertEqual(len(empty_penalties), 0)
        
        # Single object
        single_obj = self.state.object_data[:1]
        single_penalties = compute_object_penalties(single_obj, np.ones((8, 8)), grid_size=8)
        self.assertEqual(len(single_penalties), 1)
        self.assertGreater(single_penalties[0], 0)
        
        # All objects filtered out
        tiny_objects = []
        filtered_penalties = compute_object_penalties(tiny_objects, np.ones((8, 8)), grid_size=8, area_frac_cutoff=1.0)
        self.assertEqual(len(filtered_penalties), 0)


class TestReCoNIntegrationFlags(unittest.TestCase):
    """Test ReCoN integration flags and ablation study support."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_graph = BlindSquirrelStateGraph()
        
    def test_recon_flag_configuration(self):
        """Test that ReCoN flags can be configured."""
        # Test default configuration
        self.assertFalse(self.state_graph.use_recon_click_arbiter)
        self.assertAlmostEqual(self.state_graph.recon_exploration_rate, 0.1)
        
        # Test configuration
        self.state_graph.configure_recon(
            use_click_arbiter=True,
            exploration_rate=0.2,
            area_frac_cutoff=0.01,
            border_penalty=0.8
        )
        
        self.assertTrue(self.state_graph.use_recon_click_arbiter)
        self.assertAlmostEqual(self.state_graph.recon_exploration_rate, 0.2)
        self.assertAlmostEqual(self.state_graph.recon_area_frac_cutoff, 0.01)
        self.assertAlmostEqual(self.state_graph.recon_border_penalty, 0.8)
        
    def test_recon_statistics_tracking(self):
        """Test that ReCoN usage statistics are tracked."""
        stats = self.state_graph.get_recon_statistics()
        
        self.assertIn('use_recon_click_arbiter', stats)
        self.assertIn('recon_click_selections', stats)
        self.assertIn('total_click_selections', stats)
        

if __name__ == '__main__':
    unittest.main()
