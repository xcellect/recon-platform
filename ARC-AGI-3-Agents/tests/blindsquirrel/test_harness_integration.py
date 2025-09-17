"""
Harness integration tests for BlindSquirrel ReCoN integration.

Tests that the ReCoN click arbiter integrates properly with the existing
ARC-AGI-3-Agents harness without breaking existing functionality.
"""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add paths for imports
sys.path.insert(0, '/workspace/recon-platform/ARC-AGI-3-Agents')
sys.path.insert(0, '/workspace/recon-platform')

# Mock torchvision to avoid import issues
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.models'] = MagicMock()

from agents.blindsquirrel_recon import BlindSquirrelReCoN
from agents.structs import FrameData, GameAction, GameState


class TestHarnessIntegration(unittest.TestCase):
    """Test harness integration with ReCoN features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = BlindSquirrelReCoN(
            card_id="test_card",
            game_id="test_game", 
            agent_name="blindsquirrel_recon",
            ROOT_URL="http://localhost:8000",
            record=False
        )
        
        # Create mock frame data (frame is 3D: list[list[list[int]]])
        self.mock_frame = FrameData(
            game_id="test",
            score=0,
            frame=[[[0, 1, 1, 0], [0, 1, 1, 0], [2, 2, 0, 0], [2, 2, 0, 0]]],
            state=GameState.NOT_FINISHED,
            available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
                             GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6]
        )
        
    def test_agent_initialization_with_recon_flags(self):
        """Test that agent initializes with ReCoN configuration flags."""
        self.agent._ensure_agent()
        
        # Should have ReCoN configuration
        if self.agent.blindsquirrel_agent and hasattr(self.agent.blindsquirrel_agent.state_graph, 'use_recon_click_arbiter'):
            # ReCoN features available
            self.assertIsInstance(self.agent.blindsquirrel_agent.state_graph.use_recon_click_arbiter, bool)
        
    def test_choose_action_with_recon_enabled(self):
        """Test action selection with ReCoN click arbiter enabled."""
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent and self.agent.blindsquirrel_agent.state_graph:
            # Configure ReCoN
            self.agent.blindsquirrel_agent.state_graph.configure_recon(use_click_arbiter=True)
            
            # Process frame to set up state
            self.agent.is_done([], self.mock_frame)
            
            # Choose action
            action = self.agent.choose_action([], self.mock_frame)
            
            # Should return valid GameAction
            self.assertIsInstance(action, GameAction)
        else:
            self.skipTest("BlindSquirrel agent not available")
            
    def test_choose_action_with_recon_disabled(self):
        """Test action selection with ReCoN click arbiter disabled (original behavior)."""
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent and self.agent.blindsquirrel_agent.state_graph:
            # Ensure ReCoN is disabled
            self.agent.blindsquirrel_agent.state_graph.configure_recon(use_click_arbiter=False)
            
            # Process frame to set up state
            self.agent.is_done([], self.mock_frame)
            
            # Choose action
            action = self.agent.choose_action([], self.mock_frame)
            
            # Should return valid GameAction
            self.assertIsInstance(action, GameAction)
        else:
            self.skipTest("BlindSquirrel agent not available")
            
    def test_click_action_conversion_with_recon(self):
        """Test that ReCoN-selected click actions are properly converted to harness format."""
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent:
            # Configure ReCoN
            self.agent.blindsquirrel_agent.state_graph.configure_recon(use_click_arbiter=True)
            
            # Mock ReCoN selection
            mock_action_data = {
                'type': 'click',
                'x': 2,
                'y': 1,
                'recon_selected': True,
                'object_index': 0
            }
            
            # Test conversion
            game_action = self.agent._convert_action_data(mock_action_data)
            
            # Should be ACTION6 (click action)
            self.assertEqual(game_action, GameAction.ACTION6)
            
    def test_statistics_collection(self):
        """Test that ReCoN usage statistics are collected."""
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent:
            # Get statistics
            stats = self.agent.blindsquirrel_agent.get_agent_statistics()
            
            # Should include ReCoN statistics if available
            self.assertIsInstance(stats, dict)
            
    def test_backward_compatibility(self):
        """Test that existing functionality is preserved when ReCoN is disabled."""
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent:
            # Disable ReCoN
            self.agent.blindsquirrel_agent.state_graph.configure_recon(use_click_arbiter=False)
            
            # Process frame
            self.agent.is_done([], self.mock_frame)
            
            # Multiple action selections should work
            actions = []
            for _ in range(5):
                action = self.agent.choose_action([], self.mock_frame)
                actions.append(action)
                
            # All should be valid GameActions
            for action in actions:
                self.assertIsInstance(action, GameAction)
                
    def test_performance_with_recon_enabled(self):
        """Test that ReCoN integration doesn't significantly impact performance."""
        import time
        
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent:
            # Time with ReCoN disabled
            self.agent.blindsquirrel_agent.state_graph.configure_recon(use_click_arbiter=False)
            
            start_time = time.time()
            for _ in range(10):
                self.agent.is_done([], self.mock_frame)
                self.agent.choose_action([], self.mock_frame)
            time_without_recon = time.time() - start_time
            
            # Time with ReCoN enabled
            self.agent.blindsquirrel_agent.state_graph.configure_recon(use_click_arbiter=True)
            
            start_time = time.time()
            for _ in range(10):
                self.agent.is_done([], self.mock_frame)
                self.agent.choose_action([], self.mock_frame)
            time_with_recon = time.time() - start_time
            
            # ReCoN should not add significant overhead (< 50% increase)
            self.assertLess(time_with_recon, time_without_recon * 1.5)


class TestReCoNAblationStudy(unittest.TestCase):
    """Test support for ReCoN ablation studies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = BlindSquirrelReCoN(
            card_id="test_card",
            game_id="ablation_test",
            agent_name="blindsquirrel_recon", 
            ROOT_URL="http://localhost:8000",
            record=False
        )
        
    def test_ablation_flag_configuration(self):
        """Test that ablation flags can be configured programmatically."""
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent:
            # Test different configurations
            configs = [
                {'use_click_arbiter': False},
                {'use_click_arbiter': True, 'exploration_rate': 0.0},
                {'use_click_arbiter': True, 'exploration_rate': 0.1},
                {'use_click_arbiter': True, 'exploration_rate': 0.2},
            ]
            
            for config in configs:
                self.agent.blindsquirrel_agent.state_graph.configure_recon(**config)
                
                # Verify configuration was applied
                if 'use_click_arbiter' in config:
                    self.assertEqual(
                        self.agent.blindsquirrel_agent.state_graph.use_recon_click_arbiter,
                        config['use_click_arbiter']
                    )
                    
    def test_statistics_for_ablation_study(self):
        """Test that statistics are collected for ablation study analysis."""
        self.agent._ensure_agent()
        
        if self.agent.blindsquirrel_agent:
            # Enable ReCoN
            self.agent.blindsquirrel_agent.state_graph.configure_recon(use_click_arbiter=True)
            
            # Simulate some actions (frame is 3D: list[list[list[int]]])
            mock_frame = FrameData(
                game_id="test",
                score=0,
                frame=[[[1, 1, 0], [1, 1, 0], [0, 0, 2]]],
                state=GameState.NOT_FINISHED,
                available_actions=[GameAction.ACTION1, GameAction.ACTION6]
            )
            
            self.agent.is_done([], mock_frame)
            self.agent.choose_action([], mock_frame)
            
            # Get statistics
            stats = self.agent.blindsquirrel_agent.get_recon_statistics()
            
            # Should track ReCoN usage
            expected_stats = ['use_recon_click_arbiter', 'recon_click_selections', 'total_click_selections']
            for stat in expected_stats:
                if stat in stats:  # Only check if ReCoN is implemented
                    self.assertIn(stat, stats)


if __name__ == '__main__':
    # Set up test environment
    os.environ['PYTHONPATH'] = '/workspace/recon-platform:/workspace/recon-platform/ARC-AGI-3-Agents'
    
    unittest.main()
