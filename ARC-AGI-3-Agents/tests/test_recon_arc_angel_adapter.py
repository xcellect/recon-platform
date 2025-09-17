"""
Test suite for ReCoN ARC Angel harness adapter.

TDD approach: Define expected behavior before implementation.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Add recon-platform to path
sys.path.insert(0, '/workspace/recon-platform')

# Import harness types
from agents.structs import FrameData, GameAction, GameState


class TestReCoNArcAngelAdapter:
    """Test the ReCoN ARC Angel harness adapter."""
    
    def test_adapter_exists_and_inherits_agent(self):
        """Test that the adapter class exists and inherits from Agent."""
        from agents.recon_arc_angel import ReCoNArcAngel
        from agents.agent import Agent
        
        assert issubclass(ReCoNArcAngel, Agent)
    
    def test_adapter_has_correct_max_actions(self):
        """Test that MAX_ACTIONS is set to 50000 like other ReCoN agents."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        assert ReCoNArcAngel.MAX_ACTIONS == 50000
    
    def test_adapter_lazy_initialization(self):
        """Test that the underlying agent is lazily initialized."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        # Create adapter without triggering initialization
        adapter = ReCoNArcAngel(
            card_id="test_card",
            game_id="test_game",
            agent_name="recon_arc_angel",
            ROOT_URL="http://localhost:8001",
            record=False
        )
        
        # Should not have underlying agent yet
        assert adapter.recon_arc_angel_agent is None
    
    @patch('agents.recon_arc_angel.ReCoNArcAngelAgent')
    def test_adapter_ensures_agent_on_first_use(self, mock_agent_class):
        """Test that _ensure_agent creates the underlying agent on first use."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        adapter = ReCoNArcAngel(
            card_id="test_card",
            game_id="test_game",
            agent_name="recon_arc_angel",
            ROOT_URL="http://localhost:8001",
            record=False
        )
        
        # Trigger lazy initialization
        adapter._ensure_agent()
        
        # Should have created underlying agent
        assert adapter.recon_arc_angel_agent is mock_agent_instance
        mock_agent_class.assert_called_once_with("test_game")
    
    @patch('agents.recon_arc_angel.ReCoNArcAngelAgent')
    def test_is_done_proxies_to_underlying_agent(self, mock_agent_class):
        """Test that is_done proxies to underlying agent's is_done method."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        mock_agent_instance = Mock()
        mock_agent_instance.is_done.return_value = True
        mock_agent_class.return_value = mock_agent_instance
        
        adapter = ReCoNArcAngel(
            card_id="test_card",
            game_id="test_game",
            agent_name="recon_arc_angel",
            ROOT_URL="http://localhost:8001",
            record=False
        )
        
        # Create mock frame data
        frames = []
        latest_frame = Mock(spec=FrameData)
        latest_frame.state = GameState.NOT_FINISHED
        
        result = adapter.is_done(frames, latest_frame)
        
        assert result is True
        mock_agent_instance.is_done.assert_called_once_with(frames, latest_frame)
    
    @patch('agents.recon_arc_angel.ReCoNArcAngelAgent')
    def test_choose_action_proxies_to_underlying_agent(self, mock_agent_class):
        """Test that choose_action proxies to underlying agent's choose_action method."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        mock_agent_instance = Mock()
        mock_game_action = Mock(spec=GameAction)
        mock_game_action.name = "ACTION1"
        mock_agent_instance.choose_action.return_value = mock_game_action
        mock_agent_class.return_value = mock_agent_instance
        
        adapter = ReCoNArcAngel(
            card_id="test_card",
            game_id="test_game", 
            agent_name="recon_arc_angel",
            ROOT_URL="http://localhost:8001",
            record=False
        )
        
        # Create mock frame data
        frames = []
        latest_frame = Mock(spec=FrameData)
        latest_frame.state = GameState.NOT_FINISHED
        latest_frame.available_actions = [GameAction.ACTION1, GameAction.ACTION2]
        
        result = adapter.choose_action(frames, latest_frame)
        
        assert result is mock_game_action
        mock_agent_instance.choose_action.assert_called_once_with(frames, latest_frame)
    
    def test_handles_import_error_gracefully(self):
        """Test that adapter handles import errors gracefully."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        # Mock import failure
        with patch('agents.recon_arc_angel.ReCoNArcAngelAgent', side_effect=ImportError("Module not found")):
            adapter = ReCoNArcAngel(
                card_id="test_card",
                game_id="test_game",
                agent_name="recon_arc_angel", 
                ROOT_URL="http://localhost:8001",
                record=False
            )
            
            # Should handle import error gracefully
            adapter._ensure_agent()
            assert adapter.recon_arc_angel_agent is None
    
    def test_fallback_behavior_when_agent_not_available(self):
        """Test fallback behavior when underlying agent is not available."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        with patch('agents.recon_arc_angel.ReCoNArcAngelAgent', side_effect=ImportError("Module not found")):
            adapter = ReCoNArcAngel(
                card_id="test_card",
                game_id="test_game",
                agent_name="recon_arc_angel",
                ROOT_URL="http://localhost:8001", 
                record=False
            )
            
            # Create mock frame data
            frames = []
            latest_frame = Mock(spec=FrameData)
            latest_frame.state = GameState.NOT_FINISHED
            latest_frame.available_actions = [GameAction.ACTION1, GameAction.ACTION2]
            
            # Should fallback gracefully
            result = adapter.choose_action(frames, latest_frame)
            assert result in latest_frame.available_actions
    
    def test_handles_reset_states(self):
        """Test that adapter handles NOT_PLAYED and GAME_OVER states correctly."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        with patch('agents.recon_arc_angel.ReCoNArcAngelAgent'):
            adapter = ReCoNArcAngel(
                card_id="test_card",
                game_id="test_game",
                agent_name="recon_arc_angel",
                ROOT_URL="http://localhost:8001",
                record=False
            )
            
            frames = []
            
            # Test NOT_PLAYED state
            not_played_frame = Mock(spec=FrameData)
            not_played_frame.state = GameState.NOT_PLAYED
            result = adapter.choose_action(frames, not_played_frame)
            assert result == GameAction.RESET
            
            # Test GAME_OVER state
            game_over_frame = Mock(spec=FrameData)
            game_over_frame.state = GameState.GAME_OVER
            result = adapter.choose_action(frames, game_over_frame)
            assert result == GameAction.RESET


class TestReCoNArcAngelRegistration:
    """Test that ReCoN ARC Angel is properly registered in the harness."""
    
    def test_agent_registered_in_available_agents(self):
        """Test that recon_arc_angel is registered in AVAILABLE_AGENTS."""
        from agents import AVAILABLE_AGENTS
        
        assert "recon_arc_angel" in AVAILABLE_AGENTS
        
        # Verify it maps to the correct class
        from agents.recon_arc_angel import ReCoNArcAngel
        assert AVAILABLE_AGENTS["recon_arc_angel"] == ReCoNArcAngel
    
    def test_agent_can_be_instantiated_by_swarm(self):
        """Test that the agent can be instantiated by the Swarm class."""
        from agents import AVAILABLE_AGENTS
        from agents.swarm import Swarm
        
        agent_class = AVAILABLE_AGENTS["recon_arc_angel"]
        
        # Should be able to instantiate with Swarm parameters
        agent = agent_class(
            card_id="test_card",
            game_id="test_game", 
            agent_name="recon_arc_angel",
            ROOT_URL="http://localhost:8001",
            record=False
        )
        
        assert agent is not None
        assert hasattr(agent, 'is_done')
        assert hasattr(agent, 'choose_action')


class TestReCoNArcAngelIntegration:
    """Integration tests for the complete adapter."""
    
    @patch('agents.recon_arc_angel.ReCoNArcAngelAgent')
    def test_full_action_loop(self, mock_agent_class):
        """Test a complete action selection loop."""
        from agents.recon_arc_angel import ReCoNArcAngel
        
        # Setup mock underlying agent
        mock_agent_instance = Mock()
        mock_agent_instance.is_done.return_value = False
        
        mock_game_action = Mock(spec=GameAction)
        mock_game_action.name = "ACTION6"
        mock_game_action.action_data = Mock()
        mock_game_action.action_data.x = 32
        mock_game_action.action_data.y = 16
        mock_agent_instance.choose_action.return_value = mock_game_action
        
        mock_agent_class.return_value = mock_agent_instance
        
        adapter = ReCoNArcAngel(
            card_id="test_card",
            game_id="test_game",
            agent_name="recon_arc_angel", 
            ROOT_URL="http://localhost:8001",
            record=False
        )
        
        # Create realistic frame data
        frames = []
        latest_frame = Mock(spec=FrameData)
        latest_frame.state = GameState.NOT_FINISHED
        latest_frame.score = 0
        latest_frame.available_actions = [GameAction.ACTION1, GameAction.ACTION6]
        
        # Test is_done
        done = adapter.is_done(frames, latest_frame)
        assert done is False
        
        # Test choose_action
        action = adapter.choose_action(frames, latest_frame)
        assert action is mock_game_action
        
        # Verify underlying agent was called
        mock_agent_instance.is_done.assert_called_once()
        mock_agent_instance.choose_action.assert_called_once()
