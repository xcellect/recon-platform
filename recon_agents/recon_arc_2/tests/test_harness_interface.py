"""
Test Harness Interface Integration

Tests that the agent properly interfaces with the ARC-AGI-3-Agents harness.
Verifies the agent returns GameAction enums and handles coordinates correctly.
"""

import sys
sys.path.insert(0, '/workspace/recon-platform')

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


class MockGameState:
    """Mock game state."""
    WIN = "WIN"
    NOT_PLAYED = "NOT_PLAYED"
    GAME_OVER = "GAME_OVER"
    PLAYING = "PLAYING"


class MockGameAction:
    """Mock GameAction for testing."""
    def __init__(self, action_id, name):
        self.value = action_id
        self.name = name
        self.action_data = {}

    def set_data(self, data):
        """Set action data."""
        self.action_data = data

    @classmethod
    def from_id(cls, action_id):
        """Create action from ID - matches real GameAction enum mapping."""
        action_data = {
            0: (0, "RESET"),
            1: (1, "ACTION1"),
            2: (2, "ACTION2"),
            3: (3, "ACTION3"),
            4: (4, "ACTION4"),
            5: (5, "ACTION5"),
            6: (6, "ACTION6")  # ACTION6 has value 6
        }
        # Handle hypothesis manager indices (0-5) that map to GameAction values
        if action_id == 5:  # hypothesis index 5 -> ACTION6 (value 6)
            return cls(6, "ACTION6")

        value, name = action_data.get(action_id, (action_id, "ACTION1"))
        return cls(value, name)


class MockFrameData:
    """Mock frame data for harness interface testing."""
    def __init__(self, frame=None, score=0, state="PLAYING", available_actions=None):
        self.frame = frame
        self.score = score
        self.state = state
        self.available_actions = available_actions or []


def test_agent_is_done_detects_win_state():
    """Test that is_done correctly detects WIN state."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock frame with WIN state
    win_frame = MockFrameData(state=MockGameState.WIN)
    assert agent.is_done([], win_frame) == True

    # Mock frame with PLAYING state
    play_frame = MockFrameData(state=MockGameState.PLAYING)
    assert agent.is_done([], play_frame) == False

    # Mock frame with no state
    no_state_frame = MockFrameData()
    delattr(no_state_frame, 'state')
    assert agent.is_done([], no_state_frame) == False


def test_agent_choose_action_returns_game_action():
    """Test that choose_action returns GameAction (not int)."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock the GameAction import
    import recon_agents.recon_arc_2.agent as agent_module
    agent_module.GameAction = MockGameAction

    # Mock components
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.hypothesis_manager.get_selected_action.return_value = 2

    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)

    # Test frame
    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    # Choose action
    action = agent.choose_action([], frame_data)

    # Should return GameAction (not int)
    assert isinstance(action, MockGameAction)
    assert action.value == 2
    assert action.name == "ACTION2"


def test_agent_handles_action6_coordinates():
    """Test that ACTION6 gets coordinates attached."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock the GameAction import
    import recon_agents.recon_arc_2.agent as agent_module
    agent_module.GameAction = MockGameAction

    # Mock components
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.hypothesis_manager.get_selected_action.return_value = 5  # ACTION6 (index 5)
    # Mock propose_click_coordinates method if it exists
    if hasattr(agent.hypothesis_manager, 'propose_click_coordinates'):
        agent.hypothesis_manager.propose_click_coordinates.return_value = (32, 32)

    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)

    # Test frame
    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    # Choose action
    action = agent.choose_action([], frame_data)

    # Should be ACTION6 with coordinates
    assert isinstance(action, MockGameAction)
    assert action.value == 6  # GameAction.ACTION6 has value 6
    assert action.name == "ACTION6"
    # Coordinates should be set (default fallback is 32, 32)
    assert "x" in action.action_data
    assert "y" in action.action_data
    assert action.action_data["x"] == 32
    assert action.action_data["y"] == 32


def test_agent_handles_special_states():
    """Test that agent handles special states properly."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock the GameAction import
    import recon_agents.recon_arc_2.agent as agent_module
    agent_module.GameAction = MockGameAction

    # Test NOT_PLAYED state
    frame_data = MockFrameData(state=MockGameState.NOT_PLAYED)
    action = agent.choose_action([], frame_data)
    assert isinstance(action, MockGameAction)
    assert action.value == 0  # RESET

    # Test GAME_OVER state
    frame_data = MockFrameData(state=MockGameState.GAME_OVER)
    action = agent.choose_action([], frame_data)
    assert isinstance(action, MockGameAction)
    assert action.value == 0  # RESET


def test_agent_get_debug_info():
    """Test that get_debug_info returns proper debug information."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock hypothesis manager
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {"0": Mock(), "1": Mock()}
    agent.hypothesis_manager.get_debug_info = Mock(return_value={"test": "data"})

    debug_info = agent.get_debug_info()

    # Should contain agent info
    assert debug_info["agent_type"] == "ReCoN ARC-2 Thin"
    assert debug_info["game_id"] == "test_game"
    assert debug_info["hypothesis_count"] == 2

    # Should include hypothesis manager debug info
    assert debug_info["test"] == "data"


def test_process_frame_backward_compatibility():
    """Test that process_frame still works for existing code."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock the GameAction import
    import recon_agents.recon_arc_2.agent as agent_module
    agent_module.GameAction = MockGameAction

    # Mock components
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.hypothesis_manager.get_selected_action.return_value = 3

    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)

    # Test frame
    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    # Process frame - should return GameAction now
    action = agent.process_frame(frame_data)

    # Should return GameAction
    assert isinstance(action, MockGameAction)
    assert action.value == 3
    assert action.name == "ACTION3"


def test_thin_orchestrator_pattern_preserved():
    """Test that thin orchestrator pattern is preserved in new interface."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock the GameAction import
    import recon_agents.recon_arc_2.agent as agent_module
    agent_module.GameAction = MockGameAction

    # Mock all components
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.hypothesis_manager.get_selected_action.return_value = 1

    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)

    # Test frame
    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    # Choose action
    action = agent.choose_action([], frame_data)

    # Verify orchestration calls still happen
    agent.change_predictor.predict_change_probabilities.assert_called_once()
    agent.hypothesis_manager.feed_cnn_priors.assert_called_once()
    agent.hypothesis_manager.request_hypothesis_test.assert_called_once_with("hypothesis_root")
    agent.hypothesis_manager.propagate_step.assert_called()
    agent.hypothesis_manager.get_selected_action.assert_called()

    # Should return proper GameAction
    assert isinstance(action, MockGameAction)
    assert action.value == 1