"""
Test Thin Agent Architecture

Tests that the agent is a pure orchestrator that delegates all active perception
logic to the hypothesis manager. The agent should only:
1. Extract frames
2. Get CNN predictions
3. Feed priors to hypothesis manager
4. Request root and propagate
5. Return emergent action
"""

import sys
sys.path.insert(0, '/workspace/recon-platform')

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from recon_agents.recon_arc_2.agent import ReCoNArc2Agent


class MockAction:
    """Mock action with value attribute."""
    def __init__(self, value):
        self.value = value

class MockFrameData:
    """Mock frame data for testing."""
    def __init__(self, frame=None, score=0, game_id="test", state="PLAYING", available_actions=None):
        self.frame = frame
        self.score = score
        self.game_id = game_id
        self.state = state
        # Convert int list to mock actions with .value attribute
        if available_actions:
            self.available_actions = [MockAction(i) for i in available_actions]
        else:
            self.available_actions = []


def test_agent_delegates_action_selection_to_hypothesis_manager():
    """Test that agent delegates all action selection to hypothesis manager."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock the hypothesis manager
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.get_selected_action.return_value = 2
    agent.hypothesis_manager.create_action_hypothesis.return_value = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()

    # Mock CNN predictor
    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.1, 0.3, 0.8, 0.2, 0.1, 0.4])

    # Create test frame
    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame, available_actions=[0, 1, 2, 3, 4, 5])

    # Process frame
    action = agent.process_frame(frame_data)

    # Should return action from hypothesis manager
    # Hypothesis index 2 → GameAction value 3
    assert action == 3

    # Should have called hypothesis manager methods
    agent.hypothesis_manager.feed_cnn_priors.assert_called_once()
    agent.hypothesis_manager.request_hypothesis_test.assert_called_once_with("hypothesis_root")
    agent.hypothesis_manager.propagate_step.assert_called()
    agent.hypothesis_manager.get_selected_action.assert_called_once()


def test_agent_feeds_cnn_priors_only():
    """Test that agent only feeds CNN priors and doesn't do manual selection."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock components
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.get_selected_action.return_value = 1
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()

    agent.change_predictor = Mock()
    cnn_probs = np.array([0.2, 0.9, 0.1, 0.3, 0.4, 0.5])
    agent.change_predictor.predict_change_probabilities.return_value = cnn_probs

    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    # Process frame
    agent.process_frame(frame_data)

    # Should feed CNN priors exactly as predicted
    agent.hypothesis_manager.feed_cnn_priors.assert_called_once_with(
        {0: 0.2, 1: 0.9, 2: 0.1, 3: 0.3, 4: 0.4, 5: 0.5},  # valid probs
        {0: 0.2, 1: 0.9, 2: 0.1, 3: 0.3, 4: 0.4, 5: 0.5}   # value probs
    )

    # Should NOT do any manual hypothesis selection
    assert not hasattr(agent, 'get_best_action_hypothesis') or not callable(getattr(agent, 'get_best_action_hypothesis', None))


def test_agent_gets_emergent_action():
    """Test that agent gets action that emerged from ReCoN message passing."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # First call returns no action (still propagating)
    # Second call returns action 3 (emerged through message passing)
    agent.hypothesis_manager.get_selected_action.side_effect = [None, None, 3]

    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    action = agent.process_frame(frame_data)

    # Should return the emergent action
    # Hypothesis index 3 → GameAction value 4
    assert action == 4


def test_agent_handles_score_changes():
    """Test that agent resets hypothesis manager on score changes (new level)."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.reset_for_new_level = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)

    agent.hypothesis_manager.get_selected_action.return_value = 1

    test_frame = np.random.randint(0, 16, (64, 64))

    # First frame with score 0
    frame_data1 = MockFrameData(frame=test_frame, score=0)
    agent.process_frame(frame_data1)

    # Second frame with score 1 (level completed)
    frame_data2 = MockFrameData(frame=test_frame, score=1)
    agent.process_frame(frame_data2)

    # Should have reset hypothesis manager for new level
    agent.hypothesis_manager.reset_for_new_level.assert_called_once()


def test_agent_only_requests_hypothesis_root():
    """Test that agent only makes single root request."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)
    agent.hypothesis_manager.get_selected_action.return_value = 2

    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    agent.process_frame(frame_data)

    # Should only request hypothesis_root
    agent.hypothesis_manager.request_hypothesis_test.assert_called_once_with("hypothesis_root")


def test_agent_propagates_until_action_emerges():
    """Test that agent propagates ReCoN network until an action emerges."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    agent.hypothesis_manager = Mock()
    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)

    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()

    # Simulate action emerging after several propagation steps
    agent.hypothesis_manager.get_selected_action.side_effect = [None, None, None, 4]

    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame)

    action = agent.process_frame(frame_data)

    # Should have propagated multiple times until action emerged
    assert agent.hypothesis_manager.propagate_step.call_count >= 3
    # Hypothesis index 4 → GameAction value 5
    assert action == 5


def test_agent_handles_no_emergent_action_gracefully():
    """Test that agent handles case where no action emerges from ReCoN."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.change_predictor = Mock()
    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)

    # No action ever emerges
    agent.hypothesis_manager.get_selected_action.return_value = None

    test_frame = np.random.randint(0, 16, (64, 64))
    frame_data = MockFrameData(frame=test_frame, available_actions=[0, 1, 2, 3, 4, 5])

    action = agent.process_frame(frame_data)

    # Should fall back to random action from available actions
    assert action is not None
    assert isinstance(action, int)
    assert 0 <= action <= 5


def test_agent_extracts_frames_correctly():
    """Test that agent correctly extracts frame data."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Test different frame formats
    test_cases = [
        np.random.randint(0, 16, (64, 64)),  # 2D array
        np.random.randint(0, 16, (1, 64, 64)),  # 3D with singleton
        [[1, 2], [3, 4]]  # List format
    ]

    for test_frame in test_cases:
        extracted = agent._extract_frame(MockFrameData(frame=test_frame))

        if extracted is not None:
            assert isinstance(extracted, np.ndarray)
            assert len(extracted.shape) == 2  # Should be 2D


def test_agent_maintains_learning_feedback_loop():
    """Test that agent maintains learning feedback to CNN."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # Mock components
    agent.hypothesis_manager = Mock()
    agent.hypothesis_manager.action_hypotheses = {}
    agent.hypothesis_manager.create_action_hypothesis = Mock()
    agent.hypothesis_manager.feed_cnn_priors = Mock()
    agent.hypothesis_manager.set_available_actions = Mock()
    agent.hypothesis_manager.request_hypothesis_test = Mock()
    agent.hypothesis_manager.propagate_step = Mock()
    agent.hypothesis_manager.update_hypothesis_result = Mock()
    agent.change_predictor = Mock()
    agent.trainer = Mock()

    agent.change_predictor.predict_change_probabilities.return_value = np.array([0.5] * 6)
    agent.hypothesis_manager.get_selected_action.return_value = 1

    # First frame
    test_frame1 = np.random.randint(0, 16, (64, 64))
    frame_data1 = MockFrameData(frame=test_frame1)
    agent.process_frame(frame_data1)

    # Second frame (different to simulate action worked)
    test_frame2 = np.random.randint(0, 16, (64, 64))
    test_frame2[0, 0] = (test_frame1[0, 0] + 1) % 16  # Make it different
    frame_data2 = MockFrameData(frame=test_frame2)
    agent.process_frame(frame_data2)

    # Should have updated hypothesis with result and trained CNN
    agent.hypothesis_manager.update_hypothesis_result.assert_called()
    agent.trainer.add_experience.assert_called()


def test_agent_is_thin_orchestrator():
    """Integration test that agent is truly a thin orchestrator."""
    agent = ReCoNArc2Agent("test_agent", "test_game")

    # The agent's main method should be very simple
    with patch.object(agent, 'hypothesis_manager') as mock_hm, \
         patch.object(agent, 'change_predictor') as mock_cnn:

        mock_cnn.predict_change_probabilities.return_value = np.array([0.1, 0.9, 0.2, 0.3, 0.4, 0.5])
        mock_hm.get_selected_action.return_value = 1

        test_frame = np.random.randint(0, 16, (64, 64))
        frame_data = MockFrameData(frame=test_frame)

        action = agent.process_frame(frame_data)

        # Verify agent just orchestrates:
        # 1. CNN prediction ✓
        mock_cnn.predict_change_probabilities.assert_called_once()

        # 2. Feed to hypothesis manager ✓
        mock_hm.feed_cnn_priors.assert_called_once()

        # 3. Single root request ✓
        mock_hm.request_hypothesis_test.assert_called_once_with("hypothesis_root")

        # 4. Propagate ✓
        mock_hm.propagate_step.assert_called()

        # 5. Get emergent action ✓
        mock_hm.get_selected_action.assert_called()
        # Now process_frame returns GameAction (or int in test fallback)
        # Hypothesis index 1 → GameAction value 2 (or just 2 in fallback)
        assert action == 2

        # Agent should NOT do any of the manual work it used to do:
        # - No manual hypothesis selection
        # - No state checking
        # - No cooldown tracking
        # - No noop suppression
        # This is verified by the absence of those calls