"""
ReCoN ARC-2 Agent - Active Perception for ARC Puzzle Solving

Uses hypothesis testing with CNN change prediction for active exploration.
Minimal architecture focused on ReCoN's core strengths.
"""

import sys
import os
import random
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Add paths for imports
sys.path.insert(0, '/workspace/recon-platform')

from recon_agents.base_agent import ReCoNBaseAgent
from .perception import ChangePredictor, ChangePredictorTrainer
from .hypothesis import HypothesisManager, ActionHypothesis


class ReCoNArc2Agent(ReCoNBaseAgent):
    """
    ReCoN ARC-2 agent using active perception for puzzle solving.

    Core concept: Form hypotheses about productive actions, test them
    via active exploration, and build hierarchical understanding.
    """

    def __init__(self, agent_id: str = "recon_arc_2", game_id: str = "default"):
        # Initialize components before calling parent
        self.change_predictor = None
        self.trainer = None
        self.hypothesis_manager = None

        # Game state tracking
        self.current_frame = None
        self.previous_frame = None
        self.game_id = None
        self.action_count = 0
        self.score = 0

        # Action tracking for learning
        self.last_action = None
        self.waiting_for_result = False

        # Active perception parameters
        self.exploration_threshold = 0.3  # Min probability to consider action
        self.max_hypotheses_per_frame = 3  # Limit hypothesis generation

        # Initialize parent (builds ReCoN architecture)
        super().__init__(agent_id, game_id)

        # Initialize neural components
        self._initialize_neural_components()

    def _build_architecture(self):
        """Build ReCoN architecture for active perception."""
        # Create hypothesis manager (contains ReCoN graph)
        self.hypothesis_manager = HypothesisManager()

        # Root perception controller
        root = self.add_script_node("perception_root")

        # Connect to hypothesis manager's graph
        # The hypothesis manager handles the detailed ReCoN structure

    def _initialize_neural_components(self):
        """Initialize CNN change predictor and trainer."""
        self.change_predictor = ChangePredictor(input_channels=16, grid_size=64)
        self.trainer = ChangePredictorTrainer(self.change_predictor)

    def process_frame(self, frame_data: Any) -> Any:
        """
        Process new frame using active perception.

        Active Perception Loop:
        1. Check if previous action worked (learning feedback)
        2. Extract features and predict changes
        3. Generate hypotheses about productive actions
        4. Test highest-confidence hypothesis
        5. Update understanding based on results
        """
        # Store previous frame for change detection
        self.previous_frame = self.current_frame
        self.current_frame = self._extract_frame(frame_data)

        # Update game state
        if hasattr(frame_data, 'score'):
            self.score = frame_data.score
        if hasattr(frame_data, 'game_id'):
            self.game_id = frame_data.game_id

        # Learning feedback: check if last action worked
        if self.waiting_for_result and self.last_action is not None:
            self._process_action_feedback()

        # Handle special game states
        if hasattr(frame_data, 'state'):
            state_str = str(frame_data.state)
            if state_str in ('NOT_PLAYED', 'GAME_OVER', 'WIN'):
                return self._handle_special_state(state_str)

        # Active perception: generate and test hypotheses
        action = self._active_perception_step(frame_data)

        # Track this action for learning feedback
        self.last_action = action
        self.waiting_for_result = True

        self.action_count += 1
        return action

    def _extract_frame(self, frame_data: Any) -> Optional[np.ndarray]:
        """
        Extract frame as numpy array for processing.

        Args:
            frame_data: Frame from game environment

        Returns:
            frame: numpy array (H, W) with values 0-15, or None
        """
        try:
            if hasattr(frame_data, 'frame') and frame_data.frame is not None:
                frame = np.array(frame_data.frame)
                # Ensure we have a 2D array
                if len(frame.shape) == 2:
                    return frame
                elif len(frame.shape) == 3 and frame.shape[0] == 1:
                    return frame[0]  # Remove singleton dimension

            return None

        except Exception as e:
            print(f"Error extracting frame: {e}")
            return None

    def _active_perception_step(self, frame_data: Any) -> int:
        """
        Core active perception loop.

        Returns:
            action: Action index to execute
        """
        # Determine allowed actions from the harness (exclude ACTION6 until coords supported)
        allowed_indices = self._allowed_action_indices(frame_data)

        if self.current_frame is None:
            return self._get_random_action(allowed_indices)

        # 1. Use CNN to predict which actions might cause changes
        change_probs = self.change_predictor.predict_change_probabilities(self.current_frame)

        # 2. Generate hypotheses for promising actions (only allowed)
        self._generate_action_hypotheses(change_probs, allowed_indices)

        # 3. Select best hypothesis to test
        best_hypothesis = self.hypothesis_manager.get_best_action_hypothesis()

        if best_hypothesis is None:
            return self._get_random_action(allowed_indices)

        # Ensure the selected hypothesis' action is allowed
        if best_hypothesis.action_idx not in allowed_indices:
            return self._get_random_action(allowed_indices)

        # 4. Request testing of the hypothesis
        self.hypothesis_manager.request_hypothesis_test(best_hypothesis.id)

        # 5. Process ReCoN messages
        for _ in range(3):  # Limited message passing
            self.hypothesis_manager.propagate_step()

        # Return the action to test
        return best_hypothesis.action_idx

    def _process_action_feedback(self):
        """
        Process feedback from the last action to enable learning.

        Compares current frame with previous frame to see if action worked.
        """
        if self.current_frame is None or self.previous_frame is None or self.last_action is None:
            self.waiting_for_result = False
            return

        # Check if frame actually changed
        try:
            frame_changed = not np.array_equal(self.current_frame, self.previous_frame)

            # Update hypothesis with result
            self.hypothesis_manager.update_hypothesis_result(self.last_action, frame_changed)

            # Add experience for neural network training
            self.trainer.add_experience(self.previous_frame, self.last_action, frame_changed)

            # Debug output
            if self.action_count % 50 == 0:  # Every 50 actions
                print(f"ReCoN ARC-2: Action {self.last_action}, Changed: {frame_changed}, Buffer: {len(self.trainer.experience_buffer)}")

            # Periodic training
            if self.action_count % 20 == 0:  # Train every 20 actions
                loss = self.trainer.train_step()
                if loss is not None:
                    print(f"ReCoN ARC-2 Training loss: {loss:.4f}")

                    # Debug hypothesis states
                    debug_info = self.hypothesis_manager.get_debug_info()
                    print(f"Hypotheses: {debug_info.get('num_action_hypotheses', 0)}")
                    for action_idx, conf_info in debug_info.get('action_confidences', {}).items():
                        print(f"  Action {action_idx}: conf={conf_info['confidence']:.3f}, tests={conf_info['confirmations']+conf_info['failures']}")

        except Exception as e:
            print(f"Error in action feedback: {e}")

        self.waiting_for_result = False

    def _generate_action_hypotheses(self, change_probs: np.ndarray, allowed_indices: List[int]):
        """
        Generate hypotheses for actions with high change probability.

        Args:
            change_probs: Array of shape (6,) with change probabilities
        """
        # Sort actions by predicted change probability
        sorted_actions = np.argsort(change_probs)[::-1]  # Descending order

        hypotheses_created = 0

        for action_idx in sorted_actions:
            # Only consider allowed simple actions (0..4). ACTION6 (5) is excluded here.
            if action_idx not in allowed_indices:
                continue
            prob = change_probs[action_idx]

            # Only create hypothesis if probability is above threshold
            if prob < self.exploration_threshold:
                break

            # Limit number of hypotheses per frame
            if hypotheses_created >= self.max_hypotheses_per_frame:
                break

            # Create hypothesis if it doesn't exist or update existing one
            if action_idx not in self.hypothesis_manager.action_hypotheses:
                self.hypothesis_manager.create_action_hypothesis(
                    action_idx, prob, self.current_frame
                )
                hypotheses_created += 1

    def update_with_result(self, action_idx: int, new_frame_data: Any):
        """
        Update hypothesis with actual result after action execution.

        Args:
            action_idx: Action that was executed
            new_frame_data: Frame after action execution
        """
        # Extract new frame
        new_frame = self._extract_frame(new_frame_data)

        if new_frame is None or self.current_frame is None:
            return

        # Check if frame actually changed
        frame_changed = not np.array_equal(self.current_frame, new_frame)

        # Update hypothesis with result
        self.hypothesis_manager.update_hypothesis_result(action_idx, frame_changed)

        # Add experience for neural network training
        self.trainer.add_experience(self.current_frame, action_idx, frame_changed)

        # Periodic training
        if self.action_count % 10 == 0:
            loss = self.trainer.train_step()
            if loss is not None:
                print(f"Training loss: {loss:.4f}")

        # Update current frame
        self.current_frame = new_frame

    def _handle_special_state(self, state_str: str) -> int:
        """Handle special game states."""
        if state_str == 'NOT_PLAYED':
            return 0  # ACTION1
        elif state_str in ('GAME_OVER', 'WIN'):
            # Reset for next game
            self._reset_for_new_game()
            return 0  # ACTION1
        else:
            return self._get_random_action([0, 1, 2, 3, 4])

    def _get_random_action(self, allowed_indices: List[int]) -> int:
        """Get a random allowed action as fallback."""
        if not allowed_indices:
            allowed_indices = [0, 1, 2, 3, 4]
        return random.choice(allowed_indices)

    def _allowed_action_indices(self, frame_data: Any) -> List[int]:
        """
        Compute allowed action indices based on latest_frame.available_actions.
        - Map ACTION1..ACTION6 to indices 0..5.
        - Suppress ACTION6 (index 5) until coordinate policy is implemented.
        - If unavailable, default to all simple actions [0..4].
        """
        try:
            actions = getattr(frame_data, 'available_actions', None)
            allowed: List[int] = []
            if actions:
                for a in actions:
                    # GameAction has integer value IDs
                    val = getattr(a, 'value', None)
                    if isinstance(val, int):
                        idx = val - 1  # ACTION1..6 -> 0..5
                        if 0 <= idx <= 4:  # exclude ACTION6 (idx 5)
                            if idx not in allowed:
                                allowed.append(idx)
            if not allowed:
                return [0, 1, 2, 3, 4]
            return sorted(allowed)
        except Exception:
            return [0, 1, 2, 3, 4]

    def _reset_for_new_game(self):
        """Reset state for new game while keeping learned knowledge."""
        self.action_count = 0
        self.score = 0
        self.current_frame = None
        self.previous_frame = None
        self.last_action = None
        self.waiting_for_result = False

        # Keep neural network weights but clear experience buffer for new game
        if self.trainer:
            # Don't clear buffer - keep learning across games
            pass

        # Reset hypothesis manager but keep successful patterns
        if self.hypothesis_manager:
            # Clear current hypotheses but keep learned confidences
            self.hypothesis_manager.action_hypotheses.clear()

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Check if agent is done processing."""
        if hasattr(latest_frame, 'state'):
            return str(latest_frame.state) == 'WIN'
        return False

    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """Choose action (frame assumed already processed by is_done)."""
        action_idx = self.process_frame(latest_frame)

        # Update with result (simplified - in real usage this would be called after action execution)
        # For now, assume no change for action selection
        # The real update would happen in the game loop

        return action_idx

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about agent state."""
        info = {
            'agent_type': 'ReCoN ARC-2',
            'game_id': self.game_id,
            'action_count': self.action_count,
            'score': self.score,
            'has_current_frame': self.current_frame is not None,
            'training_buffer_size': len(self.trainer.experience_buffer) if self.trainer else 0
        }

        # Add hypothesis information
        if self.hypothesis_manager:
            info.update(self.hypothesis_manager.get_debug_info())

        return info

    def reset(self):
        """Reset agent state."""
        super().reset()
        self._reset_for_new_game()

        # Reinitialize neural components if needed
        if self.change_predictor is None:
            self._initialize_neural_components()