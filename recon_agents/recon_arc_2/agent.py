"""
ReCoN ARC-2 Thin Orchestrator Agent

Pure orchestrator that delegates ALL active perception logic to the hypothesis manager.
The agent only:
1. Extracts frames
2. Gets CNN predictions
3. Feeds priors to hypothesis manager
4. Requests root and propagates
5. Returns emergent action

NO manual state control, hypothesis selection, or cooldown tracking.
"""

import sys
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Add paths for imports
sys.path.insert(0, '/workspace/recon-platform')
sys.path.insert(0, '/workspace/recon-platform/ARC-AGI-3-Agents')

from recon_agents.base_agent import ReCoNBaseAgent
from .perception import ChangePredictor, ChangePredictorTrainer
from .hypothesis import HypothesisManager

# Import harness types
try:
    from agents.structs import GameAction
except ImportError:
    # Fallback if harness not available (for tests)
    class MockGameAction:
        @classmethod
        def from_id(cls, action_id: int):
            return action_id
    GameAction = MockGameAction


class ReCoNArc2ThinAgent(ReCoNBaseAgent):
    """
    Thin ReCoN ARC-2 agent - pure orchestrator.

    Delegates all active perception logic to HypothesisManager.
    """

    def __init__(self, agent_id: str = "recon_arc_2", game_id: str = "default"):
        # Initialize components before calling parent
        self.change_predictor = None
        self.trainer = None
        self.hypothesis_manager = None

        # Game state tracking (minimal)
        self.current_frame = None
        self.previous_frame = None
        self.last_action = None
        self.score = 0

        # Initialize parent
        super().__init__(agent_id, game_id)

        # Initialize neural components and hypothesis manager
        self._initialize_components()

    def _build_architecture(self):
        """Build minimal ReCoN architecture - hypothesis manager handles the rest."""
        pass  # HypothesisManager builds its own pure ReCoN architecture

    def _initialize_components(self):
        """Initialize CNN predictor and hypothesis manager."""
        self.change_predictor = ChangePredictor(input_channels=16, grid_size=64)
        self.trainer = ChangePredictorTrainer(self.change_predictor)
        self.hypothesis_manager = HypothesisManager()

    def process_frame(self, frame_data: Any) -> Any:
        """
        Pure orchestrator frame processing.

        1. Extract frame and handle game state
        2. Get CNN predictions
        3. Feed priors to hypothesis manager
        4. Single root request
        5. Propagate until action emerges
        6. Return emergent action
        """
        # 1. Extract frame and handle basic game state
        self.previous_frame = self.current_frame
        self.current_frame = self._extract_frame(frame_data)

        # Handle score changes (new level)
        if hasattr(frame_data, 'score') and frame_data.score != self.score:
            self.hypothesis_manager.reset_for_new_level()
            self.score = frame_data.score

        # Handle special game states
        if hasattr(frame_data, 'state'):
            state_str = str(frame_data.state)
            if state_str in ('NOT_PLAYED', 'GAME_OVER', 'WIN'):
                action_idx = self._handle_special_state(state_str)  # Returns hypothesis index
                self.last_action = action_idx  # Store for learning
                return action_idx  # Return hypothesis index for adapter conversion

        # Learning feedback from previous action
        if self.last_action is not None and self.previous_frame is not None:
            self._update_learning(self.last_action)

        # If no current frame, return random action
        if self.current_frame is None:
            allowed = self._get_allowed_actions(frame_data)
            action_idx = self._get_random_action(allowed)  # Returns hypothesis index
            self.last_action = action_idx  # Store for learning
            return action_idx  # Return hypothesis index for adapter conversion

        # 2. Get CNN predictions
        change_probs = self.change_predictor.predict_change_probabilities(self.current_frame)

        # 3. Feed priors to hypothesis manager (create hypotheses if needed)
        self._ensure_hypotheses_exist(change_probs, frame_data)
        priors_dict = {i: float(change_probs[i]) for i in range(len(change_probs))}
        self.hypothesis_manager.feed_cnn_priors(priors_dict, priors_dict)

        # Set available actions
        allowed_actions = self._get_allowed_actions(frame_data)
        self.hypothesis_manager.set_available_actions(allowed_actions)

        # 4. Single root request - let ReCoN handle everything
        self.hypothesis_manager.request_hypothesis_test("hypothesis_root")

        # 5. Propagate until action emerges (with reasonable limit)
        for _ in range(20):
            self.hypothesis_manager.propagate_step()
            selected_action = self.hypothesis_manager.get_selected_action()
            if selected_action is not None:
                break
        else:
            # Fallback if no action emerges
            selected_action = self._get_random_action(allowed_actions)

        # 6. Store for learning and return raw action index
        self.last_action = selected_action  # Store hypothesis index (0-5) for internal use
        return selected_action  # Return hypothesis index for adapter conversion

    def _extract_frame(self, frame_data: Any) -> Optional[np.ndarray]:
        """Extract frame as numpy array."""
        try:
            if hasattr(frame_data, 'frame') and frame_data.frame is not None:
                frame = np.array(frame_data.frame)
                if len(frame.shape) == 2:
                    return frame
                elif len(frame.shape) == 3 and frame.shape[0] == 1:
                    return frame[0]
            return None
        except Exception:
            return None

    def _ensure_hypotheses_exist(self, change_probs: np.ndarray, frame_data: Any):
        """Ensure action hypotheses exist for all actions."""
        for i in range(len(change_probs)):
            if i not in self.hypothesis_manager.action_hypotheses:
                self.hypothesis_manager.create_action_hypothesis(
                    i, float(change_probs[i]), self.current_frame
                )

    def _get_allowed_actions(self, frame_data: Any) -> List[int]:
        """Get allowed actions as hypothesis indices (0-5)."""
        if hasattr(frame_data, 'available_actions') and frame_data.available_actions:
            # Convert GameAction values to hypothesis indices
            return [self._gameaction_value_to_hypothesis_index(action.value)
                   for action in frame_data.available_actions]
        return list(range(6))  # Default: all hypothesis indices (0-5)

    def _get_random_action(self, allowed_actions: List[int]) -> int:
        """Get random action from allowed actions."""
        if allowed_actions:
            return int(np.random.choice(allowed_actions))
        return int(np.random.randint(0, 6))

    def _update_learning(self, action_idx: int):
        """Update learning with previous action result."""
        if self.current_frame is None or self.previous_frame is None:
            return

        # Check if frame changed
        frame_changed = not np.array_equal(self.current_frame, self.previous_frame)

        # Update hypothesis manager
        self.hypothesis_manager.update_hypothesis_result(action_idx, frame_changed)

        # Add experience to CNN trainer
        self.trainer.add_experience(self.previous_frame, action_idx, frame_changed)

        # Periodic training
        if hasattr(self, 'action_count'):
            self.action_count = getattr(self, 'action_count', 0) + 1
            if self.action_count % 20 == 0:
                self.trainer.train_step()

    def _handle_special_state(self, state_str: str) -> int:
        """Handle special game states."""
        if state_str in ('NOT_PLAYED', 'GAME_OVER'):
            return 0  # RESET action
        return 0

    def propose_click_coordinates(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Delegate coordinate selection to hypothesis manager for ACTION6.
        """
        # Use hypothesis manager's region selection logic
        if hasattr(self.hypothesis_manager, 'propose_click_coordinates'):
            return self.hypothesis_manager.propose_click_coordinates(frame)

        # Simple fallback - center of frame
        return 32, 32

    def _gameaction_value_to_hypothesis_index(self, gameaction_value: int) -> int:
        """Convert GameAction value to hypothesis index (0-5)."""
        # GameAction values: 0=RESET, 1=ACTION1, 2=ACTION2, 3=ACTION3, 4=ACTION4, 5=ACTION5, 6=ACTION6
        # Hypothesis indices: 0=ACTION1, 1=ACTION2, 2=ACTION3, 3=ACTION4, 4=ACTION5, 5=ACTION6
        if gameaction_value == 0:  # RESET
            return 0  # Map to ACTION1 hypothesis
        elif 1 <= gameaction_value <= 6:
            return gameaction_value - 1  # ACTION1(1)→0, ACTION2(2)→1, ..., ACTION6(6)→5
        else:
            return 0  # Fallback to ACTION1

    def _hypothesis_index_to_gameaction_value(self, hypothesis_index: int) -> int:
        """Convert hypothesis index (0-5) to GameAction value."""
        # Hypothesis indices: 0=ACTION1, 1=ACTION2, 2=ACTION3, 3=ACTION4, 4=ACTION5, 5=ACTION6
        # GameAction values: 1=ACTION1, 2=ACTION2, 3=ACTION3, 4=ACTION4, 5=ACTION5, 6=ACTION6
        if 0 <= hypothesis_index <= 5:
            return hypothesis_index + 1  # 0→1(ACTION1), 1→2(ACTION2), ..., 5→6(ACTION6)
        else:
            return 1  # Fallback to ACTION1

    def _convert_to_game_action(self, hypothesis_index: int) -> Any:
        """Convert hypothesis index to GameAction with coordinates if needed."""
        # Convert hypothesis index to GameAction value, then to GameAction enum
        gameaction_value = self._hypothesis_index_to_gameaction_value(hypothesis_index)
        action = GameAction.from_id(gameaction_value)

        # Only set data for ACTION6 (click actions) - matches recon_arc_1 pattern
        if hasattr(action, 'value') and action.value == 6 and self.current_frame is not None:
            x, y = self.propose_click_coordinates(self.current_frame)
            if hasattr(action, 'set_data'):
                action.set_data({"x": int(x), "y": int(y)})

        return action

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Check if agent is done (WIN state)."""
        if hasattr(latest_frame, 'state'):
            return str(latest_frame.state) == 'WIN'
        return False

    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """Choose action - delegates to process_frame."""
        return self.process_frame(latest_frame)

    def process_latest_frame(self, latest_frame: Any) -> None:
        """Process frame for harness interface - stores result for later retrieval."""
        # Process frame and store result
        self._last_action_result = self.process_frame(latest_frame)

    def get_action_from_processed_frame(self) -> Any:
        """Get action result from previously processed frame."""
        if hasattr(self, '_last_action_result'):
            return self._last_action_result
        # Fallback - return random action
        return 0

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about agent state."""
        info = {
            'agent_type': 'ReCoN ARC-2 Thin',
            'game_id': self.game_id,
            'score': self.score,
            'has_current_frame': self.current_frame is not None,
            'hypothesis_count': len(self.hypothesis_manager.action_hypotheses) if self.hypothesis_manager else 0
        }

        # Add hypothesis manager debug info if available
        if self.hypothesis_manager and hasattr(self.hypothesis_manager, 'get_debug_info'):
            info.update(self.hypothesis_manager.get_debug_info())

        return info


# Compatibility alias
ReCoNArc2Agent = ReCoNArc2ThinAgent