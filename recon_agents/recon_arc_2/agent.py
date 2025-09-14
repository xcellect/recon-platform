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

from recon_agents.base_agent import ReCoNBaseAgent
from .perception import ChangePredictor, ChangePredictorTrainer
from .hypothesis import HypothesisManager


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
                return self._handle_special_state(state_str)

        # Learning feedback from previous action
        if self.last_action is not None and self.previous_frame is not None:
            self._update_learning(self.last_action)

        # If no current frame, return random action
        if self.current_frame is None:
            allowed = self._get_allowed_actions(frame_data)
            return self._get_random_action(allowed)

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

        # 6. Return emergent action
        self.last_action = selected_action
        return selected_action

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
        """Get allowed actions from frame data."""
        if hasattr(frame_data, 'available_actions') and frame_data.available_actions:
            return [action.value for action in frame_data.available_actions]
        return list(range(6))  # Default: all actions

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


# Compatibility alias
ReCoNArc2Agent = ReCoNArc2ThinAgent