"""
BlindSquirrel Agent Implementation

Main agent class that implements the BlindSquirrel architecture using ReCoN platform.
"""

import random
import time
from typing import Any, Dict, Optional, List
import torch
import numpy as np

from recon_agents.base_agent import ReCoNBaseAgent
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from recon_engine.neural_terminal import NeuralTerminal
from .state_graph import BlindSquirrelStateGraph, BlindSquirrelState
from .models import BlindSquirrelActionModel, ActionEncoder


class BlindSquirrelAgent(ReCoNBaseAgent):
    """
    BlindSquirrel agent implemented as a ReCoN application.

    Faithfully reproduces the 2nd place ARC-AGI-3 winner architecture
    using the ReCoN platform as the cognitive foundation.
    """

    # Constants from original implementation
    MAX_ACTIONS = 50000
    LOOP_SLEEP = 0.1
    EPSILON = 0.5  # Exploration parameter
    RWEIGHT_MIN = 0.1
    RWEIGHT_RANK_DISCOUNT = 0.5
    RWEIGHT_NO_DISCOUNT = 0.5

    def __init__(self, agent_id: str = "blindsquirrel", game_id: str = "default"):
        # Initialize BlindSquirrel-specific state FIRST
        # NOTE: StateGraph is created per-game, not once in __init__
        self.state_graph = None  # Will be created when game starts
        self.current_state = None
        self.prev_state = None  # Will be set to 'NOT_PLAYED' string
        self.prev_action = None

        # Counters
        self.game_counter = 0
        self.level_counter = 0

        # Game tracking
        self.game_id = None  # Set when game starts

        # First frame flag
        self.is_first_frame = True

        # THEN call parent init (which calls _build_architecture)
        super().__init__(agent_id, game_id)

    def _build_architecture(self):
        """Build ReCoN graph representing BlindSquirrel architecture."""

        # Root controller
        root = self.add_script_node("blindsquirrel_root")

        # State processing pipeline
        state_processor = self.add_script_node("state_processor")
        action_selector = self.add_script_node("action_selector")

        # Decision branches
        model_branch = self.add_script_node("model_branch")
        rules_branch = self.add_script_node("rules_branch")

        # Connect the pipeline
        self.connect_nodes("blindsquirrel_root", "state_processor", "sub")
        self.connect_nodes("state_processor", "action_selector", "por")

        # Parallel action selection branches
        self.connect_nodes("action_selector", "rules_branch", "sub")
        self.connect_nodes("action_selector", "model_branch", "sub")

        # Value prediction terminal (when available)
        # Note: action_model is only available after training begins per-game
        if (self.state_graph and
            hasattr(self.state_graph, 'action_model') and
            self.state_graph.action_model):
            value_terminal = self.add_neural_terminal(
                "value_predictor",
                self.state_graph.action_model,
                "value"
            )
            self.connect_nodes("model_branch", "value_predictor", "sub")

    def process_latest_frame(self, latest_frame: Any):
        """
        Process new frame (separate from action selection like original).

        Equivalent to original process_latest_frame method.
        """
        time.sleep(self.LOOP_SLEEP)

        # Handle NOT_PLAYED state (like original - don't set game_id here)
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'NOT_PLAYED':
            self.prev_state = 'NOT_PLAYED'
            return

        # Handle GAME_OVER state
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'GAME_OVER':
            return

        # Handle active game states (NOT_FINISHED, WIN)
        if self.prev_state == 'NOT_PLAYED' or self.is_first_frame:
            # Initialize new game - create fresh StateGraph
            self.game_counter = 0
            self.level_counter = 0
            self.game_id = getattr(latest_frame, 'game_id', 'default')

            # Create NEW StateGraph for this game (like original)
            self.state_graph = BlindSquirrelStateGraph()
            self.current_state = self.state_graph.get_state(latest_frame)
            self.state_graph.add_init_state(self.current_state)

            self.is_first_frame = False
            return

        # Regular game step processing
        new_state = self.state_graph.get_state(latest_frame)

        # Update counters
        self.game_counter += 1
        if new_state.score > (self.current_state.score if self.current_state else -1):
            self.level_counter = 0
        else:
            self.level_counter += 1

        # Update state graph with transition
        if self.current_state and self.prev_action is not None:
            self.state_graph.update(self.current_state, self.prev_action, new_state)

        # Update current state for next iteration
        self.current_state = new_state

    def process_frame(self, frame_data: Any) -> Any:
        """
        Process new frame and return action.

        Equivalent to original process_latest_frame + choose_action.
        """
        # First process the frame (state updates)
        self.process_latest_frame(frame_data)

        # Then choose action if not in special states
        if hasattr(frame_data, 'state'):
            if str(frame_data.state) in ('NOT_PLAYED', 'GAME_OVER'):
                return None

        # Choose action for active game states
        return self._choose_action(frame_data)


    def _choose_action(self, frame_data: Any) -> Any:
        """Choose action using BlindSquirrel strategy."""
        if not self.current_state:
            return self._get_default_action()

        # Decide between model-based and rule-based selection
        use_model = (
            random.random() > self.EPSILON and
            hasattr(frame_data, 'score') and
            getattr(frame_data, 'score', 0) > 0 and
            len(self.current_state.future_states) > 0 and
            self.state_graph.action_model is not None
        )

        if use_model:
            action_idx = self._get_model_action()
        else:
            action_idx = self._get_rweights_action()

        # Convert to action object
        try:
            action_obj = self.current_state.get_action_obj(action_idx)
        except ImportError:
            # Handle GameAction import issue - return fallback
            return self._get_default_action()

        # Update state for next iteration (important for next frame processing)
        self.prev_state = self.current_state
        self.prev_action = action_idx

        return action_obj

    def _get_model_action(self) -> int:
        """Get action using neural value model."""
        if not self.current_state or not self.state_graph.action_model:
            return self._get_rweights_action()

        model_values = {}

        # Evaluate all valid actions
        for action_idx, rweight in self.current_state.action_rweights.items():
            if rweight == 0:  # Skip disabled actions
                continue

            # Get model prediction
            try:
                value = self.state_graph.predict_action_value(self.current_state, action_idx)

                # Apply rule-based weighting if needed
                if rweight is None:
                    weight = self._calculate_rweight(action_idx)
                    value *= weight

                model_values[action_idx] = value
            except Exception as e:
                print(f"Error predicting value for action {action_idx}: {e}")
                continue

        if not model_values:
            print(f'Warning: No Actions {self.game_id}')
            return self._get_random_action()

        # Return action with highest value
        return max(model_values, key=model_values.get)

    def _get_rweights_action(self) -> int:
        """Get action using rule-based weights."""
        if not self.current_state:
            return self._get_random_action()

        actions = []
        weights = []

        # Collect valid actions and their weights
        for action_idx, rweight in self.current_state.action_rweights.items():
            if rweight == 0:  # Skip disabled actions
                continue

            if rweight is None:
                weight = self._calculate_rweight(action_idx)
            else:
                weight = rweight

            actions.append(action_idx)
            weights.append(weight)

        if not actions:
            print(f'Warning: No Actions {self.game_id}')
            return self._get_random_action()

        # Weighted random selection
        try:
            return random.choices(actions, weights=weights, k=1)[0]
        except Exception:
            return random.choice(actions)

    def _calculate_rweight(self, action_idx: int) -> float:
        """Calculate rule-based weight for action."""
        game_id = self.current_state.game_id
        score = self.current_state.score

        # Get action statistics
        counter_key = (game_id, score, action_idx)
        failures, successes = self.state_graph.action_counter.get(counter_key, [0, 0])

        if successes > 0:
            # Use success rate
            weight = max(self.RWEIGHT_MIN, successes / (failures + successes))
        else:
            # Use heuristic based on action type
            if action_idx < 5:
                # Basic actions
                base_weight = 1.0
            else:
                # Click actions - discount by rank
                base_weight = self.RWEIGHT_RANK_DISCOUNT ** (action_idx - 5)

            # Apply failure discount
            weight = max(
                self.RWEIGHT_MIN,
                base_weight * (self.RWEIGHT_NO_DISCOUNT ** failures)
            )

        return weight

    def _get_random_action(self) -> int:
        """Get random action as fallback."""
        if self.current_state and self.current_state.num_actions > 0:
            return random.randint(0, self.current_state.num_actions - 1)
        else:
            return 0  # Default to first action

    def _get_default_action(self):
        """Get default action when no state is available."""
        # Return action index instead of GameAction object
        return 0  # ACTION1

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Check if agent is done processing."""
        # Process frame first (like original)
        self.process_latest_frame(latest_frame)

        # Check for WIN condition
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'WIN':
            return True

        return False

    def train_model_if_needed(self, max_score: int):
        """Train model if conditions are met."""
        if max_score > 0:
            self.state_graph.train_model(self.game_id, max_score)

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the agent."""
        base_stats = self.get_performance_metrics()

        agent_stats = {
            'game_counter': self.game_counter,
            'level_counter': self.level_counter,
            'current_score': self.current_state.score if self.current_state else 0,
            'epsilon': self.EPSILON,
            'has_model': self.state_graph.action_model is not None
        }

        graph_stats = self.state_graph.get_statistics()

        return {
            **base_stats,
            **agent_stats,
            **graph_stats
        }

    def reset(self):
        """Reset agent to initial state."""
        super().reset()

        # Reset BlindSquirrel-specific state
        self.current_state = None
        self.prev_state = None
        self.prev_action = None
        self.game_counter = 0
        self.level_counter = 0
        self.is_first_frame = True

        # Keep the state graph but reset current game
        # (In original BlindSquirrel, the state graph persists across games)


def create_blindsquirrel_agent(game_id: str = "test") -> BlindSquirrelAgent:
    """Factory function to create a BlindSquirrel agent."""
    return BlindSquirrelAgent("blindsquirrel", game_id)


# Example usage and testing functions
def test_blindsquirrel_agent():
    """Test BlindSquirrel agent functionality."""
    agent = create_blindsquirrel_agent("test")

    # Mock frame data
    class MockFrame:
        def __init__(self, game_id, score, frame_data, state="NOT_FINISHED"):
            self.game_id = game_id
            self.score = score
            self.frame = [frame_data]
            self.state = state
            self.available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]

    # Test initial frame
    initial_frame = MockFrame("test", 0, [[0] * 64 for _ in range(64)], "NOT_PLAYED")
    agent.process_frame(initial_frame)

    # Test game frame
    game_frame = MockFrame("test", 0, [[1] * 64 for _ in range(64)])
    action = agent.process_frame(game_frame)

    print(f"BlindSquirrel agent selected action: {action}")
    print(f"Agent statistics: {agent.get_agent_statistics()}")

    return agent


if __name__ == "__main__":
    test_blindsquirrel_agent()