"""
ReCoN ARC-1 Agent Implementation

Main agent class implementing BlindSquirrel architecture using ReCoN platform.
Faithful reproduction with enhanced message passing coordination.
"""

import random
import time
from typing import Any, Dict, Optional, List
import torch
import numpy as np

from recon_agents.base_agent import ReCoNBaseAgent
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from recon_engine.neural_terminal import NeuralTerminal
from .state_graph import ReCoNArc1StateGraph, ReCoNArc1State
from .models import ReCoNArc1ActionModel, ReCoNArc1NeuralTerminal
from .valid_actions import ValidActionsNode, ValidActionsManager


class ReCoNArc1Agent(ReCoNBaseAgent):
    """
    ReCoN ARC-1 agent - faithful BlindSquirrel reproduction using ReCoN platform.

    Uses minimal ReCoN architecture:
    - State graph as explicit nodes
    - Valid actions as hybrid node
    - Action value model as neural terminal
    """

    # Constants from BlindSquirrel
    MAX_ACTIONS = 50000
    LOOP_SLEEP = 0.1
    EPSILON = 0.5  # Exploration parameter
    RWEIGHT_MIN = 0.1
    RWEIGHT_RANK_DISCOUNT = 0.5
    RWEIGHT_NO_DISCOUNT = 0.5

    def __init__(self, agent_id: str = "recon_arc_1", game_id: str = "default"):
        # Initialize ReCoN ARC-1 specific state
        self.state_graph = None  # Will be created when game starts
        self.current_state = None
        self.prev_state = None
        self.prev_action = None

        # Counters
        self.game_counter = 0
        self.level_counter = 0

        # Game tracking
        self.game_id = None

        # First frame flag
        self.is_first_frame = True

        # ReCoN components
        self.valid_actions_node = None
        self.valid_actions_manager = None
        self.neural_terminal = None

        # Shared knowledge repository (persistent across games)
        self.global_action_stats = {}  # (score, action) -> [failures, successes]
        self.global_model = None  # Shared neural model across games

        # Initialize parent (which calls _build_architecture)
        super().__init__(agent_id, game_id)

    def _build_architecture(self):
        """Build minimal ReCoN graph for BlindSquirrel functionality."""

        # Root controller
        root = self.add_script_node("recon_arc1_root")

        # State graph manager
        state_manager = self.add_script_node("state_manager")

        # Shared knowledge repository (persistent across games)
        knowledge_repo = self.add_script_node("knowledge_repository")

        # Valid actions node (hybrid - starts in explicit mode)
        self.valid_actions_node = ValidActionsNode("valid_actions")
        self.graph.add_node(self.valid_actions_node)
        self.valid_actions_manager = ValidActionsManager(self.valid_actions_node)

        # Neural terminal for action values (will be created when model is available)
        # Note: Neural terminal is added dynamically when model is trained

        # Connect the architecture
        self.connect_nodes("recon_arc1_root", "state_manager", "sub")
        self.connect_nodes("state_manager", "knowledge_repository", "sub")
        self.connect_nodes("knowledge_repository", "valid_actions", "sub")

        # Set up message flow
        # state_manager -> knowledge_repository (query global stats)
        # knowledge_repository -> valid_actions (provide baseline weights)
        # valid_actions -> neural_terminal (if available)

    def process_latest_frame(self, latest_frame: Any):
        """
        Process new frame (identical to BlindSquirrel flow).
        """
        time.sleep(self.LOOP_SLEEP)

        # Handle NOT_PLAYED state
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'NOT_PLAYED':
            self.prev_state = 'NOT_PLAYED'
            return

        # Handle GAME_OVER state
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'GAME_OVER':
            return

        # Handle active game states (NOT_FINISHED, WIN)
        if isinstance(self.prev_state, str) or self.is_first_frame:
            # Initialize new game - create fresh StateGraph
            self.game_counter = 0
            self.level_counter = 0
            self.game_id = getattr(latest_frame, 'game_id', 'default')

            # Clean up caches for new game to prevent memory growth
            self._cleanup_for_new_game()

            # Create NEW StateGraph for this game (with global knowledge access)
            self.state_graph = ReCoNArc1StateGraph(global_knowledge_provider=self)
            self.current_state = self.state_graph.get_state(latest_frame)
            self.state_graph.add_init_state(self.current_state)

            # Update valid actions manager
            self.valid_actions_manager.process_state_for_actions(
                self.current_state, self.state_graph
            )

            self.is_first_frame = False
            return

        # Regular game step processing
        prev_state_for_update = self.current_state

        # Get new current state
        self.current_state = self.state_graph.get_state(latest_frame)

        # Update counters
        self.game_counter += 1
        if self.current_state.score > (prev_state_for_update.score if prev_state_for_update else -1):
            self.level_counter = 0
        else:
            self.level_counter += 1

        # Update state graph with transition
        if prev_state_for_update and self.prev_action is not None:
            self.state_graph.update(prev_state_for_update, self.prev_action, self.current_state)

        # Update valid actions for current state
        self.valid_actions_manager.process_state_for_actions(
            self.current_state, self.state_graph
        )

        # Update neural terminal if model is available
        self._update_neural_terminal()

    def _update_neural_terminal(self):
        """Update or create neural terminal when model becomes available."""
        if self.state_graph and self.state_graph.action_model:
            if not self.neural_terminal:
                # Create neural terminal
                self.neural_terminal = ReCoNArc1NeuralTerminal(self.state_graph.action_model)

                # Add to ReCoN graph
                terminal_node = self.add_neural_terminal(
                    "action_value_terminal",
                    self.neural_terminal.model,
                    "value"
                )

                # Connect to valid actions
                self.connect_nodes("valid_actions", "action_value_terminal", "sub")

            else:
                # Update existing terminal
                self.neural_terminal.update_model(self.state_graph.action_model)

    def process_frame(self, frame_data: Any) -> Any:
        """
        Process new frame and return action.
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
        """Choose action using ReCoN-coordinated BlindSquirrel strategy."""
        if not self.current_state:
            return self._get_default_action()

        # Trigger ReCoN message passing
        self._send_recon_request()

        # Decide between model-based and rule-based selection (identical to BlindSquirrel)
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
        except Exception:
            return self._get_default_action()

        # Update state for next iteration
        self.prev_state = self.current_state
        self.prev_action = action_idx

        return action_obj

    def _update_global_knowledge(self, score: int, action: int, success: bool):
        """Update global action statistics across games with bounded growth."""
        # Prevent unbounded growth of global stats
        if len(self.global_action_stats) > 2000:  # Reasonable limit
            # Remove oldest/least useful entries
            items_to_remove = []
            for key, (failures, successes) in self.global_action_stats.items():
                # Remove entries with very low activity
                if failures + successes < 2:
                    items_to_remove.append(key)
                if len(items_to_remove) > 500:  # Remove up to 500 old entries
                    break

            for key in items_to_remove:
                del self.global_action_stats[key]

        key = (score, action)
        if key not in self.global_action_stats:
            self.global_action_stats[key] = [0, 0]  # [failures, successes]

        if success:
            self.global_action_stats[key][1] += 1
        else:
            self.global_action_stats[key][0] += 1

        # Invalidate cache for this key
        if hasattr(self, '_weight_cache') and key in self._weight_cache:
            del self._weight_cache[key]

    def get_global_action_weight(self, score: int, action: int) -> float:
        """Get action weight from global knowledge repository with bounded caching."""
        # Use bounded caching to avoid memory growth
        cache_key = (score, action)
        if not hasattr(self, '_weight_cache'):
            self._weight_cache = {}

        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]

        # Clear cache if it gets too large (prevent unbounded growth)
        if len(self._weight_cache) > 1000:  # Reasonable limit
            self._weight_cache.clear()

        # Calculate weight
        if cache_key not in self.global_action_stats:
            weight = self.RWEIGHT_MIN  # Default exploratory weight
        else:
            failures, successes = self.global_action_stats[cache_key]
            if successes > 0:
                # Use success rate with minimum threshold
                success_rate = successes / (failures + successes)
                weight = max(self.RWEIGHT_MIN, success_rate)
            elif failures > 5:  # Action tried many times and failed
                weight = self.RWEIGHT_MIN * 0.5  # Reduce but don't eliminate
            else:
                weight = self.RWEIGHT_MIN  # Still exploring

        # Cache result (only if cache isn't full)
        if len(self._weight_cache) < 1000:
            self._weight_cache[cache_key] = weight
        return weight

    def _cleanup_for_new_game(self):
        """Clean up caches and temporary data between games to prevent memory growth."""
        # Clear weight cache periodically to prevent unbounded growth
        if hasattr(self, '_weight_cache'):
            self._weight_cache.clear()

        # Clear neural terminal's state cache if it exists
        if hasattr(self.state_graph, '_action_value_cache'):
            self.state_graph._action_value_cache.clear()

    def _send_recon_request(self):
        """Send request through ReCoN graph to coordinate action selection (lightweight)."""
        # Only do ReCoN processing occasionally to avoid overhead
        if self.graph and self.game_counter % 10 == 0:  # Every 10th action
            # Request root to process current state
            self.graph.request_root("recon_arc1_root")

            # Minimal message passing to avoid slowdown
            self.graph.propagate_step()

    def _get_model_action(self) -> int:
        """Get action using neural value model with ReCoN prioritization."""
        if not self.current_state or not self.state_graph.action_model:
            return self._get_rweights_action()

        # Get ReCoN-prioritized actions for efficient search space reduction
        available_actions = list(range(self.current_state.num_actions))
        prioritized_actions = self.state_graph.get_prioritized_actions(available_actions)

        model_values = {}

        # Evaluate only top priority actions for efficiency
        for action_idx, priority_tier in prioritized_actions[:20]:  # Limit neural evaluations
            rweight = self.valid_actions_manager.get_action_weight_for_model(action_idx)
            if rweight == 0:
                continue

            try:
                # Use state graph's prediction method (neural terminal)
                value = self.state_graph.predict_action_value(self.current_state, action_idx)

                # Apply rule-based weighting
                if rweight != 1.0:
                    value *= rweight

                # Apply ReCoN priority boost
                tier_boost = {
                    "high_confirmed": 1.5,
                    "medium_confirmed": 1.2,
                    "untested": 1.0,
                    "last_resort": 0.5
                }.get(priority_tier, 1.0)

                model_values[action_idx] = value * tier_boost

            except Exception as e:
                print(f"Error predicting value for action {action_idx}: {e}")
                continue

        if not model_values:
            return self._get_rweights_action()

        # Return action with highest value
        return max(model_values, key=model_values.get)

    def _get_rweights_action(self) -> int:
        """Get action using ReCoN-enhanced but comprehensive selection (like BlindSquirrel)."""
        if not self.current_state:
            return self._get_random_action()

        # Efficient action collection (limit computation while maintaining exploration)
        actions = []
        weights = []

        # Fast collection of valid actions with cached weights
        for action_idx in range(min(self.current_state.num_actions, 50)):  # Reasonable limit
            # Get rule-based weight (with global knowledge caching)
            rweight = self.valid_actions_manager.get_action_weight_for_model(action_idx)

            if rweight > 0:  # Only consider valid actions
                # Use base weight without expensive value lookups for speed
                actions.append(action_idx)
                weights.append(rweight)

        if not actions:
            return self._get_random_action()

        # Weighted random selection with ReCoN enhancement (like BlindSquirrel)
        try:
            return random.choices(actions, weights=weights, k=1)[0]
        except Exception:
            return random.choice(actions)

    def _get_random_action(self) -> int:
        """Get random action as fallback."""
        if self.current_state and self.current_state.num_actions > 0:
            return random.randint(0, self.current_state.num_actions - 1)
        else:
            return 0

    def _get_default_action(self):
        """Get default action when no state is available."""
        return 0  # ACTION1

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Check if agent is done processing."""
        # Process frame first
        self.process_latest_frame(latest_frame)

        # Check WIN condition
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'WIN':
            return True

        return False

    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """
        Choose action (assumes frame already processed by is_done).
        """
        # Handle special cases
        if hasattr(latest_frame, 'state'):
            if str(latest_frame.state) in ('NOT_PLAYED', 'GAME_OVER'):
                return self._get_default_action()

        # Choose action for active game states
        return self._choose_action(latest_frame)

    def reset(self):
        """Reset agent state."""
        super().reset()
        self.state_graph = None
        self.current_state = None
        self.prev_state = None
        self.prev_action = None
        self.game_counter = 0
        self.level_counter = 0
        self.is_first_frame = True

        # Reset ReCoN components
        if self.valid_actions_node:
            self.valid_actions_node.reset_weights()

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about agent state."""
        info = {
            'agent_type': 'ReCoN ARC-1',
            'game_id': self.game_id,
            'game_counter': self.game_counter,
            'level_counter': self.level_counter,
            'current_state_score': self.current_state.score if self.current_state else None,
            'has_model': self.state_graph.action_model is not None if self.state_graph else False,
            'num_states': len(self.state_graph.states) if self.state_graph else 0,
            'recon_nodes': len(self.graph.nodes) if self.graph else 0
        }

        # Add action weights if available
        if self.valid_actions_manager:
            info['action_weights'] = self.valid_actions_manager.get_all_action_weights()

        return info