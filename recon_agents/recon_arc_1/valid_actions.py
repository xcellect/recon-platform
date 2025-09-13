"""
ReCoN ARC-1 Valid Actions Implementation

Implements action filtering and weighting using ReCoN Hybrid Node.
Faithful reproduction of BlindSquirrel's rule-based action validation.
"""

import random
from typing import Dict, Any, List, Optional
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from recon_engine import ReCoNState
from recon_engine.messages import MessageType


class ValidActionsNode(HybridReCoNNode):
    """
    Hybrid ReCoN node for action validation and weighting.

    Starts in explicit mode with rule-based weights (BlindSquirrel style).
    Can switch to neural mode for learned action validation in the future.
    """

    # Constants from BlindSquirrel
    RWEIGHT_MIN = 0.1
    RWEIGHT_RANK_DISCOUNT = 0.5
    RWEIGHT_NO_DISCOUNT = 0.5

    def __init__(self, node_id: str = "valid_actions"):
        super().__init__(node_id, "script", NodeMode.EXPLICIT)

        # Action validation state
        self.current_state_data = None
        self.state_graph = None
        self.action_weights = {}

    def set_state_graph(self, state_graph):
        """Set reference to state graph for action counter access."""
        self.state_graph = state_graph

    def process_state(self, state_data: Any):
        """Process current state for action validation."""
        self.current_state_data = state_data
        self._calculate_action_weights()

    def _calculate_action_weights(self):
        """Calculate rule-based weights for all actions (BlindSquirrel logic)."""
        if not self.current_state_data:
            return

        self.action_weights = {}

        # Process all actions
        for action_idx in range(self.current_state_data.num_actions):
            # Check if action is disabled
            if self.current_state_data.action_rweights.get(action_idx) == 0:
                self.action_weights[action_idx] = 0.0
                continue

            # Get existing weight or calculate new one
            existing_weight = self.current_state_data.action_rweights.get(action_idx)
            if existing_weight is not None and existing_weight != 0:
                self.action_weights[action_idx] = existing_weight
            else:
                self.action_weights[action_idx] = self._calculate_rweight(action_idx)

    def _calculate_rweight(self, action_idx: int) -> float:
        """Calculate rule-based weight for action with global knowledge fallback."""
        if not self.current_state_data or not self.state_graph:
            return 1.0

        game_id = self.current_state_data.game_id
        score = self.current_state_data.score

        # Get local action statistics
        counter_key = (game_id, score, action_idx)
        failures, successes = self.state_graph.action_counter.get(counter_key, [0, 0])

        if successes > 0:
            # Use local success rate
            weight = max(self.RWEIGHT_MIN, successes / (failures + successes))
        elif failures > 0:
            # Apply local failure discount
            if action_idx < 5:
                base_weight = 1.0
            else:
                base_weight = self.RWEIGHT_RANK_DISCOUNT ** (action_idx - 5)

            weight = max(
                self.RWEIGHT_MIN,
                base_weight * (self.RWEIGHT_NO_DISCOUNT ** failures)
            )
        else:
            # No local data - use fast heuristic with occasional global knowledge check
            if (action_idx < 20 and  # Only check global knowledge for first 20 actions (most important)
                self.state_graph.global_knowledge_provider and
                hasattr(self.state_graph.global_knowledge_provider, 'get_global_action_weight')):
                weight = self.state_graph.global_knowledge_provider.get_global_action_weight(score, action_idx)
            else:
                # Fast default heuristic weights for exploration
                if action_idx < 5:
                    weight = 1.0  # Basic actions get full weight
                else:
                    weight = self.RWEIGHT_RANK_DISCOUNT ** (action_idx - 5)
                weight = max(self.RWEIGHT_MIN, weight)

        return weight

    def get_valid_actions(self) -> List[int]:
        """Get list of valid (non-zero weight) actions."""
        if not self.action_weights:
            return []

        return [action_idx for action_idx, weight in self.action_weights.items() if weight > 0]

    def get_action_weight(self, action_idx: int) -> float:
        """Get weight for specific action."""
        return self.action_weights.get(action_idx, 0.0)

    def sample_weighted_action(self) -> Optional[int]:
        """Sample action using weighted random selection (BlindSquirrel logic)."""
        valid_actions = []
        weights = []

        for action_idx, weight in self.action_weights.items():
            if weight > 0:
                valid_actions.append(action_idx)
                weights.append(weight)

        if not valid_actions:
            return None

        try:
            return random.choices(valid_actions, weights=weights, k=1)[0]
        except Exception:
            return random.choice(valid_actions)

    def should_inhibit_action(self, action_idx: int) -> bool:
        """Check if action should be inhibited (weight == 0)."""
        return self.action_weights.get(action_idx, 0.0) == 0.0

    def process_messages(self):
        """Process incoming ReCoN messages for action validation."""
        # In explicit mode, use rule-based weights
        if self.mode == NodeMode.EXPLICIT:
            return self._process_explicit_mode()
        elif self.mode == NodeMode.NEURAL:
            return self._process_neural_mode()

    def _process_explicit_mode(self):
        """Process messages in explicit mode (rule-based)."""
        # Check for request messages
        for msg in self.incoming_messages.get("sub", []):
            if msg.type == MessageType.REQUEST:
                # Activate and calculate weights
                if self.current_state_data:
                    self._calculate_action_weights()
                    self.state = ReCoNState.ACTIVE

        # Send inhibit messages for invalid actions
        for action_idx, weight in self.action_weights.items():
            if weight == 0.0:
                # Send inhibit message (this would be handled by the graph)
                pass

    def _process_neural_mode(self):
        """Process messages in neural mode (future implementation)."""
        # Future: Neural network-based action validation
        # For now, fallback to explicit mode
        self._process_explicit_mode()

    def get_action_scores(self) -> Dict[int, float]:
        """Get action scores for ranking (used by model-based selection)."""
        return self.action_weights.copy()

    def reset_weights(self):
        """Reset action weights."""
        self.action_weights = {}


class ValidActionsManager:
    """
    Manager class for coordinating action validation in ReCoN graph.

    Bridges between ReCoN message passing and BlindSquirrel action logic.
    """

    def __init__(self, valid_actions_node: ValidActionsNode):
        self.valid_actions_node = valid_actions_node

    def process_state_for_actions(self, state_data: Any, state_graph: Any):
        """Process state data for action validation."""
        self.valid_actions_node.set_state_graph(state_graph)
        self.valid_actions_node.process_state(state_data)

    def filter_actions(self, action_list: List[int]) -> List[int]:
        """Filter action list to only include valid actions."""
        valid_actions = self.valid_actions_node.get_valid_actions()
        return [action for action in action_list if action in valid_actions]

    def get_weighted_action_selection(self) -> Optional[int]:
        """Get action using weighted random selection."""
        return self.valid_actions_node.sample_weighted_action()

    def get_action_weight_for_model(self, action_idx: int) -> float:
        """Get action weight for model-based value adjustment."""
        return self.valid_actions_node.get_action_weight(action_idx)

    def should_inhibit(self, action_idx: int) -> bool:
        """Check if action should be inhibited in ReCoN graph."""
        return self.valid_actions_node.should_inhibit_action(action_idx)

    def get_all_action_weights(self) -> Dict[int, float]:
        """Get all action weights for debugging/analysis."""
        return self.valid_actions_node.get_action_scores()