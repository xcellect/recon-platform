"""
ReCoN ARC-1 State Graph Implementation

Implements state tracking and graph management using ReCoN architecture.
Faithful reproduction of BlindSquirrel's state management with ReCoN message passing.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque
import scipy.ndimage
from recon_agents.base_agent import StateTracker
from recon_engine import ReCoNGraph
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from .models import ReCoNArc1ActionModel, ReCoNArc1Trainer, ActionEncoder

# Import GameAction for action availability checks
GameAction = None
try:
    from agents.structs import GameAction
except ImportError:
    try:
        import sys
        import os
        sys.path.append('/workspace/recon-platform/ARC-AGI-3-Agents')
        from agents.structs import GameAction
    except ImportError:
        GameAction = None


class ReCoNArc1State:
    """
    Represents a game state in ReCoN ARC-1's state graph.

    Faithful reproduction of BlindSquirrel's State class with ReCoN integration.
    """

    def __init__(self, latest_frame: Any):
        self.latest_frame = latest_frame
        self.game_id = getattr(latest_frame, 'game_id', 'default')
        self.score = getattr(latest_frame, 'score', 0)

        # Frame representation (identical to BlindSquirrel)
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'WIN':
            self.frame = 'WIN'
        elif hasattr(latest_frame, 'frame') and latest_frame.frame:
            frame_data = latest_frame.frame[-1] if isinstance(latest_frame.frame, list) else latest_frame.frame
            self.frame = tuple(tuple(inner) for inner in frame_data)
        else:
            self.frame = 'UNKNOWN'

        # State connections (identical to BlindSquirrel)
        self.future_states = {}  # action -> state
        self.prior_states = []   # (state, action) pairs

        # Object analysis and action space (identical to BlindSquirrel)
        self.object_data = []
        if self.frame != 'WIN' and self.frame != 'UNKNOWN':
            self._analyze_objects()

        self.num_actions = len(self.object_data) + 5  # 5 basic + click objects
        self.action_rweights = {i: None for i in range(self.num_actions)}

        # Initialize action availability based on frame (identical to BlindSquirrel)
        self._initialize_action_availability(latest_frame)

    def _analyze_objects(self):
        """Analyze objects in the frame for click action generation. (Identical to BlindSquirrel)"""
        if isinstance(self.frame, str):
            return

        try:
            grid = np.array(self.frame)
            self.object_data = []
            orig_idx = 0

            # Process each color
            for colour in range(16):
                raw_labeled, num_features = scipy.ndimage.label((grid == colour))
                slices = scipy.ndimage.find_objects(raw_labeled)

                for i, slc in enumerate(slices):
                    if slc is None:
                        continue

                    # Extract object properties
                    mask = (raw_labeled[slc] == (i + 1))
                    area = np.sum(mask)
                    h = slc[0].stop - slc[0].start
                    w = slc[1].stop - slc[1].start
                    bbox_area = h * w
                    size = h * w / (64 * 64)
                    regularity = area / bbox_area if bbox_area > 0 else 0

                    # Calculate centroid
                    ys, xs = np.nonzero(mask)
                    y_centroid = ys.mean() + slc[0].start if len(ys) > 0 else 0
                    x_centroid = xs.mean() + slc[1].start if len(xs) > 0 else 0

                    self.object_data.append({
                        "orig_idx": orig_idx,
                        "colour": colour,
                        "slice": slc,
                        "mask": mask,
                        "area": area,
                        "bbox_area": bbox_area,
                        "size": size,
                        "regularity": regularity,
                        "y_centroid": y_centroid,
                        "x_centroid": x_centroid
                    })
                    orig_idx += 1

            # Sort objects by importance (regularity, area, colour, orig_idx)
            self.object_data.sort(
                key=lambda obj: (-obj["regularity"], -obj["area"], -obj["colour"], obj["orig_idx"])
            )

        except Exception as e:
            print(f"Error analyzing objects: {e}")
            self.object_data = []

    def _initialize_action_availability(self, latest_frame: Any):
        """Initialize which actions are available. (Identical to BlindSquirrel)"""
        if not hasattr(latest_frame, 'available_actions'):
            return

        available_actions = latest_frame.available_actions

        # Use GameAction enum directly like original implementation
        if GameAction:
            # Disable unavailable basic actions
            if GameAction.ACTION1 not in available_actions:
                self.action_rweights[0] = 0
            if GameAction.ACTION2 not in available_actions:
                self.action_rweights[1] = 0
            if GameAction.ACTION3 not in available_actions:
                self.action_rweights[2] = 0
            if GameAction.ACTION4 not in available_actions:
                self.action_rweights[3] = 0
            if GameAction.ACTION5 not in available_actions:
                self.action_rweights[4] = 0

            # Disable click actions if ACTION6 not available
            if GameAction.ACTION6 not in available_actions:
                for i in range(5, self.num_actions):
                    self.action_rweights[i] = 0

    def get_action_obj(self, action_idx: int) -> Any:
        """Convert action index to action object. (Identical to BlindSquirrel)"""
        if action_idx < 5:
            # Basic actions
            if action_idx == 0:
                return {"type": "basic", "action_id": 0}
            elif action_idx == 1:
                return {"type": "basic", "action_id": 1}
            elif action_idx == 2:
                return {"type": "basic", "action_id": 2}
            elif action_idx == 3:
                return {"type": "basic", "action_id": 3}
            elif action_idx == 4:
                return {"type": "basic", "action_id": 4}
        else:
            # Click actions
            object_idx = action_idx - 5
            if object_idx < len(self.object_data):
                obj = self.object_data[object_idx]
                return {
                    "type": "click",
                    "x": int(obj["x_centroid"]),
                    "y": int(obj["y_centroid"])
                }

        # Fallback
        return {"type": "basic", "action_id": 0}

    def __hash__(self):
        """Hash based on frame and score."""
        return hash((self.frame, self.score))

    def __eq__(self, other):
        """Equality based on frame and score."""
        if not isinstance(other, ReCoNArc1State):
            return False
        return self.frame == other.frame and self.score == other.score


class ReCoNArc1StateGraph(StateTracker):
    """
    State graph implementation using ReCoN architecture.

    Uses ReCoN for efficient search space reduction via hierarchical hypothesis testing.
    """

    def __init__(self, global_knowledge_provider=None):
        super().__init__()

        # Core state tracking (identical to BlindSquirrel)
        self.states = {}  # state_hash -> state
        self.action_counter = {}  # (game_id, score, action) -> [failures, successes]

        # Global knowledge provider (shared across games)
        self.global_knowledge_provider = global_knowledge_provider

        # Neural model components
        self.action_model = None
        self.trainer = None
        self.action_encoder = ActionEncoder()

        # Simplified for performance - removed unused state categorization
        # Only keep essential data structures

    def _build_hypothesis_architecture(self):
        """Simplified ReCoN architecture for performance."""
        # Minimal ReCoN structure to avoid overhead
        pass

    def _categorize_state(self, prev_state: ReCoNArc1State, current_state: ReCoNArc1State) -> str:
        """Simple state categorization for success/failure tracking."""
        if current_state.frame == 'WIN':
            return "winning"
        elif current_state.score > prev_state.score:
            return "progress"
        else:
            return "neutral"

    def get_state(self, latest_frame: Any) -> ReCoNArc1State:
        """Get or create state for frame. (Identical to BlindSquirrel logic)"""
        state = ReCoNArc1State(latest_frame)
        state_hash = hash(state)

        if state_hash not in self.states:
            self.states[state_hash] = state

        return self.states[state_hash]

    def add_init_state(self, state: ReCoNArc1State):
        """Add initial state (simplified for performance)."""
        # Just track the state, no categorization needed
        pass

    def update(self, prev_state: ReCoNArc1State, action: int, current_state: ReCoNArc1State):
        """Update state graph using efficient ReCoN categorization."""
        # Store transition (unchanged from BlindSquirrel)
        prev_state.future_states[action] = current_state
        current_state.prior_states.append((prev_state, action))

        # Update action counters (unchanged from BlindSquirrel)
        game_id = prev_state.game_id
        score = prev_state.score
        counter_key = (game_id, score, action)

        if counter_key not in self.action_counter:
            self.action_counter[counter_key] = [0, 0]  # [failures, successes]

        # Categorize state transition for efficient search space reduction
        transition_category = self._categorize_state(prev_state, current_state)

        # Simplified update for performance
        success = (transition_category == "progress" or transition_category == "winning")

        if success:
            self.action_counter[counter_key][1] += 1  # Success
        else:
            self.action_counter[counter_key][0] += 1  # Failure

        # Update global knowledge only (skip expensive local categorization for speed)
        if self.global_knowledge_provider:
            self.global_knowledge_provider._update_global_knowledge(score, action, success)

        # Trigger model training if needed
        self._check_training_trigger(current_state)

    def _update_action_hypothesis(self, action: int, outcome: str, category: str):
        """Update action hypothesis tracking based on outcome."""
        # Estimate action value based on historical performance
        action_value = self._estimate_action_value(action)
        value_category = self._categorize_action_value(action_value)

        if outcome == "confirm":
            self.action_hypotheses[value_category]["confirmed"].add(action)
            self.action_hypotheses[value_category]["testing"].discard(action)
        elif outcome == "fail":
            self.action_hypotheses[value_category]["failed"].add(action)
            self.action_hypotheses[value_category]["testing"].discard(action)

    def _update_hypothesis_states(self, transition_category: str, action: int):
        """Update ReCoN hypothesis node states based on transition outcome."""
        action_value = self._estimate_action_value(action)
        value_category = self._categorize_action_value(action_value)

        hypothesis_node = f"{value_category}_actions"
        if hypothesis_node in self.recon_graph.nodes:
            node = self.recon_graph.nodes[hypothesis_node]

            if transition_category in ["progress", "winning"]:
                # Successful transition - confirm hypothesis
                from recon_engine.node import ReCoNState
                node.state = ReCoNState.CONFIRMED
            elif transition_category == "regress":
                # Failed transition - mark hypothesis as failed
                from recon_engine.node import ReCoNState
                node.state = ReCoNState.FAILED

    def _estimate_action_value(self, action: int) -> float:
        """Estimate action value from historical data."""
        if not hasattr(self, '_action_value_cache'):
            self._action_value_cache = {}

        if action in self._action_value_cache:
            return self._action_value_cache[action]

        # Calculate success rate across all games/scores for this action
        total_successes = 0
        total_attempts = 0

        for (game_id, score, act_idx), (failures, successes) in self.action_counter.items():
            if act_idx == action:
                total_successes += successes
                total_attempts += successes + failures

        if total_attempts == 0:
            value = 0.5  # Neutral for unknown actions
        else:
            value = total_successes / total_attempts

        self._action_value_cache[action] = value
        return value


    def _check_training_trigger(self, current_state: ReCoNArc1State):
        """Check if we should train the model. (Adapted from BlindSquirrel)"""
        # Train at end of level 1, then every other level
        if current_state.score == 1 or (current_state.score > 1 and current_state.score % 2 == 0):
            self._train_model()

    def _train_model(self):
        """Train the action value model. (Adapted from BlindSquirrel)"""
        if not self.trainer:
            self.trainer = ReCoNArc1Trainer()

        # Collect training data from state graph
        training_data = self._collect_training_data()

        if training_data and len(training_data) > 10:  # Minimum data requirement
            self.action_model = self.trainer.train(training_data, self.action_encoder)

    def _collect_training_data(self) -> List[Tuple]:
        """Collect training data from state transitions."""
        training_data = []

        for state in self.states.values():
            if state.frame == 'WIN' or state.frame == 'UNKNOWN':
                continue

            # Calculate distances to winning state for each action
            for action_idx in range(state.num_actions):
                if action_idx in state.future_states:
                    next_state = state.future_states[action_idx]
                    distance = self._calculate_distance_to_win(next_state)

                    if distance is not None:
                        training_data.append((state, action_idx, 1.0 / (1.0 + distance)))

        return training_data

    def _calculate_distance_to_win(self, state: ReCoNArc1State) -> Optional[int]:
        """Calculate shortest distance to winning state using BFS."""
        if state.frame == 'WIN':
            return 0

        visited = set()
        queue = deque([(state, 0)])

        while queue:
            current_state, distance = queue.popleft()

            if hash(current_state) in visited:
                continue
            visited.add(hash(current_state))

            if current_state.frame == 'WIN':
                return distance

            # Check all future states
            for next_state in current_state.future_states.values():
                if hash(next_state) not in visited and distance < 50:  # Limit search depth
                    queue.append((next_state, distance + 1))

        return None  # No path to win found

    def predict_action_value(self, state: ReCoNArc1State, action_idx: int) -> float:
        """Predict value of action for given state. (Identical to BlindSquirrel interface)"""
        if not self.action_model:
            return 0.5  # Default neutral value

        try:
            # Convert state to tensor
            if isinstance(state.frame, str):
                return 0.5

            grid = torch.tensor(state.frame, dtype=torch.float32).unsqueeze(0)  # Add batch dim

            # Get action encoding
            action_encoding = self.action_encoder.encode_action(state, action_idx)
            action_tensor = torch.tensor(action_encoding, dtype=torch.float32).unsqueeze(0)

            # Predict value
            with torch.no_grad():
                value = self.action_model(grid, action_tensor)
                return float(value.item())

        except Exception as e:
            print(f"Error predicting action value: {e}")
            return 0.5