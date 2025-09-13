"""
BlindSquirrel State Graph Implementation

Implements state tracking and graph management using ReCoN architecture.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque
import scipy.ndimage
from recon_agents.base_agent import StateTracker
from recon_engine import ReCoNGraph
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from .models import BlindSquirrelActionModel, BlindSquirrelTrainer, ActionEncoder

# Import GameAction for action availability checks
# Try multiple import paths to handle different contexts
GameAction = None
try:
    from agents.structs import GameAction
except ImportError:
    try:
        # Try absolute path if relative doesn't work
        import sys
        import os
        sys.path.append('/workspace/recon-platform/ARC-AGI-3-Agents')
        from agents.structs import GameAction
    except ImportError:
        # Will fallback to string comparison
        GameAction = None


class BlindSquirrelState:
    """
    Represents a game state in BlindSquirrel's state graph.

    Equivalent to original State class but integrates with ReCoN.
    """

    def __init__(self, latest_frame: Any):
        self.latest_frame = latest_frame
        self.game_id = getattr(latest_frame, 'game_id', 'default')
        self.score = getattr(latest_frame, 'score', 0)

        # Frame representation
        if hasattr(latest_frame, 'state') and str(latest_frame.state) == 'WIN':
            self.frame = 'WIN'
        elif hasattr(latest_frame, 'frame') and latest_frame.frame:
            # Convert frame to tuple for hashing
            frame_data = latest_frame.frame[-1] if isinstance(latest_frame.frame, list) else latest_frame.frame
            self.frame = tuple(tuple(inner) for inner in frame_data)
        else:
            self.frame = 'UNKNOWN'

        # State connections
        self.future_states = {}  # action -> state
        self.prior_states = []   # (state, action) pairs

        # Object analysis and action space
        self.object_data = []
        if self.frame != 'WIN' and self.frame != 'UNKNOWN':
            self._analyze_objects()

        self.num_actions = len(self.object_data) + 5  # 5 basic + click objects
        self.action_rweights = {i: None for i in range(self.num_actions)}

        # Initialize action availability based on frame
        self._initialize_action_availability(latest_frame)

    def _analyze_objects(self):
        """Analyze objects in the frame for click action generation."""
        if isinstance(self.frame, str):
            return

        try:
            grid = np.array(self.frame)
            self.object_data = []
            orig_idx = 0

            # Process each color
            for colour in range(16):
                # Find connected components of this color
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
        """Initialize which actions are available."""
        # Check available actions from frame
        if not hasattr(latest_frame, 'available_actions'):
            return

        available_actions = latest_frame.available_actions

        # Use GameAction enum directly like original implementation
        if GameAction:
            # Disable unavailable basic actions (using enum comparison like original)
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
                for i in range(len(self.object_data)):
                    self.action_rweights[i + 5] = 0
        else:
            # Fallback to string comparison if GameAction not available
            action_map = {
                'ACTION1': 0, 'ACTION2': 1, 'ACTION3': 2, 'ACTION4': 3, 'ACTION5': 4
            }

            for action_name, action_idx in action_map.items():
                action_available = any(
                    str(action) == action_name for action in available_actions
                )
                if not action_available:
                    self.action_rweights[action_idx] = 0

            # Disable click actions if ACTION6 not available
            action6_available = any(
                str(action) == 'ACTION6' for action in available_actions
            )
            if not action6_available:
                for i in range(len(self.object_data)):
                    self.action_rweights[i + 5] = 0

    def get_action_tensor(self, action: int) -> torch.Tensor:
        """Get action encoding tensor for the given action index."""
        return ActionEncoder.encode_action(action, self.object_data)

    def get_action_obj(self, action: int) -> Any:
        """Convert action index to action object for the harness."""
        # This will be handled by the harness adapter
        # Return action data that the adapter can convert
        if action <= 4:
            return {
                'type': 'basic',
                'action_id': action,
                'action_name': f'ACTION{action + 1}'
            }
        else:
            return self._get_click_action_obj(action)

    def _get_click_action_obj(self, action: int):
        """Generate click action object for object-based action."""

        if action - 5 >= len(self.object_data):
            # Fallback click action - return data for harness conversion
            return {
                'type': 'click',
                'x': 32,
                'y': 32
            }

        obj = self.object_data[action - 5]
        slc = obj["slice"]
        mask = obj["mask"]

        # Choose random point within the object
        local_coords = np.argwhere(mask)
        if len(local_coords) > 0:
            idx = np.random.choice(len(local_coords))
            local_y, local_x = local_coords[idx]
            global_y = slc[0].start + local_y
            global_x = slc[1].start + local_x
        else:
            # Fallback to object centroid
            global_y = int(obj["y_centroid"])
            global_x = int(obj["x_centroid"])

        return {
            'type': 'click',
            'x': global_x,
            'y': global_y
        }

    def zero_back(self):
        """Propagate failure backwards through the state graph."""
        if all(v == 0 for v in self.action_rweights.values()):
            for state, action in self.prior_states:
                if state.action_rweights[action] == 1:
                    state.action_rweights[action] = 0
                    state.zero_back()

    def __eq__(self, other):
        """Equality based on game_id, score, and frame."""
        if not isinstance(other, BlindSquirrelState):
            return NotImplemented
        return (self.game_id, self.score, self.frame) == (other.game_id, other.score, other.frame)

    def __hash__(self):
        """Hash for use in sets and dictionaries."""
        return hash((self.game_id, self.score, self.frame))


class BlindSquirrelStateGraph:
    """
    State graph implementation using ReCoN architecture.

    Manages state transitions, milestone tracking, and model training.
    """

    # Constants from original implementation
    EPSILON = 0.5  # Exploration parameter for training condition

    def __init__(self):
        self.init_state = None
        self.milestones = {}  # (game_id, score) -> state
        self.states = set()
        self.action_counter = {}  # (game_id, score, action) -> [failures, successes]
        self.game_id = None

        # Neural model for value prediction
        self.action_model = None
        self.trainer = None

        # ReCoN integration
        self.recon_graph = ReCoNGraph()
        self._setup_recon_nodes()

    def _setup_recon_nodes(self):
        """Setup ReCoN nodes for state graph management."""
        # State validator node
        state_validator = HybridReCoNNode("state_validator", "script", NodeMode.EXPLICIT)
        self.recon_graph.add_node(state_validator)

        # Transition tracker node
        transition_tracker = HybridReCoNNode("transition_tracker", "script", NodeMode.EXPLICIT)
        self.recon_graph.add_node(transition_tracker)

        # Connect nodes
        self.recon_graph.add_link("state_validator", "transition_tracker", "por")

    def get_state(self, latest_frame: Any) -> BlindSquirrelState:
        """Get or create state for the given frame."""
        new_state = BlindSquirrelState(latest_frame)

        # Check if state already exists
        existing_state = next((s for s in self.states if s == new_state), None)
        if existing_state:
            return existing_state

        # Add new state
        self.states.add(new_state)
        return new_state

    def add_init_state(self, state: BlindSquirrelState):
        """Set the initial state and add as milestone."""
        self.init_state = state
        self.add_milestone(state)
        self.game_id = state.game_id

    def add_milestone(self, state: BlindSquirrelState):
        """Add a milestone state (level completion)."""
        key = (state.game_id, state.score)
        if key in self.milestones:
            assert self.milestones[key] == state
        else:
            self.milestones[key] = state

    def update(self, prev_state: BlindSquirrelState, action: int, new_state: BlindSquirrelState):
        """Update state graph with new transition."""
        game_id = prev_state.game_id
        score = prev_state.score

        # Check for Markov violations
        if action in prev_state.future_states:
            if prev_state.future_states[action] != new_state:
                print(f'Warning: Markov Violation in {game_id}')

        # Record transition
        assert prev_state in self.states
        prev_state.future_states[action] = new_state
        new_state.prior_states.append((prev_state, action))

        # Update action counter
        counter_key = (game_id, score, action)
        if counter_key not in self.action_counter:
            self.action_counter[counter_key] = [0, 0]  # [failures, successes]

        # Process outcome
        if new_state == prev_state:
            # No progress - bad action
            self.action_counter[counter_key][0] += 1
            prev_state.action_rweights[action] = 0
            prev_state.zero_back()
            print(f'Warning: Bad Action in {game_id}')

        elif new_state == self.milestones.get((game_id, score)):
            # Returned to milestone - failure
            self.action_counter[counter_key][1] += 1
            prev_state.action_rweights[action] = 0
            prev_state.zero_back()

        elif new_state.score > prev_state.score:
            # Progress made - good action
            self.action_counter[counter_key][1] += 1
            prev_state.action_rweights[action] = 1
            self.add_milestone(new_state)

            # Train model if epsilon condition is met (matches original)
            if self.EPSILON < 1:
                self.train_model(game_id, score + 1)

        else:
            # Same level, different state - neutral
            self.action_counter[counter_key][1] += 1
            prev_state.action_rweights[action] = 1

    def get_level_training_data(self, old_milestone: BlindSquirrelState,
                              new_milestone: BlindSquirrelState) -> List[Dict]:
        """Generate training data for model between two milestones."""
        if new_milestone.frame == 'WIN':
            final_frame = new_milestone.frame
        else:
            final_frame = tuple(tuple(inner) for inner in new_milestone.latest_frame.frame[0])

        # Calculate distances from new milestone
        state_data = {new_milestone: {'distance': 0, 'frame': final_frame}}
        max_distance = 0
        queue = deque([new_milestone])

        # Backward search from new milestone
        while queue:
            state = queue.popleft()
            current_distance = state_data[state]['distance']

            for prev_state, action in state.prior_states:
                if prev_state.score != old_milestone.score:
                    continue

                if prev_state not in state_data:
                    distance = current_distance + 1
                    state_data[prev_state] = {
                        'frame': prev_state.frame,
                        'distance': distance
                    }
                    max_distance = max(max_distance, distance)
                    queue.append(prev_state)

        # Forward search from old milestone
        queue = deque([old_milestone])
        while queue:
            state = queue.popleft()
            current_distance = state_data.get(state, {}).get('distance', max_distance)

            for action, future_state in state.future_states.items():
                if future_state.score != old_milestone.score:
                    continue

                if future_state not in state_data:
                    distance = current_distance + 1
                    state_data[future_state] = {
                        'frame': future_state.frame,
                        'distance': distance
                    }
                    max_distance = max(max_distance, distance)
                    queue.append(future_state)

        # Generate training examples
        training_data = []
        score_magnitude = 1

        for state in self.states:
            if state.score != old_milestone.score:
                continue

            for action, future_state in state.future_states.items():
                if future_state not in state_data or state not in state_data:
                    continue

                # Prepare tensors
                state_tensor = torch.tensor(state_data[state]['frame'], dtype=torch.long)
                action_tensor = state.get_action_tensor(action)

                # Calculate score based on distance improvement
                if state.action_rweights[action] == 0:
                    score = torch.tensor(-score_magnitude, dtype=torch.float32).unsqueeze(0)
                else:
                    state_distance = state_data[state]['distance']
                    future_distance = state_data[future_state]['distance']
                    if max_distance > 0:
                        score_val = score_magnitude * (state_distance - future_distance) / max_distance
                    else:
                        score_val = score_magnitude
                    score = torch.tensor(score_val, dtype=torch.float32).unsqueeze(0)

                training_data.append({
                    'state': state_tensor,
                    'action': action_tensor,
                    'score': score
                })

        return training_data

    def train_model(self, game_id: str, max_score: int, verbose: bool = True):
        """Train the action value model."""
        # Always create fresh model like original implementation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_model = BlindSquirrelActionModel(game_id).to(device)
        self.trainer = BlindSquirrelTrainer(self.action_model)

        # Collect training data from all levels
        training_data = []
        for score in range(max_score):
            old_milestone = self.milestones.get((game_id, score))
            new_milestone = self.milestones.get((game_id, score + 1))

            if old_milestone and new_milestone:
                level_data = self.get_level_training_data(old_milestone, new_milestone)
                training_data.extend(level_data)

        if training_data:
            if verbose:
                print(f"Training model with {len(training_data)} examples")
            self.trainer.train_model(training_data, verbose=verbose)
        else:
            if verbose:
                print("No training data available")

    def predict_action_value(self, state: BlindSquirrelState, action: int) -> float:
        """Predict value for state-action pair."""
        if self.trainer is None:
            return 0.0

        state_tensor = torch.tensor(state.frame, dtype=torch.long) if state.frame != 'WIN' else torch.zeros(64, 64, dtype=torch.long)
        action_tensor = state.get_action_tensor(action)

        return self.trainer.predict_value(state_tensor, action_tensor)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the state graph."""
        return {
            'total_states': len(self.states),
            'milestones': len(self.milestones),
            'action_counters': len(self.action_counter),
            'has_model': self.action_model is not None,
            'game_id': self.game_id
        }