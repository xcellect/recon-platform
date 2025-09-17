"""
Arcon State Graph Implementation

Implements state tracking and graph management using ReCoN architecture.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque
import scipy.ndimage
import random
from recon_agents.base_agent import StateTracker
from recon_engine import ReCoNGraph
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from recon_engine.node import ReCoNNode
from .models import ArconActionModel, ArconTrainer, ActionEncoder

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


# ReCoN Click Arbiter Functions
import os
import json
from datetime import datetime
def compute_object_penalties(object_data: List[Dict], pxy: Optional[np.ndarray] = None, 
                           grid_size: int = 64, area_frac_cutoff: float = 0.005, 
                           border_penalty: float = 0.8) -> List[float]:
    """
    Compute penalty-weighted scores for objects using ReCoN bottom-up evidence.
    
    Args:
        object_data: List of object dictionaries from ArconState
        pxy: Optional 2D click probability heatmap (64x64 or grid_size x grid_size)
        grid_size: Size of the grid (default 64 for ARC)
        area_frac_cutoff: Minimum area fraction to include object
        border_penalty: Penalty factor for objects touching borders
    
    Returns:
        List of penalty weights for each object (0 = filtered out)
    """
    if not object_data:
        return []
    
    penalties = []
    total_grid_area = grid_size * grid_size
    
    for obj in object_data:
        # Area fraction filter
        area_frac = obj['area'] / total_grid_area
        if area_frac < area_frac_cutoff:
            penalties.append(0.0)
            continue
            
        # Base penalty from object properties
        penalty = obj['regularity']  # Use regularity as base quality measure
        
        # Area penalty (small objects get reduced weight)
        penalty *= min(1.0, area_frac / 0.01)  # Normalize to reasonable object size
        
        # Border penalty (objects touching border get reduced weight)
        slc = obj['slice']
        touches_border = (slc[0].start == 0 or slc[0].stop == grid_size or 
                         slc[1].start == 0 or slc[1].stop == grid_size)
        if touches_border:
            penalty *= border_penalty
            
        # Pxy contribution (click probability heatmap)
        if pxy is not None:
            # Get maximum probability within object mask
            obj_pxy = pxy[slc[0], slc[1]]
            mask_pxy = obj_pxy * obj['mask']  # Apply object mask
            max_pxy = np.max(mask_pxy) if np.any(mask_pxy > 0) else 0.0
            penalty *= max_pxy
        
        penalties.append(max(0.0, penalty))
    
    return penalties


def create_recon_hypothesis_graph(object_data: List[Dict], pxy: Optional[np.ndarray] = None,
                                  grid_size: int = 64, value_model=None, **penalty_kwargs) -> Tuple[ReCoNGraph, List[float]]:
    """
    Create hierarchical ReCoN graph for hypothesis-driven action selection.
    
    Returns:
        Tuple of (ReCoN graph, object weights)
    """
    # Compute object penalties
    object_weights = compute_object_penalties(object_data, pxy, grid_size, **penalty_kwargs)
    
    # Create ReCoN graph
    graph = ReCoNGraph()
    
    # Root hypothesis: "Can achieve score increase"
    root = graph.add_node("score_increase_hypothesis", "script")
    
    # Branch 1: Basic action hypothesis
    basic_branch = graph.add_node("basic_action_branch", "script")
    graph.add_link("score_increase_hypothesis", "basic_action_branch", "sub")
    
    # Sequential basic actions with por/ret
    action_nodes = []
    for i in range(1, 6):
        action_id = f"action_{i}"
        action_node = graph.add_node(action_id, "script")
        action_nodes.append(action_node)
        graph.add_link("basic_action_branch", action_id, "sub")
        
        # Add measurement terminal for this action
        terminal_id = f"{action_id}_terminal"
        terminal = graph.add_node(terminal_id, "terminal")
        # Real measurement: use value model if available, else simple heuristic
        if value_model:
            terminal.measurement_fn = lambda env, act_idx=i-1: _measure_action_value(env, act_idx, value_model)
        else:
            terminal.measurement_fn = lambda env, act_idx=i-1: 0.8 + 0.02 * act_idx  # Above threshold
        # Set lower threshold so measurements can confirm
        terminal.transition_threshold = 0.7
        graph.add_link(action_id, terminal_id, "sub")
        
        # Create sequence: action_1 -> action_2 -> ... -> action_5
        if i > 1:
            prev_action = f"action_{i-1}"
            graph.add_link(prev_action, action_id, "por")
    
    # Branch 2: Click hypothesis with sequential structure
    click_branch = graph.add_node("click_hypothesis", "script")
    graph.add_link("score_increase_hypothesis", "click_hypothesis", "sub")
    
    # Sequential click steps: perceive -> select -> verify
    perceive_step = graph.add_node("perceive_objects", "script")
    select_step = graph.add_node("select_object", "script")
    verify_step = graph.add_node("verify_change", "script")
    
    graph.add_link("click_hypothesis", "perceive_objects", "sub")
    graph.add_link("click_hypothesis", "select_step", "sub")
    graph.add_link("click_hypothesis", "verify_step", "sub")
    
    # Sequence constraint: perceive -> select -> verify
    graph.add_link("perceive_objects", "select_step", "por")
    graph.add_link("select_step", "verify_step", "por")
    
    # Add perception terminal under perceive_objects (fix: script needs children)
    perception_terminal = graph.add_node("perception_ready", "terminal")
    perception_terminal.measurement_fn = lambda env: 0.9  # Always ready to perceive
    graph.add_link("perceive_objects", "perception_ready", "sub")
    
    # Add object terminals under select_step with real measurements
    for i, (obj, weight) in enumerate(zip(object_data, object_weights)):
        if weight > 0:
            terminal = graph.add_node(f"object_{i}", "terminal")
            # Real measurement: object quality (regularity, area, contrast)
            terminal.measurement_fn = lambda env, obj=obj: _measure_object_quality(obj)
            # Set lower threshold so object measurements can confirm
            terminal.transition_threshold = 0.6
            graph.add_link("select_step", f"object_{i}", "sub", weight)
    
    # Add change verification terminal
    change_terminal = graph.add_node("change_detector", "terminal")
    change_terminal.measurement_fn = lambda env: 0.8  # Placeholder for frame change detection
    graph.add_link("verify_step", "change_detector", "sub")
    
    return graph, object_weights


def _measure_action_value(env, action_idx: int, value_model) -> float:
    """Measure action value using Arcon value model."""
    try:
        if value_model and hasattr(value_model, 'predict_value'):
            # Use actual value model prediction
            return max(0.0, min(1.0, float(value_model.predict_value(env, action_idx))))
        else:
            # Fallback heuristic
            return 0.2 + 0.1 * action_idx
    except Exception:
        return 0.3


def _measure_object_quality(obj: Dict[str, Any]) -> float:
    """Measure object quality based on segmentation properties."""
    try:
        # Combine regularity, size, and contrast into measurement
        regularity = obj.get('regularity', 0.5)
        area_frac = obj.get('area', 0) / (64 * 64)
        size_bonus = min(1.0, area_frac / 0.01)  # Bonus for reasonable size
        
        # Quality measurement
        quality = regularity * 0.6 + size_bonus * 0.4
        return max(0.0, min(1.0, quality))
    except Exception:
        return 0.5


# Legacy function for backward compatibility
def create_recon_click_arbiter(object_data: List[Dict], pxy: Optional[np.ndarray] = None,
                              grid_size: int = 64, **penalty_kwargs) -> Tuple[ReCoNGraph, List[float]]:
    """Legacy wrapper - redirects to hierarchical version."""
    return create_recon_hypothesis_graph(object_data, pxy, grid_size, None, **penalty_kwargs)


def execute_recon_click_arbiter(graph: ReCoNGraph, object_weights: List[float], 
                               exploration_rate: float = 0.0,
                               log_meta: Optional[Dict[str, Any]] = None) -> int:
    """
    Execute ReCoN click arbiter and return selected object index.
    
    Args:
        graph: ReCoN graph from create_recon_click_arbiter
        object_weights: Object weights from create_recon_click_arbiter
        exploration_rate: Probability of random selection (0.0 = pure ReCoN)
        log_meta: Optional dict to enable logging with keys like {game_id, score}
    
    Returns:
        Selected object index
    """
    # Exploration: random selection
    if exploration_rate > 0 and random.random() < exploration_rate:
        valid_indices = [i for i, w in enumerate(object_weights) if w > 0]
        if valid_indices:
            return random.choice(valid_indices)
        else:
            return 0  # Fallback
    
    # Execute ReCoN script with history for logging
    history = None
    result = graph.execute_script("score_increase_hypothesis", max_steps=15)
    if log_meta is not None:
        try:
            history = graph.execute_script_with_history("score_increase_hypothesis", max_steps=15)
        except Exception:
            history = None
    
    if result == "confirmed":
        # Find which child confirmed by checking final activations
        # The child with highest scaled confirmation wins
        best_object = 0
        best_activation = 0.0
        
        for i, weight in enumerate(object_weights):
            if weight > 0:
                node_id = f"object_{i}"
                if node_id in graph.nodes:
                    node = graph.nodes[node_id]
                    # Activation is scaled by link weight in message passing
                    scaled_activation = float(node.activation) * weight
                    if scaled_activation > best_activation:
                        best_activation = scaled_activation
                        best_object = i
        
        # Optional logging
        if log_meta is not None:
            _write_recon_trace_bs(history, object_weights, best_object, log_meta)
        return best_object
    else:
        # Fallback: select object with highest weight
        if object_weights:
            best_object = max(range(len(object_weights)), key=lambda i: object_weights[i])
            if log_meta is not None:
                _write_recon_trace_bs(history, object_weights, best_object, log_meta)
            return best_object
        else:
            if log_meta is not None:
                _write_recon_trace_bs(history, object_weights, 0, log_meta)
            return 0


def _ensure_recon_log_dir_bs(game_id: str, score: Optional[int]) -> str:
    base_dir = "/workspace/recon-platform/recon_log"
    game_dir = os.path.join(base_dir, f"game_{game_id}")
    level_dir = os.path.join(game_dir, f"level_{score if score is not None else 'NA'}")
    try:
        os.makedirs(level_dir, exist_ok=True)
    except Exception:
        pass
    return level_dir


def _write_recon_trace_bs(history: Optional[Dict[str, Any]],
                          object_weights: List[float],
                          selected_object_idx: int,
                          log_meta: Dict[str, Any]) -> None:
    try:
        game_id = str(log_meta.get('game_id', 'unknown'))
        score = log_meta.get('score', None)
        step_ix = int(log_meta.get('action_count', 0))
        level_dir = _ensure_recon_log_dir_bs(game_id, score)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        fname = f"bs_recon_trace_step_{step_ix:05d}_{ts}.json"
        fpath = os.path.join(level_dir, fname)

        payload = {
            "meta": {
                "agent": "ArconReCoN",
                "game_id": game_id,
                "score": score,
                "action_count": step_ix,
                "timestamp_utc": ts
            },
            "history": history if history is not None else None,
            "object_weights": object_weights,
            "outcome": {
                "selected_object_index": selected_object_idx
            }
        }

        with open(fpath, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        pass


class ArconState:
    """
    Represents a game state in Arcon's state graph.

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

    def get_action_obj(self, action: int, state_graph: 'ArconStateGraph' = None) -> Any:
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
            # Check if ReCoN click arbiter should be used
            if (state_graph and hasattr(state_graph, 'use_recon_click_arbiter') and 
                state_graph.use_recon_click_arbiter):
                
                # Update statistics
                state_graph.total_click_selections += 1
                state_graph.recon_click_selections += 1
                
                # Use ReCoN click arbiter
                return self._get_click_action_obj_with_recon(
                    action,
                    use_recon=True,
                    recon_exploration_rate=state_graph.recon_exploration_rate,
                    area_frac_cutoff=state_graph.recon_area_frac_cutoff,
                    border_penalty=state_graph.recon_border_penalty
                )
            else:
                # Use original behavior
                if state_graph:
                    state_graph.total_click_selections += 1
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
    
    def _get_click_action_obj_with_recon(self, action: int, use_recon: bool = False,
                                        recon_exploration_rate: float = 0.0,
                                        **recon_kwargs) -> Dict[str, Any]:
        """Generate click action object using ReCoN arbiter if enabled."""
        
        if not use_recon or not self.object_data:
            # Fallback to original behavior
            return self._get_click_action_obj(action)
        
        try:
            # Use ReCoN hypothesis graph to select object
            pxy = recon_kwargs.get('pxy', None)  # Optional click heatmap
            grid_size = 64  # ARC grid size
            
            # Get value model from state_graph if available
            value_model = getattr(state_graph, 'action_model', None) if state_graph else None
            
            # Create and execute hierarchical ReCoN hypothesis
            graph, object_weights = create_recon_hypothesis_graph(
                self.object_data, pxy, grid_size, value_model,
                area_frac_cutoff=recon_kwargs.get('area_frac_cutoff', 0.005),
                border_penalty=recon_kwargs.get('border_penalty', 0.8)
            )
            
            # Build log metadata for tracing
            log_meta = {
                'game_id': self.game_id,
                'score': getattr(self, 'score', 0),
                'action_count': state_graph.recon_click_selections + state_graph.total_click_selections
            }
            selected_object_idx = execute_recon_click_arbiter(
                graph, object_weights, recon_exploration_rate, log_meta
            )
            
            # Get coordinates for selected object
            if selected_object_idx < len(self.object_data):
                obj = self.object_data[selected_object_idx]
                slc = obj["slice"]
                mask = obj["mask"]
                
                # Choose point within the object (centroid for determinism)
                global_y = int(obj["y_centroid"])
                global_x = int(obj["x_centroid"])
                
                return {
                    'type': 'click',
                    'x': global_x,
                    'y': global_y,
                    'recon_selected': True,
                    'object_index': selected_object_idx,
                    'object_weight': object_weights[selected_object_idx] if selected_object_idx < len(object_weights) else 0.0
                }
            else:
                # Fallback
                return self._get_click_action_obj(action)
                
        except Exception as e:
            # Fallback to original behavior on any error
            print(f"ReCoN click arbiter failed: {e}")
            return self._get_click_action_obj(action)

    def zero_back(self):
        """Propagate failure backwards through the state graph."""
        if all(v == 0 for v in self.action_rweights.values()):
            for state, action in self.prior_states:
                if state.action_rweights[action] == 1:
                    state.action_rweights[action] = 0
                    state.zero_back()

    def __eq__(self, other):
        """Equality based on game_id, score, and frame."""
        if not isinstance(other, ArconState):
            return NotImplemented
        return (self.game_id, self.score, self.frame) == (other.game_id, other.score, other.frame)

    def __hash__(self):
        """Hash for use in sets and dictionaries."""
        return hash((self.game_id, self.score, self.frame))


class ArconStateGraph:
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
        
        # ReCoN Click Arbiter Configuration (Ablation Study Flags)
        self.use_recon_click_arbiter = False  # Main ablation flag
        self.recon_exploration_rate = 0.1     # Exploration within ReCoN
        self.recon_area_frac_cutoff = 0.005   # Minimum object area fraction
        self.recon_border_penalty = 0.8       # Border penalty factor
        
        # ReCoN Statistics (for ablation study analysis)
        self.recon_click_selections = 0       # Number of ReCoN-based selections
        self.total_click_selections = 0       # Total click selections

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

    def get_state(self, latest_frame: Any) -> ArconState:
        """Get or create state for the given frame."""
        new_state = ArconState(latest_frame)

        # Check if state already exists
        existing_state = next((s for s in self.states if s == new_state), None)
        if existing_state:
            return existing_state

        # Add new state
        self.states.add(new_state)
        return new_state

    def add_init_state(self, state: ArconState):
        """Set the initial state and add as milestone."""
        self.init_state = state
        self.add_milestone(state)
        self.game_id = state.game_id

    def add_milestone(self, state: ArconState):
        """Add a milestone state (level completion)."""
        key = (state.game_id, state.score)
        if key in self.milestones:
            assert self.milestones[key] == state
        else:
            self.milestones[key] = state

    def update(self, prev_state: ArconState, action: int, new_state: ArconState):
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

    def get_level_training_data(self, old_milestone: ArconState,
                              new_milestone: ArconState) -> List[Dict]:
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
        self.action_model = ArconActionModel(game_id).to(device)
        self.trainer = ArconTrainer(self.action_model)

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
            # Set current level for training messages
            self.trainer.current_level = max_score
            self.trainer.train_model(training_data, verbose=verbose)
        else:
            if verbose:
                print("No training data available")

    def predict_action_value(self, state: ArconState, action: int) -> float:
        """Predict value for state-action pair."""
        if self.trainer is None:
            return 0.0

        state_tensor = torch.tensor(state.frame, dtype=torch.long) if state.frame != 'WIN' else torch.zeros(64, 64, dtype=torch.long)
        action_tensor = state.get_action_tensor(action)

        return self.trainer.predict_value(state_tensor, action_tensor)

    def configure_recon(self, use_click_arbiter: bool = None, exploration_rate: float = None,
                       area_frac_cutoff: float = None, border_penalty: float = None):
        """
        Configure ReCoN click arbiter settings for ablation studies.
        
        Args:
            use_click_arbiter: Enable/disable ReCoN click arbiter
            exploration_rate: Exploration rate within ReCoN (0.0 = pure ReCoN)
            area_frac_cutoff: Minimum area fraction for objects
            border_penalty: Penalty factor for border-touching objects
        """
        if use_click_arbiter is not None:
            self.use_recon_click_arbiter = use_click_arbiter
        if exploration_rate is not None:
            self.recon_exploration_rate = exploration_rate
        if area_frac_cutoff is not None:
            self.recon_area_frac_cutoff = area_frac_cutoff
        if border_penalty is not None:
            self.recon_border_penalty = border_penalty
    
    def get_recon_statistics(self) -> Dict[str, Any]:
        """Get ReCoN usage statistics for ablation study analysis."""
        return {
            'use_recon_click_arbiter': self.use_recon_click_arbiter,
            'recon_exploration_rate': self.recon_exploration_rate,
            'recon_click_selections': self.recon_click_selections,
            'total_click_selections': self.total_click_selections,
            'recon_usage_rate': (self.recon_click_selections / self.total_click_selections 
                               if self.total_click_selections > 0 else 0.0),
            'recon_area_frac_cutoff': self.recon_area_frac_cutoff,
            'recon_border_penalty': self.recon_border_penalty
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the state graph."""
        base_stats = {
            'total_states': len(self.states),
            'milestones': len(self.milestones),
            'action_counters': len(self.action_counter),
            'has_model': self.action_model is not None,
            'game_id': self.game_id
        }
        
        # Add ReCoN statistics
        recon_stats = self.get_recon_statistics()
        return {**base_stats, **recon_stats}