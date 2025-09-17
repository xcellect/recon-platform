"""
Hypothesis Manager

Manages the ReCoN graph structure for action hypotheses and coordinates.
Implements the hierarchical action/coordinate structure described in the plan:
- Root frame_change_hypothesis
- 5 action hypotheses (ACTION1-ACTION5) 
- 1 click hypothesis (ACTION6) with 8x8 region refinement
- CNN terminal for probability computation
"""

import torch
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.graph import ReCoNGraph
from recon_engine.neural_terminal import CNNValidActionTerminal
from .region_aggregator import RegionAggregator


class HypothesisManager:
    """
    Manages the ReCoN hypothesis graph for ARC-AGI action selection.
    
    This class builds and maintains the hierarchical structure described in the plan:
    - Root hypothesis that requests frame change detection
    - Individual action hypotheses for ACTION1-ACTION5
    - Click hypothesis for ACTION6 with regional refinement
    - CNN terminal integration for probability-based weighting
    """
    
    def __init__(self, region_size: int = 8, grid_size: int = 64):
        """
        Initialize the hypothesis manager.
        
        Args:
            region_size: Size of each region for coordinate refinement (default 8)
            grid_size: Size of the full coordinate grid (default 64)
        """
        self.graph = ReCoNGraph()
        self.cnn_terminal = None
        self.region_aggregator = RegionAggregator(region_size, grid_size)
        self.region_size = region_size
        self.grid_size = grid_size
        self.regions_per_dim = grid_size // region_size
        
        # Cache coordinate probabilities for argmax selection
        self.cached_coord_probs = None
        
        self._built = False
        
    def build_structure(self) -> 'HypothesisManager':
        """
        Build the complete hypothesis structure.
        
        Returns:
            Self for method chaining
        """
        if self._built:
            raise ValueError("Structure already built")
        
        self._build_basic_structure()
        self._add_cnn_terminal() 
        self._add_region_hypotheses()
        
        self._built = True
        return self
    
    def _build_basic_structure(self):
        """Build the basic root -> actions structure"""
        # Root hypothesis - represents the overall frame change detection task
        self.graph.add_node("frame_change_hypothesis", node_type="script")
        
        # Individual action hypotheses as SCRIPT nodes with terminal children (ACTION1-ACTION5)
        # This follows the plan's preference for scripts with terminal children
        for i in range(1, 6):
            action_id = f"action_{i}"
            self.graph.add_node(action_id, node_type="script")
            self.graph.add_link("frame_change_hypothesis", action_id, "sub", weight=1.0)
            
            # Each action script has a terminal child for confirmation
            terminal_id = f"{action_id}_terminal"
            self.graph.add_node(terminal_id, node_type="terminal")
            self.graph.add_link(action_id, terminal_id, "sub", weight=1.0)
            
            # Set measurement function to always confirm (action is always possible)
            terminal_node = self.graph.get_node(terminal_id)
            terminal_node.measurement_fn = lambda env=None: 1.0
        
        # Click action hypothesis as SCRIPT with region children (ACTION6)
        self.graph.add_node("action_click", node_type="script")
        self.graph.add_link("frame_change_hypothesis", "action_click", "sub", weight=1.0)
    
    def _add_cnn_terminal(self):
        """Add CNN terminal for action/coordinate prediction"""
        # Create and integrate CNN terminal with GPU acceleration
        self.cnn_terminal = CNNValidActionTerminal("cnn_terminal", use_gpu=True)
        # CRITICAL FIX: Store original measure and create ReCoN wrapper
        self.cnn_terminal._original_measure = self.cnn_terminal.measure
        self.cnn_terminal.measure = lambda env=None: 0.9 if env is None else self.cnn_terminal._original_measure(env)
        self.graph.add_node(self.cnn_terminal)
        
        # Connect to root for global frame analysis
        self.graph.add_link("frame_change_hypothesis", "cnn_terminal", "sub", weight=1.0)
    
    def _add_region_hypotheses(self):
        """Add region hypotheses as TERMINAL nodes under action_click for coordinate refinement"""
        for region_y in range(self.regions_per_dim):
            for region_x in range(self.regions_per_dim):
                region_id = f"region_{region_y}_{region_x}"
                self.graph.add_node(region_id, node_type="terminal")
                self.graph.add_link("action_click", region_id, "sub", weight=1.0)
                
                # Set measurement function to confirm based on region score
                # This will be updated dynamically based on CNN coordinate probabilities
                region_node = self.graph.get_node(region_id)
                region_node.measurement_fn = lambda env=None: 1.0  # Set to 1.0 so sur equals region weight directly
    
    def update_weights_from_frame(self, frame: torch.Tensor):
        """
        Update all link weights based on CNN predictions for the current frame.
        
        Args:
            frame: Input frame tensor of shape (16, 64, 64) - one-hot encoded
        """
        if not self._built:
            raise ValueError("Structure not built - call build_structure() first")
        
        if self.cnn_terminal is None:
            raise ValueError("CNN terminal not available")
        
        # Get CNN predictions
        measurement = self.cnn_terminal.measure(frame)
        result = self.cnn_terminal._process_measurement(measurement)
        
        action_probs = result["action_probabilities"]
        coord_probs = result["coordinate_probabilities"]
        
        # Cache coordinate probabilities for argmax selection
        self.cached_coord_probs = coord_probs
        
        # Update action hypothesis weights (ACTION1-ACTION5)
        for i in range(5):
            action_id = f"action_{i + 1}"
            weight = float(action_probs[i])
            
            # Find and update the sub link weight
            for link in self.graph.get_links(source="frame_change_hypothesis", target=action_id):
                if link.type == "sub":
                    link.weight = weight
        
        # Update click action weight (normalized sum of coordinate probabilities)
        click_weight = float(coord_probs.sum() / (self.grid_size * self.grid_size))
        for link in self.graph.get_links(source="frame_change_hypothesis", target="action_click"):
            if link.type == "sub":
                link.weight = click_weight
        
        # Update region weights using aggregator
        region_scores = self.region_aggregator.aggregate_to_regions(coord_probs)
        
        for region_y in range(self.regions_per_dim):
            for region_x in range(self.regions_per_dim):
                region_id = f"region_{region_y}_{region_x}"
                weight = float(region_scores[region_y, region_x])
                
                # Update sub link weight from action_click to region
                for link in self.graph.get_links(source="action_click", target=region_id):
                    if link.type == "sub":
                        link.weight = weight
    
    def request_frame_change(self):
        """Start hypothesis evaluation by requesting the root"""
        if not self._built:
            raise ValueError("Structure not built")
        
        self.graph.request_root("frame_change_hypothesis")
    
    def propagate_step(self):
        """Execute one step of ReCoN propagation"""
        self.graph.propagate_step()
    
    def get_action_states(self) -> Dict[str, str]:
        """
        Get the current states of all action hypotheses.
        
        Returns:
            Dictionary mapping action IDs to their current states
        """
        states = {}
        
        # Individual actions
        for i in range(1, 6):
            action_id = f"action_{i}"
            if action_id in self.graph.nodes:
                states[action_id] = self.graph.nodes[action_id].state.name
        
        # Click action
        if "action_click" in self.graph.nodes:
            states["action_click"] = self.graph.nodes["action_click"].state.name
        
        return states
    
    def get_region_states(self) -> Dict[str, str]:
        """
        Get the current states of all region hypotheses.
        
        Returns:
            Dictionary mapping region IDs to their current states
        """
        states = {}
        
        for region_y in range(self.regions_per_dim):
            for region_x in range(self.regions_per_dim):
                region_id = f"region_{region_y}_{region_x}"
                if region_id in self.graph.nodes:
                    states[region_id] = self.graph.nodes[region_id].state.name
        
        return states
    
    def get_best_action(self, available_actions: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """
        Get the best action based on current states and activations.
        
        Uses state priority (CONFIRMED > TRUE > WAITING > others) and activation magnitude
        for tie-breaking as described in the plan.
        
        Args:
            available_actions: List of available action names for airtight filtering
        
        Returns:
            Tuple of (action_type, coordinates)
            - action_type: "action_1" through "action_5" or "action_click"
            - coordinates: (y, x) tuple if action_click, None otherwise
        """
        if not self._built:
            raise ValueError("Structure not built")
        
        # AIRTIGHT AVAILABILITY: Create allowed actions set
        allowed_actions = set()
        if available_actions:
            action_mapping = {
                "ACTION1": "action_1",
                "ACTION2": "action_2", 
                "ACTION3": "action_3",
                "ACTION4": "action_4",
                "ACTION5": "action_5",
                "ACTION6": "action_click"
            }
            for harness_action in available_actions:
                if harness_action in action_mapping:
                    allowed_actions.add(action_mapping[harness_action])
        
        # State priority mapping (higher = better)
        state_priority = {
            "CONFIRMED": 4,
            "TRUE": 3,
            "WAITING": 2,
            "ACTIVE": 1,
            "REQUESTED": 1,
            "INACTIVE": 0,
            "FAILED": -1,
            "SUPPRESSED": -1
        }
        
        # Track if we have any viable actions (score > negative)
        viable_actions_found = False
        
        best_action = None
        best_score = float('-inf')
        best_coords = None
        
        # Check individual actions (ACTION1-ACTION5)
        for i in range(1, 6):
            action_id = f"action_{i}"
            if action_id not in self.graph.nodes:
                continue
            
            # AIRTIGHT AVAILABILITY: Skip if not in allowed actions
            if available_actions and action_id not in allowed_actions:
                continue
            
            # PURE RECON AVAILABILITY: Skip if effectively unavailable due to low sub weight
            if self._is_action_effectively_unavailable(action_id):
                continue
                
            node = self.graph.nodes[action_id]
            state_score = state_priority.get(node.state.name, 0)
            
            # AIRTIGHT AVAILABILITY: FAILED and SUPPRESSED actions are never viable
            if state_score < 0:  # FAILED (-1) or SUPPRESSED (-1)
                continue
            
            activation = float(node.activation) if hasattr(node, 'activation') else 0.0
            
            # Combined score: state priority + normalized activation
            total_score = state_score + activation
            
            if total_score >= 0:  # Consider all non-negative actions (including INACTIVE)
                viable_actions_found = True
            
            if total_score > best_score:
                best_score = total_score
                best_action = action_id
                best_coords = None
        
        # Check click action
        if "action_click" in self.graph.nodes:
            # AIRTIGHT AVAILABILITY: Skip if not in allowed actions
            if available_actions and "action_click" not in allowed_actions:
                pass  # Skip click action entirely
            # PURE RECON AVAILABILITY: Skip if effectively unavailable due to low sub weight
            elif self._is_action_effectively_unavailable("action_click"):
                pass  # Skip click action entirely
            else:
                click_node = self.graph.nodes["action_click"]
                click_state_score = state_priority.get(click_node.state.name, 0)
                
                # AIRTIGHT AVAILABILITY: FAILED and SUPPRESSED actions are never viable
                if click_state_score < 0:  # FAILED (-1) or SUPPRESSED (-1)
                    # Skip click action entirely, don't even check regions
                    pass
                else:
                    click_activation = float(click_node.activation) if hasattr(click_node, 'activation') else 0.0
                    
                    click_total_score = click_state_score + click_activation
                    
                    if click_total_score >= 0:  # Consider all non-negative actions (including INACTIVE)
                        viable_actions_found = True
                    
                    # If click action is competitive, find best region
                    if click_total_score > best_score:
                        best_region_score = float('-inf')
                        best_region_coords = None
                        
                        for region_y in range(self.regions_per_dim):
                            for region_x in range(self.regions_per_dim):
                                region_id = f"region_{region_y}_{region_x}"
                                if region_id not in self.graph.nodes:
                                    continue
                                    
                                # PURE RECON AVAILABILITY: Skip if effectively unavailable due to low sub weight
                                if self._is_region_effectively_unavailable(region_id):
                                    continue
                                
                                region_node = self.graph.nodes[region_id]
                                region_state_score = state_priority.get(region_node.state.name, 0)
                                
                                # AIRTIGHT AVAILABILITY: FAILED and SUPPRESSED regions are never viable
                                if region_state_score < 0:  # FAILED (-1) or SUPPRESSED (-1)
                                    continue
                                
                                region_activation = float(region_node.activation) if hasattr(region_node, 'activation') else 0.0
                                
                                region_total_score = region_state_score + region_activation
                                
                                if region_total_score > best_region_score:
                                    best_region_score = region_total_score
                                    # Get argmax coordinate within the region from cached CNN probabilities
                                    best_region_coords = self._get_argmax_in_region(region_y, region_x)
                        
                        if best_region_coords is not None:
                            best_score = click_total_score
                            best_action = "action_click"
                            best_coords = best_region_coords
        
        # If no viable actions found, return None
        if not viable_actions_found or best_score < 0:
            return None, None
        
        return best_action, best_coords
    
    def _is_action_effectively_unavailable(self, action_id: str) -> bool:
        """
        Check if an action is effectively unavailable due to very low sub weight.
        
        This supports pure ReCoN semantics where unavailable actions have near-zero
        sub weights instead of being set to FAILED state.
        
        Args:
            action_id: The action node ID to check
            
        Returns:
            True if the action has a very low sub weight (< 1e-5), False otherwise
        """
        for link in self.graph.get_links(source="frame_change_hypothesis", target=action_id):
            if link.type == "sub" and link.weight < 1e-5:
                return True
        return False
    
    def _is_region_effectively_unavailable(self, region_id: str) -> bool:
        """
        Check if a region is effectively unavailable due to very low sub weight.
        
        Args:
            region_id: The region node ID to check
            
        Returns:
            True if the region has a very low sub weight (< 1e-5), False otherwise
        """
        for link in self.graph.get_links(source="action_click", target=region_id):
            if link.type == "sub" and link.weight < 1e-5:
                return True
        return False
    
    def _get_argmax_in_region(self, region_y: int, region_x: int) -> Tuple[int, int]:
        """
        Get argmax coordinate within a specific region from cached coordinate probabilities.
        
        Args:
            region_y: Region Y index
            region_x: Region X index
            
        Returns:
            Tuple of (pixel_y, pixel_x) coordinates
        """
        if self.cached_coord_probs is None:
            # Fallback to center if no cached probabilities
            pixel_y = region_y * self.region_size + self.region_size // 2
            pixel_x = region_x * self.region_size + self.region_size // 2
            return (pixel_y, pixel_x)
        
        # Get region bounds
        y_start = region_y * self.region_size
        y_end = min(y_start + self.region_size, self.grid_size)
        x_start = region_x * self.region_size  
        x_end = min(x_start + self.region_size, self.grid_size)
        
        # Extract region probabilities
        region_probs = self.cached_coord_probs[y_start:y_end, x_start:x_end]
        
        # Find argmax within region
        flat_idx = region_probs.argmax().item()
        local_y = flat_idx // region_probs.shape[1]
        local_x = flat_idx % region_probs.shape[1]
        
        # Convert to global coordinates
        global_y = y_start + local_y
        global_x = x_start + local_x
        
        return (global_y, global_x)
    
    def reset(self):
        """Reset the graph state for a new evaluation"""
        if not self._built:
            return
            
        # Clear requested roots and message queue
        self.graph.requested_roots.clear()
        self.graph.message_queue.clear()
        self.graph.step_count = 0
        
        # Reset all node states to INACTIVE
        for node in self.graph.nodes.values():
            node.reset()
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the current hypothesis manager state"""
        if not self._built:
            return {"built": False}
        
        stats = {
            "built": True,
            "total_nodes": len(self.graph.nodes),
            "total_links": len(self.graph.links),
            "step_count": self.graph.step_count,
            "requested_roots": len(self.graph.requested_roots),
            "message_queue_size": len(self.graph.message_queue),
            "action_states": self.get_action_states(),
            "cnn_cache_size": len(self.cnn_terminal._output_cache) if self.cnn_terminal else 0
        }
        
        return stats
