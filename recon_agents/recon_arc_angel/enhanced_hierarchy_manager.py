"""
Enhanced Hierarchical Hypothesis Manager

Implements the complete systematic solution to prevent getting stuck:
1. Stable object identity across frames with IoU-based matching
2. ReCoN hypothesis system with systematic reduction (test-and-prune)
3. Causal object-scoped verification (clicked object only)
4. Deterministic evidence-led coordinate selection
5. Fixed CNN probability scaling and gating
6. Principled exploration with hypothesis scheduling
7. Maximum contrast segmentation
"""

import torch
import numpy as np
import scipy.ndimage
import os
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, "/workspace/recon-platform")

try:
    # Try module-qualified imports first (for harness)
    from recon_platform.recon_engine.graph import ReCoNGraph
    from recon_platform.recon_engine.neural_terminal import CNNValidActionTerminal, ResNetActionValueTerminal
    from recon_platform.recon_engine.node import ReCoNState
except ImportError:
    # Fallback to direct imports (for development/testing)
    from recon_engine.graph import ReCoNGraph
    from recon_engine.neural_terminal import CNNValidActionTerminal, ResNetActionValueTerminal
    from recon_engine.node import ReCoNState

from stable_object_tracker import StableObjectTracker, PersistentObject


class EnhancedHierarchicalHypothesisManager:
    """
    Enhanced implementation with systematic hypothesis reduction.
    
    Key improvements over previous versions:
    - Stable object identity prevents bookkeeping errors
    - ReCoN hypothesis system with test-and-prune behavior
    - Causal object-scoped verification
    - Deterministic coordinate selection
    - Fixed CNN probability scaling
    - Principled exploration scheduling
    """
    
    def __init__(self, cnn_threshold: float = 0.1, max_objects: int = 50):
        """
        Initialize enhanced hierarchy manager.
        
        Args:
            cnn_threshold: User-definable threshold for CNN confidence usage
            max_objects: Maximum objects to track per frame
        """
        self.graph = ReCoNGraph()
        self.cnn_threshold = cnn_threshold
        self.max_objects = max_objects
        
        # Neural components with enhanced temperature control
        self.cnn_terminal = None
        self.resnet_terminal = None
        
        # Stable object tracking
        self.object_tracker = StableObjectTracker(iou_threshold=0.3, max_objects=max_objects)
        
        # Causal verification parameters
        self.tau_ratio = 0.02  # 2% pixel change ratio threshold
        self.tau_pixels = 3    # Minimum 3 pixels changed
        self.p_min = 0.15      # Minimum CNN prob to maintain hypothesis
        
        # Exploration parameters
        self.exploration_epsilon = 0.05  # Tie-breaking threshold
        self.exploration_temp = 0.5     # Temperature for tie-breaking
        
        # CNN scaling fix
        self.coord_scale_factor = 4096  # Scale masked_max by this factor
        
        self._built = False
    
    def build_enhanced_structure(self) -> 'EnhancedHierarchicalHypothesisManager':
        """Build the enhanced ReCoN hierarchy with hypothesis system."""
        if self._built:
            raise ValueError("Structure already built")
        
        self._build_root_and_basic_actions()
        self._build_action6_hypothesis_system()
        self._add_neural_terminals()
        
        self._built = True
        return self
    
    def _build_root_and_basic_actions(self):
        """Build root and basic action scripts (ACTION1-ACTION5)"""
        # Root script
        self.graph.add_node("frame_change_hypothesis", node_type="script")
        
        # Individual action scripts with terminal children (actions 1-5)
        for i in range(1, 6):
            action_id = f"action_{i}"
            self.graph.add_node(action_id, node_type="script")
            self.graph.add_link("frame_change_hypothesis", action_id, "sub", weight=1.0)
            
            # Each action script has a terminal child for confirmation
            terminal_id = f"{action_id}_terminal"
            self.graph.add_node(terminal_id, node_type="terminal")
            self.graph.add_link(action_id, terminal_id, "sub", weight=1.0)
            
            # Set user-definable threshold for CNN confidence
            terminal = self.graph.get_node(terminal_id)
            terminal.transition_threshold = self.cnn_threshold
            terminal.measurement_fn = lambda env=None: 0.9  # High confidence for basic actions
    
    def _build_action6_hypothesis_system(self):
        """Build ACTION6 with hypothesis-based ReCoN reduction system."""
        # Main action_click script
        self.graph.add_node("action_click", node_type="script")
        self.graph.add_link("frame_change_hypothesis", "action_click", "sub", weight=1.0)
        
        # Hypothesis manager script
        self.graph.add_node("hypothesis_manager", node_type="script")
        self.graph.add_link("action_click", "hypothesis_manager", "sub", weight=1.0)
        
        # CNN perception stage
        self.graph.add_node("click_cnn", node_type="script")
        self.graph.add_link("hypothesis_manager", "click_cnn", "sub", weight=1.0)
        
        # Object hypothesis testing stage
        self.graph.add_node("hypothesis_testing", node_type="script")
        self.graph.add_link("hypothesis_manager", "hypothesis_testing", "sub", weight=1.0)
        
        # Sequence: CNN perception must complete before hypothesis testing
        self.graph.add_link("click_cnn", "hypothesis_testing", "por", weight=1.0)
    
    def _add_neural_terminals(self):
        """Add neural terminals with enhanced temperature control."""
        # CNN terminal under click_cnn with decoupled softmax
        self.cnn_terminal = CNNValidActionTerminal(
            "cnn_terminal", 
            use_gpu=True, 
            action_temp=1.0,  # Standard for actions
            coord_temp=1.6    # Flattened for coordinates (exploration)
        )
        # CRITICAL FIX: Store original measure and create ReCoN wrapper
        self.cnn_terminal._original_measure = self.cnn_terminal.measure
        self.cnn_terminal.measure = lambda env=None: 0.9 if env is None else self.cnn_terminal._original_measure(env)
        self.graph.add_node(self.cnn_terminal)
        self.graph.add_link("click_cnn", "cnn_terminal", "sub", weight=1.0)
        
        # ResNet terminal under hypothesis_testing for value estimation
        self.resnet_terminal = ResNetActionValueTerminal("resnet_terminal")
        if torch.cuda.is_available():
            self.resnet_terminal.to_device('cuda')
        # CRITICAL FIX: Store original measure and create ReCoN wrapper
        self.resnet_terminal._original_measure = self.resnet_terminal.measure
        self.resnet_terminal.measure = lambda env=None: 0.9 if env is None else self.resnet_terminal._original_measure(env)
        self.graph.add_node(self.resnet_terminal)
        self.graph.add_link("hypothesis_testing", "resnet_terminal", "sub", weight=1.0)
    
    def extract_objects_from_frame(self, frame: torch.Tensor) -> List[dict]:
        """Extract objects with maximum contrast segmentation."""
        # Convert to numpy if needed
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 3:  # One-hot (16, 64, 64)
                frame_np = frame.argmax(dim=0).cpu().numpy()
            else:  # Already (64, 64)
                frame_np = frame.cpu().numpy()
        else:
            frame_np = frame
        
        objects = []
        orig_idx = 0
        
        for colour in range(16):
            # Find connected components of this color
            labeled, num_features = scipy.ndimage.label((frame_np == colour))
            slices = scipy.ndimage.find_objects(labeled)
            
            for i, slc in enumerate(slices):
                if slc is None:
                    continue
                
                mask = (labeled[slc] == (i + 1))
                area = np.sum(mask)
                
                # Skip tiny objects (noise reduction)
                if area < 2:
                    continue
                
                h = slc[0].stop - slc[0].start
                w = slc[1].stop - slc[1].start
                bbox_area = h * w
                area_frac = area / (64 * 64)
                regularity = area / bbox_area
                
                # Background suppression
                touches_left = slc[1].start == 0
                touches_right = slc[1].stop == 64
                touches_top = slc[0].start == 0
                touches_bottom = slc[0].stop == 64
                
                full_width = w >= 60
                full_height = h >= 60
                
                border_penalty = 0.0
                if touches_left and touches_right:
                    border_penalty += 0.5
                if touches_top and touches_bottom:
                    border_penalty += 0.5
                if full_width or full_height:
                    border_penalty += 0.3
                if area_frac >= 0.20:
                    border_penalty += 0.4
                
                # Maximum contrast calculation
                contrast = self._calculate_boundary_contrast(frame_np, mask, slc, colour)
                
                # Enhanced confidence with contrast emphasis
                size_bonus = min(1.0, area_frac / 0.05)
                confidence = regularity * size_bonus * (1.0 - border_penalty) * (0.5 + 0.5 * contrast)
                confidence = max(0.0, min(1.0, confidence))
                
                # Get centroid
                ys, xs = np.nonzero(mask)
                y_centroid = ys.mean() + slc[0].start
                x_centroid = xs.mean() + slc[1].start
                
                objects.append({
                    "orig_idx": orig_idx,
                    "colour": colour,
                    "slice": slc,
                    "mask": mask,
                    "area": area,
                    "bbox_area": bbox_area,
                    "area_frac": area_frac,
                    "regularity": regularity,
                    "border_penalty": border_penalty,
                    "contrast": contrast,
                    "confidence": confidence,
                    "y_centroid": y_centroid,
                    "x_centroid": x_centroid
                })
                orig_idx += 1
        
        # Sort by comprehensive confidence score (best objects first)
        objects.sort(key=lambda obj: (-obj["confidence"], -obj["contrast"], -obj["regularity"]))
        
        return objects[:self.max_objects]
    
    def _calculate_boundary_contrast(self, frame_np: np.ndarray, mask: np.ndarray, 
                                   slc: tuple, object_color: int) -> float:
        """Calculate boundary contrast for maximum contrast segmentation."""
        try:
            y_start, y_end = slc[0].start, slc[0].stop
            x_start, x_end = slc[1].start, slc[1].stop
            
            y_start, y_end = max(0, y_start), min(64, y_end)
            x_start, x_end = max(0, x_start), min(64, x_end)
            
            if y_end <= y_start or x_end <= x_start:
                return 0.0
            
            from scipy.ndimage import binary_dilation
            
            # Create full-frame object mask
            full_object_mask = np.zeros((64, 64), dtype=bool)
            full_object_mask[y_start:y_end, x_start:x_end] = mask
            
            # Dilate to get boundary
            dilated_full = binary_dilation(full_object_mask, structure=np.ones((3, 3)))
            boundary_mask = dilated_full & (~full_object_mask)
            
            if not np.any(boundary_mask):
                return 0.0
            
            # Get boundary coordinates
            boundary_y, boundary_x = np.nonzero(boundary_mask)
            
            # Filter valid coordinates
            valid_mask = (boundary_y < 64) & (boundary_x < 64) & (boundary_y >= 0) & (boundary_x >= 0)
            if not np.any(valid_mask):
                return 0.0
            
            boundary_y = boundary_y[valid_mask]
            boundary_x = boundary_x[valid_mask]
            
            # Calculate contrast
            boundary_colors = frame_np[boundary_y, boundary_x]
            different_color = boundary_colors != object_color
            contrast = np.mean(different_color) if len(boundary_colors) > 0 else 0.0
            
            return float(contrast)
            
        except Exception:
            return 0.5
    
    def update_dynamic_hypotheses(self, frame: torch.Tensor):
        """Update object hypotheses with stable identity tracking."""
        if not self._built:
            raise ValueError("Structure not built")
        
        # Extract objects with maximum contrast segmentation
        raw_objects = self.extract_objects_from_frame(frame)
        
        # Update stable object tracking
        persistent_objects = self.object_tracker.update_objects(raw_objects)
        
        # Remove old hypothesis nodes
        old_hypothesis_nodes = [node_id for node_id in self.graph.nodes.keys() 
                               if node_id.startswith("H[") or node_id.startswith("verify_")]
        for node_id in old_hypothesis_nodes:
            # Remove links
            links_to_remove = [link for link in self.graph.links 
                             if link.source == node_id or link.target == node_id]
            for link in links_to_remove:
                self.graph.links.remove(link)
            # Remove node
            del self.graph.nodes[node_id]
        
        # Add hypothesis nodes for testable objects
        testable_objects = self.object_tracker.get_testable_hypotheses()
        
        for persistent_obj in testable_objects:
            self._add_object_hypothesis(persistent_obj)
    
    def _add_object_hypothesis(self, persistent_obj: PersistentObject):
        """Add ReCoN-compliant hypothesis structure for a persistent object."""
        obj_id = persistent_obj.object_id
        
        # Main hypothesis script H[obj]
        hypothesis_id = f"H[{obj_id}]"
        self.graph.add_node(hypothesis_id, node_type="script")
        self.graph.add_link("hypothesis_testing", hypothesis_id, "sub", weight=1.0)
        
        # Click testing script H[obj]_click
        click_test_id = f"H[{obj_id}]_click"
        self.graph.add_node(click_test_id, node_type="script")
        self.graph.add_link(hypothesis_id, click_test_id, "sub", weight=1.0)
        
        # Verification script H[obj]_verify (script, not terminal)
        verify_script_id = f"H[{obj_id}]_verify"
        self.graph.add_node(verify_script_id, node_type="script")
        self.graph.add_link(hypothesis_id, verify_script_id, "sub", weight=1.0)
        
        # Sequence: click must happen before verification (both are scripts)
        self.graph.add_link(click_test_id, verify_script_id, "por", weight=1.0)
        
        # Click terminal under click script
        click_terminal_id = f"click_{obj_id}_terminal"
        self.graph.add_node(click_terminal_id, node_type="terminal")
        self.graph.add_link(click_test_id, click_terminal_id, "sub", weight=1.0)
        
        # Verification terminal under verification script
        verify_terminal_id = f"verify_{obj_id}_terminal"
        self.graph.add_node(verify_terminal_id, node_type="terminal")
        self.graph.add_link(verify_script_id, verify_terminal_id, "sub", weight=1.0)
        
        # Set up terminals
        click_terminal = self.graph.get_node(click_terminal_id)
        click_terminal.transition_threshold = self.cnn_threshold
        click_terminal.measurement_fn = lambda env=None: 0.9  # Click always succeeds
        
        verify_terminal = self.graph.get_node(verify_terminal_id)
        verify_terminal.transition_threshold = self.cnn_threshold
        # Verification will be set up dynamically based on object-scoped change detection
    
    def calculate_scaled_masked_cnn_probability(self, coord_probs: torch.Tensor, 
                                              persistent_obj: PersistentObject) -> float:
        """
        Calculate properly scaled CNN probability using object mask.
        
        Args:
            coord_probs: CNN coordinate probabilities (64, 64)
            persistent_obj: Persistent object with stable identity
            
        Returns:
            Scaled masked maximum probability
        """
        mask = persistent_obj.current_mask
        
        if not np.any(mask):
            return 0.0
        
        # Apply mask and get maximum
        masked_probs = coord_probs[mask]
        if len(masked_probs) > 0:
            masked_max = masked_probs.max().item()
            # Scale by coordinate space size to get reasonable values
            scaled_max = masked_max * self.coord_scale_factor
            return min(1.0, scaled_max)  # Cap at 1.0
        
        return 0.0
    
    def get_deterministic_coordinate(self, persistent_obj: PersistentObject, 
                                   coord_probs: torch.Tensor) -> Tuple[int, int]:
        """
        Get deterministic coordinate using evidence-led selection.
        
        Args:
            persistent_obj: Persistent object with stable identity
            coord_probs: CNN coordinate probabilities
            
        Returns:
            Deterministic coordinate within object mask
        """
        mask = persistent_obj.current_mask
        
        # Get coordinates within mask
        mask_coords = np.argwhere(mask)
        
        if len(mask_coords) == 0:
            # Fallback to centroid
            slc = persistent_obj.current_slice
            y = max(0, min(63, int((slc[0].start + slc[0].stop) / 2)))
            x = max(0, min(63, int((slc[1].start + slc[1].stop) / 2)))
            return (y, x)
        
        # Find argmax within mask
        best_prob = -1.0
        best_coord = None
        
        for coord_idx in range(len(mask_coords)):
            y, x = mask_coords[coord_idx]
            prob = coord_probs[y, x].item()
            
            if prob > best_prob:
                best_prob = prob
                best_coord = (y, x)
        
        if best_coord is not None:
            return best_coord
        
        # Fallback to first mask coordinate
        y, x = mask_coords[0]
        return (y, x)
    
    def calculate_hypothesis_score(self, persistent_obj: PersistentObject, 
                                 scaled_masked_cnn_prob: float) -> float:
        """
        Calculate comprehensive hypothesis score for systematic scheduling.
        
        Args:
            persistent_obj: Persistent object with stable identity
            scaled_masked_cnn_prob: Scaled masked CNN probability
            
        Returns:
            Comprehensive score for hypothesis scheduling
        """
        # Base components
        cnn_score = scaled_masked_cnn_prob
        regularity_bonus = persistent_obj.regularity * 0.3
        contrast_bonus = persistent_obj.contrast * 0.4
        area_penalty = (persistent_obj.area / (64 * 64)) * 0.5
        border_penalty = 0.4  # Will be calculated from persistent_obj properties
        stale_penalty = 0.2 * persistent_obj.stale_tries
        
        # Status-based gating
        status_multiplier = 1.0
        if persistent_obj.status == "FAILED":
            status_multiplier = 0.0  # Suppress failed hypotheses
        elif persistent_obj.status == "CONFIRMED":
            status_multiplier = 1.2  # Boost confirmed hypotheses
        
        # Comprehensive score
        score = (cnn_score + regularity_bonus + contrast_bonus - 
                area_penalty - border_penalty - stale_penalty) * status_multiplier
        
        return max(0.0, score)
    
    def get_best_hypothesis_with_systematic_reduction(self, available_actions: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[Tuple[int, int]], Optional[str]]:
        """
        Get best action using systematic hypothesis reduction.
        
        Returns:
            Tuple of (action, coords, object_id) where object_id is persistent
        """
        if not self._built:
            raise ValueError("Structure not built")
        
        # Filter available actions
        allowed_actions = set()
        if available_actions:
            action_mapping = {
                "ACTION1": "action_1", "ACTION2": "action_2", "ACTION3": "action_3",
                "ACTION4": "action_4", "ACTION5": "action_5", "ACTION6": "action_click"
            }
            for harness_action in available_actions:
                if harness_action in action_mapping:
                    allowed_actions.add(action_mapping[harness_action])
        
        # State priority scoring
        state_priority = {
            "CONFIRMED": 4, "TRUE": 3, "WAITING": 2,
            "ACTIVE": 1, "REQUESTED": 1, "INACTIVE": 0,
            "FAILED": -1, "SUPPRESSED": -1
        }
        
        best_action = None
        best_score = float('-inf')
        best_coords = None
        best_object_id = None
        
        # Check basic actions (1-5)
        for i in range(1, 6):
            action_id = f"action_{i}"
            
            if available_actions and action_id not in allowed_actions:
                continue
            
            if action_id in self.graph.nodes:
                node = self.graph.nodes[action_id]
                state_score = state_priority.get(node.state.name, 0)
                
                if state_score < 0:
                    continue
                
                activation = float(node.activation) if hasattr(node, 'activation') else 0.0
                total_score = state_score + activation
                
                if total_score > best_score:
                    best_score = total_score
                    best_action = action_id
                    best_coords = None
                    best_object_id = None
        
        # Check ACTION6 with systematic hypothesis reduction
        if "action_click" in self.graph.nodes:
            if not available_actions or "action_click" in allowed_actions:
                click_node = self.graph.nodes["action_click"]
                click_state_score = state_priority.get(click_node.state.name, 0)
                
                if click_state_score >= 0:
                    # Get CNN coordinate probabilities
                    coord_probs = self._get_current_coordinate_probabilities()
                    
                    if coord_probs is not None:
                        # Systematic hypothesis scheduling
                        hypothesis_scores = []
                        
                        for persistent_obj in self.object_tracker.get_testable_hypotheses():
                            scaled_cnn_prob = self.calculate_scaled_masked_cnn_probability(coord_probs, persistent_obj)
                            
                            # Skip if CNN probability too low (prevents ACTION6 with coords=None)
                            if scaled_cnn_prob < self.p_min:
                                continue
                            
                            hypothesis_score = self.calculate_hypothesis_score(persistent_obj, scaled_cnn_prob)
                            
                            if hypothesis_score > 0:
                                hypothesis_scores.append((persistent_obj, hypothesis_score, scaled_cnn_prob))
                        
                        if hypothesis_scores:
                            # Sort by score for principled exploration
                            hypothesis_scores.sort(key=lambda x: -x[1])
                            
                            # Check for ties within epsilon
                            best_score_val = hypothesis_scores[0][1]
                            tied_hypotheses = [h for h in hypothesis_scores if abs(h[1] - best_score_val) <= self.exploration_epsilon]
                            
                            if len(tied_hypotheses) > 1:
                                # Break ties with small softmax
                                scores = torch.tensor([h[1] for h in tied_hypotheses])
                                probs = torch.softmax(scores / self.exploration_temp, dim=0)
                                selected_idx = torch.multinomial(probs, 1).item()
                                selected_hypothesis = tied_hypotheses[selected_idx]
                            else:
                                # Single best hypothesis
                                selected_hypothesis = hypothesis_scores[0]
                            
                            persistent_obj, hypothesis_score, scaled_cnn_prob = selected_hypothesis
                            
                            # Get deterministic coordinate
                            coord = self.get_deterministic_coordinate(persistent_obj, coord_probs)
                            
                            click_total_score = click_state_score + hypothesis_score
                            
                            if click_total_score > best_score:
                                best_score = click_total_score
                                best_action = "action_click"
                                best_coords = coord
                                best_object_id = persistent_obj.object_id
        
        # Health check logging
        if os.getenv('RECON_DEBUG'):
            self._log_hypothesis_health_check(best_action, best_coords, best_object_id)
        
        return best_action, best_coords, best_object_id
    
    def _get_current_coordinate_probabilities(self) -> Optional[torch.Tensor]:
        """Get current CNN coordinate probabilities."""
        try:
            # This would be called after CNN inference in the update cycle
            # For now, return None to indicate no cached probabilities
            return getattr(self, '_cached_coord_probs', None)
        except Exception:
            return None
    
    def _log_hypothesis_health_check(self, best_action: str, best_coords: Tuple[int, int], best_object_id: str):
        """Log comprehensive health check information."""
        print(f"🔍 Enhanced ReCoN Hypothesis System:")
        print(f"  Persistent objects: {len(self.object_tracker.persistent_objects)}")
        print(f"  Testable hypotheses: {len(self.object_tracker.get_testable_hypotheses())}")
        
        # Show hypothesis status
        status_counts = {}
        for obj in self.object_tracker.persistent_objects.values():
            status_counts[obj.status] = status_counts.get(obj.status, 0) + 1
        print(f"  Hypothesis status: {status_counts}")
        
        # Show top-3 hypotheses
        testable = self.object_tracker.get_testable_hypotheses()
        if testable:
            print(f"  Top-3 hypotheses:")
            coord_probs = self._get_current_coordinate_probabilities()
            
            for i, obj in enumerate(testable[:3]):
                if coord_probs is not None:
                    scaled_prob = self.calculate_scaled_masked_cnn_probability(coord_probs, obj)
                    score = self.calculate_hypothesis_score(obj, scaled_prob)
                else:
                    scaled_prob = 0.0
                    score = 0.0
                
                print(f"    {obj.object_id}: status={obj.status}, score={score:.3f}, "
                      f"scaled_cnn={scaled_prob:.3f}, stale_tries={obj.stale_tries}, "
                      f"contrast={obj.contrast:.3f}")
        
        print(f"🎯 Selected: action={best_action}, coords={best_coords}, object_id={best_object_id}")
    
    def verify_clicked_object(self, clicked_coord: Tuple[int, int], 
                            prev_frame_tensor: torch.Tensor, 
                            curr_frame_tensor: torch.Tensor) -> bool:
        """
        Verify if click on specific object caused change (causal verification).
        
        Args:
            clicked_coord: Coordinate that was clicked
            prev_frame_tensor: Previous frame tensor
            curr_frame_tensor: Current frame tensor
            
        Returns:
            True if change detected within the clicked object's mask
        """
        # Find the persistent object that contains the clicked coordinate
        cause_object = self.object_tracker.get_persistent_object_containing_coord(clicked_coord)
        
        if cause_object is None:
            return False
        
        try:
            # Convert tensors to numpy
            if prev_frame_tensor.dim() == 3:
                prev_np = prev_frame_tensor.argmax(dim=0).cpu().numpy()
            else:
                prev_np = prev_frame_tensor.cpu().numpy()
            
            if curr_frame_tensor.dim() == 3:
                curr_np = curr_frame_tensor.argmax(dim=0).cpu().numpy()
            else:
                curr_np = curr_frame_tensor.cpu().numpy()
            
            # Get mask of the cause object
            mask = cause_object.current_mask
            
            # Count changes within the cause object's mask only
            prev_colors = prev_np[mask]
            curr_colors = curr_np[mask]
            
            changed_pixels = (prev_colors != curr_colors).sum()
            change_ratio = changed_pixels / mask.sum() if mask.sum() > 0 else 0.0
            
            # Success if ratio >= tau_ratio OR absolute pixels >= tau_pixels
            success = (change_ratio >= self.tau_ratio) or (changed_pixels >= self.tau_pixels)
            
            if os.getenv('RECON_DEBUG'):
                print(f"  🔍 Causal verification for {cause_object.object_id}:")
                print(f"    Clicked coord: {clicked_coord}")
                print(f"    Changed pixels: {changed_pixels}")
                print(f"    Change ratio: {change_ratio:.3f}")
                print(f"    Success: {success} (tau_ratio={self.tau_ratio}, tau_pixels={self.tau_pixels})")
            
            # Update hypothesis status
            self.object_tracker.record_click_attempt(clicked_coord, success)
            
            return success
            
        except Exception as e:
            if os.getenv('RECON_DEBUG'):
                print(f"  ❌ Error in causal verification: {e}")
            return False
    
    def update_weights_from_cnn_enhanced(self, frame: torch.Tensor):
        """Update link weights using enhanced CNN coupling with proper scaling."""
        if not self._built:
            raise ValueError("Structure not built")
        
        # Update dynamic hypotheses first
        self.update_dynamic_hypotheses(frame)
        
        # Get CNN predictions
        measurement = self.cnn_terminal.measure(frame)
        result = self.cnn_terminal._process_measurement(measurement)
        
        action_probs = result["action_probabilities"]
        coord_probs = result["coordinate_probabilities"]
        
        # Cache coordinate probabilities for deterministic selection
        self._cached_coord_probs = coord_probs
        
        # Update basic action weights
        for i in range(5):
            action_id = f"action_{i + 1}"
            weight = float(action_probs[i])
            
            for link in self.graph.get_links(source="frame_change_hypothesis", target=action_id):
                if link.type == "sub":
                    link.weight = weight
        
        # Update hypothesis weights with proper scaling
        total_click_prob = 0.0
        testable_hypotheses = self.object_tracker.get_testable_hypotheses()
        
        for persistent_obj in testable_hypotheses:
            scaled_prob = self.calculate_scaled_masked_cnn_probability(coord_probs, persistent_obj)
            total_click_prob += scaled_prob
            
            # Update hypothesis link weight
            hypothesis_id = f"H[{persistent_obj.object_id}]"
            if hypothesis_id in self.graph.nodes:
                for link in self.graph.get_links(source="hypothesis_testing", target=hypothesis_id):
                    if link.type == "sub":
                        link.weight = scaled_prob
        
        # Update action_click weight
        click_weight = min(1.0, total_click_prob / len(testable_hypotheses)) if testable_hypotheses else 0.0
        for link in self.graph.get_links(source="frame_change_hypothesis", target="action_click"):
            if link.type == "sub":
                link.weight = max(0.5, click_weight)  # Ensure minimum viable weight
    
    # Compatibility methods
    def build_structure(self):
        """Compatibility method"""
        return self.build_enhanced_structure()
    
    def update_weights_from_frame(self, frame: torch.Tensor):
        """Compatibility method"""
        return self.update_weights_from_cnn_enhanced(frame)
    
    def get_best_action(self, available_actions: Optional[List[str]] = None):
        """Compatibility method"""
        action, coords, object_id = self.get_best_hypothesis_with_systematic_reduction(available_actions)
        return action, coords
    
    # The main method is already implemented above as get_best_hypothesis_with_systematic_reduction
    
    def apply_availability_mask(self, available_actions: List[str]):
        """Apply availability mask using link weight reduction."""
        action_mapping = {
            "ACTION1": "action_1", "ACTION2": "action_2", "ACTION3": "action_3",
            "ACTION4": "action_4", "ACTION5": "action_5", "ACTION6": "action_click"
        }
        
        for harness_action, node_id in action_mapping.items():
            if harness_action not in available_actions:
                # Set link weight to near-zero for unavailable actions
                for link in self.graph.get_links(source="frame_change_hypothesis", target=node_id):
                    if link.type == "sub":
                        link.weight = 1e-6
    
    def request_frame_change(self):
        """Start hypothesis evaluation"""
        self.graph.request_root("frame_change_hypothesis")
    
    def propagate_step(self):
        """Execute one ReCoN propagation step"""
        self.graph.propagate_step()
    
    def reset(self):
        """Reset the graph state"""
        if not self._built:
            return
        
        self.graph.requested_roots.clear()
        self.graph.message_queue.clear()
        self.graph.step_count = 0
        
        for node in self.graph.nodes.values():
            node.reset()
    
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics."""
        if not self._built:
            return {"built": False}
        
        return {
            "built": True,
            "total_nodes": len(self.graph.nodes),
            "total_links": len(self.graph.links),
            "step_count": self.graph.step_count,
            "object_tracker": self.object_tracker.get_stats(),
            "cnn_threshold": self.cnn_threshold,
            "max_objects": self.max_objects,
            "tau_ratio": self.tau_ratio,
            "tau_pixels": self.tau_pixels,
            "p_min": self.p_min,
            "coord_scale_factor": self.coord_scale_factor,
            "exploration_epsilon": self.exploration_epsilon,
            "enhancement": "Stable object identity, hypothesis reduction, causal verification, deterministic selection"
        }
