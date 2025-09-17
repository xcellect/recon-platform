"""
Improved Hierarchical Hypothesis Manager

Implements the complete improved ReCoN ARC Angel solution based on agent suggestions:
1. Proper ReCoN graph structure with por/ret sequences for ACTION6
2. Mask-aware CNN coupling using masked max instead of bounding-box max
3. Background suppression with area fraction and border penalties  
4. Improved selection scoring with comprehensive object evaluation
5. Stickiness mechanism for successful clicks that cause frame changes
6. Pure ReCoN execution semantics with proper message passing
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


class ImprovedHierarchicalHypothesisManager:
    """
    Improved implementation following the agent's rigorous ReCoN plan.
    
    Key improvements:
    - Proper por/ret sequence: action_click -> click_cnn -> click_objects
    - CNN and ResNet terminals under appropriate script nodes
    - Mask-aware CNN coupling with masked max calculation
    - Background suppression using area fraction and border penalties
    - Comprehensive selection scoring
    - Stickiness mechanism for frame change persistence
    """
    
    def __init__(self, cnn_threshold: float = 0.1, max_objects: int = 50):
        """
        Initialize improved hierarchy manager.
        
        Args:
            cnn_threshold: User-definable threshold for CNN confidence usage
            max_objects: Maximum objects to track per frame (efficiency limit)
        """
        self.graph = ReCoNGraph()
        self.cnn_threshold = cnn_threshold
        self.max_objects = max_objects
        
        # Neural components
        self.cnn_terminal = None
        self.resnet_terminal = None
        
        # Dynamic state
        self.current_objects = []
        self._built = False
        
        # Improved stickiness mechanism with object-scoped change detection
        self.last_click = {
            'coords': None,
            'obj_idx': None,
            'mask': None,
            'frame_tensor': None
        }
        self.stickiness_strength = 1.0
        self.stickiness_decay_rate = 0.8
        self.sticky_attempts = 0
        self.max_sticky_attempts = 2  # K=2 stale attempts
        
        # Stickiness thresholds
        self.tau_ratio = 0.02  # 2% pixel change ratio threshold
        self.tau_pixels = 3    # Minimum 3 pixels changed
        self.p_min = 0.15      # Minimum CNN prob to maintain stickiness
        
        # Per-object stale penalty tracking
        self.stale_tries = {}  # obj_idx -> number of stale attempts
        self.stale_penalty_lambda = 0.2  # Penalty factor for stale tries
    
    def build_improved_structure(self) -> 'ImprovedHierarchicalHypothesisManager':
        """Build the improved ReCoN hierarchy structure with proper por/ret sequences"""
        if self._built:
            raise ValueError("Structure already built")
        
        self._build_root_and_basic_actions()
        self._build_action6_sequence()
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
    
    def _build_action6_sequence(self):
        """Build ACTION6 with proper por/ret sequence: action_click -> click_cnn -> click_objects"""
        # Main action_click script
        self.graph.add_node("action_click", node_type="script")
        self.graph.add_link("frame_change_hypothesis", "action_click", "sub", weight=1.0)
        
        # First stage: click_cnn (perception/CNN processing)
        self.graph.add_node("click_cnn", node_type="script")
        self.graph.add_link("action_click", "click_cnn", "sub", weight=1.0)
        
        # Second stage: click_objects (object selection/decision)
        self.graph.add_node("click_objects", node_type="script")
        self.graph.add_link("action_click", "click_objects", "sub", weight=1.0)
        
        # Sequence constraint: click_cnn must complete before click_objects starts
        self.graph.add_link("click_cnn", "click_objects", "por", weight=1.0)
        # This automatically creates the reciprocal ret link: click_objects -> click_cnn
    
    def _add_neural_terminals(self):
        """Add neural terminals under appropriate script nodes"""
        # CNN terminal under click_cnn with enhanced temperature control
        self.cnn_terminal = CNNValidActionTerminal(
            "cnn_terminal", 
            use_gpu=True, 
            action_temp=1.0,  # Standard for actions
            coord_temp=1.6    # Flattened for coordinates (fixes coupling)
        )
        self.graph.add_node(self.cnn_terminal)
        self.graph.add_link("click_cnn", "cnn_terminal", "sub", weight=1.0)
        
        # ResNet terminal under click_objects (decision stage) for value estimation
        self.resnet_terminal = ResNetActionValueTerminal("resnet_terminal")
        if torch.cuda.is_available():
            self.resnet_terminal.to_device('cuda')
        self.graph.add_node(self.resnet_terminal)
        self.graph.add_link("click_objects", "resnet_terminal", "sub", weight=1.0)
    
    def extract_objects_from_frame(self, frame: torch.Tensor) -> List[dict]:
        """
        Extract objects with improved background suppression and confidence scoring.
        
        Args:
            frame: One-hot tensor (16, 64, 64) or numpy array (64, 64)
            
        Returns:
            List of object dictionaries with comprehensive properties
        """
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
                area_frac = area / (64 * 64)  # Fraction of total frame
                regularity = area / bbox_area
                
                # Background suppression: detect background-like objects
                # Penalize objects that span full width/height or touch opposite borders
                touches_left = slc[1].start == 0
                touches_right = slc[1].stop == 64
                touches_top = slc[0].start == 0
                touches_bottom = slc[0].stop == 64
                
                full_width = w >= 60  # Nearly full width
                full_height = h >= 60  # Nearly full height
                
                border_penalty = 0.0
                if touches_left and touches_right:
                    border_penalty += 0.5  # Horizontal span
                if touches_top and touches_bottom:
                    border_penalty += 0.5  # Vertical span
                if full_width or full_height:
                    border_penalty += 0.3  # Large dimension
                
                # Large area fraction penalty
                if area_frac >= 0.20:
                    border_penalty += 0.4
                
                # Calculate boundary contrast for high-contrast object preference
                contrast = self._calculate_boundary_contrast(frame_np, mask, slc, colour)
                
                # Calculate comprehensive confidence score with contrast
                size_bonus = min(1.0, area_frac / 0.05)  # Bonus for reasonable size
                confidence = regularity * size_bonus * (1.0 - border_penalty) * (0.5 + 0.5 * contrast)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                
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
        objects.sort(key=lambda obj: (-obj["confidence"], -obj["regularity"], -obj["area"]))
        
        # Limit to max_objects for efficiency
        return objects[:self.max_objects]
    
    def _calculate_boundary_contrast(self, frame_np: np.ndarray, mask: np.ndarray, 
                                   slc: tuple, object_color: int) -> float:
        """
        Calculate boundary contrast for an object to emphasize high-contrast objects.
        
        Args:
            frame_np: Full frame array (64, 64)
            mask: Object mask within slice bounds
            slc: Slice bounds (y_slice, x_slice)
            object_color: Color of the object
            
        Returns:
            Contrast ratio [0,1] where 1 = high contrast with surroundings
        """
        try:
            # Get object bounds
            y_start, y_end = slc[0].start, slc[0].stop
            x_start, x_end = slc[1].start, slc[1].stop
            
            # Ensure bounds are within frame
            y_start, y_end = max(0, y_start), min(64, y_end)
            x_start, x_end = max(0, x_start), min(64, x_end)
            
            if y_end <= y_start or x_end <= x_start:
                return 0.0
            
            # Create boundary by working directly in full frame coordinates
            from scipy.ndimage import binary_dilation
            
            # Create full-frame object mask
            full_object_mask = np.zeros((64, 64), dtype=bool)
            full_object_mask[y_start:y_end, x_start:x_end] = mask
            
            # Dilate the full-frame mask to get boundary
            dilated_full = binary_dilation(full_object_mask, structure=np.ones((3, 3)))
            
            # Boundary is dilated - original mask
            boundary_mask = dilated_full & (~full_object_mask)
            
            if not np.any(boundary_mask):
                return 0.0
            
            # Get boundary coordinates
            boundary_y, boundary_x = np.nonzero(boundary_mask)
            
            # Ensure coordinates are within frame bounds
            valid_mask = (boundary_y < 64) & (boundary_x < 64) & (boundary_y >= 0) & (boundary_x >= 0)
            if not np.any(valid_mask):
                return 0.0
            
            boundary_y = boundary_y[valid_mask]
            boundary_x = boundary_x[valid_mask]
            
            # Get colors at boundary pixels
            boundary_colors = frame_np[boundary_y, boundary_x]
            
            # Calculate contrast as fraction of boundary pixels with different color
            different_color = boundary_colors != object_color
            contrast = np.mean(different_color) if len(boundary_colors) > 0 else 0.0
            
            return float(contrast)
            
        except Exception:
            # Fallback to neutral contrast on any error
            return 0.5
    
    def update_dynamic_objects_improved(self, frame: torch.Tensor):
        """
        Update object hierarchy with improved background suppression.
        """
        if not self._built:
            raise ValueError("Structure not built")
        
        # Extract objects with improved algorithm
        self.current_objects = self.extract_objects_from_frame(frame)
        
        # Remove old object nodes
        old_object_nodes = [node_id for node_id in self.graph.nodes.keys() 
                           if node_id.startswith("object_")]
        for node_id in old_object_nodes:
            # Remove links
            links_to_remove = [link for link in self.graph.links 
                             if link.source == node_id or link.target == node_id]
            for link in links_to_remove:
                self.graph.links.remove(link)
            # Remove node
            del self.graph.nodes[node_id]
        
        # Clear stale tries for disappeared objects
        current_obj_indices = set(range(len(self.current_objects)))
        self.stale_tries = {idx: tries for idx, tries in self.stale_tries.items() 
                           if idx in current_obj_indices}
        
        # Add new object nodes as terminals under click_objects
        for obj_idx, obj in enumerate(self.current_objects):
            object_id = f"object_{obj_idx}"
            self.graph.add_node(object_id, node_type="terminal")
            self.graph.add_link("click_objects", object_id, "sub", weight=1.0)
            
            # Set terminal measurement based on comprehensive confidence
            terminal = self.graph.get_node(object_id)
            terminal.transition_threshold = self.cnn_threshold
            
            # Use comprehensive confidence score
            confidence = obj["confidence"]
            terminal.measurement_fn = lambda env=None, conf=confidence: conf
    
    def calculate_masked_cnn_probability(self, coord_probs: torch.Tensor, obj: dict) -> float:
        """
        Calculate CNN probability using masked max instead of bounding-box max.
        
        Args:
            coord_probs: CNN coordinate probabilities (64, 64)
            obj: Object dictionary with slice and mask
            
        Returns:
            Masked maximum probability within the object
        """
        slc = obj["slice"]
        mask = obj["mask"]
        
        # Get bounding box region
        y_start, y_end = slc[0].start, slc[0].stop
        x_start, x_end = slc[1].start, slc[1].stop
        
        # Ensure bounds
        y_start, y_end = max(0, y_start), min(64, y_end)
        x_start, x_end = max(0, x_start), min(64, x_end)
        
        if y_end <= y_start or x_end <= x_start:
            return 0.0
        
        # Extract region and apply mask
        region_probs = coord_probs[y_start:y_end, x_start:x_end]
        
        # Apply mask to only consider probabilities inside the actual object
        if mask.shape == region_probs.shape:
            masked_probs = region_probs[mask]
            if len(masked_probs) > 0:
                return masked_probs.max().item()
        
        # Fallback to region max if mask doesn't align
        return region_probs.max().item()
    
    def update_weights_from_cnn_improved(self, frame: torch.Tensor):
        """
        Update link weights using mask-aware CNN coupling.
        """
        if not self._built:
            raise ValueError("Structure not built")
        
        # Update dynamic objects first
        self.update_dynamic_objects_improved(frame)
        
        # Get CNN predictions
        measurement = self.cnn_terminal.measure(frame)
        result = self.cnn_terminal._process_measurement(measurement)
        
        action_probs = result["action_probabilities"]
        coord_probs = result["coordinate_probabilities"]  # 64x64
        
        # Update basic action weights (root → action_i)
        for i in range(5):
            action_id = f"action_{i + 1}"
            weight = float(action_probs[i])
            
            for link in self.graph.get_links(source="frame_change_hypothesis", target=action_id):
                if link.type == "sub":
                    link.weight = weight
        
        # Update object weights using masked max CNN probabilities
        total_click_prob = 0.0
        for obj_idx, obj in enumerate(self.current_objects):
            object_id = f"object_{obj_idx}"
            
            if object_id in self.graph.nodes:
                # Use masked max with proper scaling (fixes CNN probability magnitude)
                masked_max_prob = self.calculate_masked_cnn_probability(coord_probs, obj)
                
                # Scale up CNN probabilities to reasonable range (fixes comp_score collapse)
                coord_scale_factor = 4096  # Scale by coordinate space size
                scaled_masked_prob = min(1.0, masked_max_prob * coord_scale_factor)
                
                if scaled_masked_prob < 0.15:  # p_min threshold
                    scaled_masked_prob = 0.0  # Gate low probabilities
                
                total_click_prob += scaled_masked_prob
                
                # Update link weight from click_objects to object
                for link in self.graph.get_links(source="click_objects", target=object_id):
                    if link.type == "sub":
                        link.weight = scaled_masked_prob
        
        # Update action_click weight (ensure minimum viable weight for ACTION6)
        if self.current_objects:
            click_weight = max(0.5, min(1.0, total_click_prob / len(self.current_objects)))
        else:
            click_weight = 0.5  # Default reasonable weight for ACTION6
            
        for link in self.graph.get_links(source="frame_change_hypothesis", target="action_click"):
            if link.type == "sub":
                link.weight = click_weight
    
    def calculate_comprehensive_object_score(self, obj_idx: int, masked_max_cnn_prob: float) -> float:
        """
        Calculate comprehensive object score for selection with improved contrast and stickiness.
        
        Args:
            obj_idx: Object index
            masked_max_cnn_prob: Masked maximum CNN probability
            
        Returns:
            Comprehensive score for object selection
        """
        if obj_idx >= len(self.current_objects):
            return 0.0
        
        obj = self.current_objects[obj_idx]
        
        # Base score components with contrast
        cnn_score = masked_max_cnn_prob
        regularity_bonus = obj["regularity"] * 0.3
        contrast_bonus = obj["contrast"] * 0.4  # Emphasize high-contrast objects
        area_penalty = obj["area_frac"] * 0.5  # Penalize large objects
        border_penalty = obj["border_penalty"] * 0.4
        stale_penalty = self.get_stale_penalty(obj_idx)  # Penalize repeatedly clicked objects
        
        # Improved stickiness bonus with conservative gating and capping
        stickiness_bonus = 0.0
        if (self.last_click['coords'] is not None and 
            self.last_click['obj_idx'] == obj_idx and 
            self.stickiness_strength > 0 and
            masked_max_cnn_prob >= self.p_min):  # Gate: only if CNN prob is decent
            
            # Cap stickiness bonus conservatively
            max_bonus = min(0.5 * self.stickiness_strength, 0.5 * masked_max_cnn_prob)
            stickiness_bonus = max_bonus
        
        # Comprehensive score with improved weighting and stale penalty
        score = (cnn_score + regularity_bonus + contrast_bonus + stickiness_bonus - 
                area_penalty - border_penalty - stale_penalty)
        
        return max(0.0, score)
    
    def get_best_action_with_improved_scoring(self, available_actions: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[Tuple[int, int]], Optional[int]]:
        """
        Get best action using improved comprehensive scoring.
        
        Returns:
            Tuple of (action, coords, obj_idx) where obj_idx is the selected object index for ACTION6
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
        best_obj_idx = None
        
        # Check basic actions (1-5)
        for i in range(1, 6):
            action_id = f"action_{i}"
            
            if available_actions and action_id not in allowed_actions:
                continue
            
            if action_id in self.graph.nodes:
                node = self.graph.nodes[action_id]
                state_score = state_priority.get(node.state.name, 0)
                
                if state_score < 0:  # Skip failed
                    continue
                
                activation = float(node.activation) if hasattr(node, 'activation') else 0.0
                total_score = state_score + activation
                
                if total_score > best_score:
                    best_score = total_score
                    best_action = action_id
                    best_coords = None
                    best_obj_idx = None
        
        # Check action_click with comprehensive object scoring
        if "action_click" in self.graph.nodes:
            if not available_actions or "action_click" in allowed_actions:
                click_node = self.graph.nodes["action_click"]
                click_state_score = state_priority.get(click_node.state.name, 0)
                
                if click_state_score >= 0:
                    click_activation = float(click_node.activation) if hasattr(click_node, 'activation') else 0.0
                    
                    # Find best object using top-K probabilistic selection
                    object_scores = []
                    
                    for obj_idx, obj in enumerate(self.current_objects):
                        object_id = f"object_{obj_idx}"
                        
                        if object_id in self.graph.nodes:
                            # Get masked CNN probability from link weight
                            masked_max_prob = 0.0
                            for link in self.graph.get_links(source="click_objects", target=object_id):
                                if link.type == "sub":
                                    masked_max_prob = float(link.weight)
                                    break
                            
                            # Scale CNN probability to prevent score collapse
                            scaled_prob = min(1.0, masked_max_prob * 4096)  # Fix CNN scaling
                            
                            # Only consider objects with decent scaled probability (prevents ACTION6 coords=None)
                            if scaled_prob < 0.15:
                                continue
                            
                            # Calculate comprehensive score with scaled probability
                            obj_score = self.calculate_comprehensive_object_score(obj_idx, scaled_prob)
                            
                            if obj_score > 0:  # Only consider objects with positive scores
                                object_scores.append((obj_idx, obj_score, scaled_prob))
                    
                    # Top-K probabilistic selection to prevent getting stuck
                    best_object_coord = None
                    best_selected_obj_idx = None
                    best_object_score = 0.0
                    
                    if object_scores:
                        # Sort by score and take top K=3
                        object_scores.sort(key=lambda x: -x[1])
                        top_k = object_scores[:3]
                        
                        if len(top_k) == 1:
                            # Only one viable object
                            best_selected_obj_idx = top_k[0][0]
                        else:
                            # Probabilistic selection from top-K with small temperature
                            scores = torch.tensor([score for _, score, _ in top_k])
                            probs = torch.softmax(scores / 0.5, dim=0)  # Small temperature for exploration
                            
                            # Sample from distribution
                            selected_idx = torch.multinomial(probs, 1).item()
                            best_selected_obj_idx = top_k[selected_idx][0]
                        
                        best_object_score = object_scores[0][1]  # Use best score for total calculation
                        best_object_coord = self._get_object_coordinate_improved(best_selected_obj_idx)
                    
                    # Total click score
                    click_total_score = click_state_score + click_activation + best_object_score
                    
                    if click_total_score > best_score and best_object_coord is not None:
                        best_score = click_total_score
                        best_action = "action_click"
                        best_coords = best_object_coord
                        best_obj_idx = best_selected_obj_idx
        
        # Debug logging
        if os.getenv('RECON_DEBUG'):
            print(f"🔍 Improved ReCoN Network State:")
            print(f"  Objects detected: {len(self.current_objects)}")
            print(f"  Available actions: {available_actions}")
            print(f"  Allowed actions: {allowed_actions}")
            print(f"  Stickiness: {self.stickiness_strength:.3f} at {self.last_click['coords']}, attempts: {self.sticky_attempts}")
            
            # Show comprehensive object scores
            if self.current_objects:
                print(f"  Top objects (comprehensive scoring):")
                for obj_idx in range(min(5, len(self.current_objects))):
                    obj = self.current_objects[obj_idx]
                    object_id = f"object_{obj_idx}"
                    
                    masked_max_prob = 0.0
                    if object_id in self.graph.nodes:
                        for link in self.graph.get_links(source="click_objects", target=object_id):
                            if link.type == "sub":
                                masked_max_prob = float(link.weight)
                                break
                    
                    comp_score = self.calculate_comprehensive_object_score(obj_idx, masked_max_prob)
                    
                    stale_count = self.stale_tries.get(obj_idx, 0)
                    stale_penalty = self.get_stale_penalty(obj_idx)
                    
                    print(f"    {object_id}: masked_max={masked_max_prob:.3f}, "
                          f"regularity={obj['regularity']:.3f}, "
                          f"contrast={obj['contrast']:.3f}, "
                          f"area_frac={obj['area_frac']:.3f}, "
                          f"border_penalty={obj['border_penalty']:.3f}, "
                          f"stale_tries={stale_count}, "
                          f"stale_penalty={stale_penalty:.3f}, "
                          f"confidence={obj['confidence']:.3f}, "
                          f"comp_score={comp_score:.3f}")
            
            print(f"🎯 Selected: action={best_action}, coords={best_coords}, score={best_score:.3f}")
        
        return best_action, best_coords, best_obj_idx
    
    def _get_object_coordinate_improved(self, object_index: int) -> Optional[Tuple[int, int]]:
        """Get coordinate strictly within object mask using improved method"""
        if 0 <= object_index < len(self.current_objects):
            obj = self.current_objects[object_index]
            
            # Get random point within the object mask (strict adherence)
            slc = obj["slice"]
            mask = obj["mask"]
            mask_coords = np.argwhere(mask)
            
            if len(mask_coords) > 0:
                # Select random coordinate from mask
                idx = np.random.choice(len(mask_coords))
                local_y, local_x = mask_coords[idx]
                global_y = slc[0].start + local_y
                global_x = slc[1].start + local_x
                
                # Ensure bounds
                global_y = max(0, min(63, global_y))
                global_x = max(0, min(63, global_x))
                
                return (global_y, global_x)
        
        return None
    
    def _create_full_frame_mask(self, obj_idx: int) -> Optional[np.ndarray]:
        """
        Create a full-frame boolean mask for an object.
        
        Args:
            obj_idx: Object index
            
        Returns:
            64x64 boolean mask or None if object doesn't exist
        """
        if obj_idx >= len(self.current_objects):
            return None
        
        obj = self.current_objects[obj_idx]
        slc = obj["slice"]
        local_mask = obj["mask"]
        
        # Create full-frame mask
        full_mask = np.zeros((64, 64), dtype=bool)
        
        # Map local mask to full frame coordinates
        y_start, y_end = slc[0].start, slc[0].stop
        x_start, x_end = slc[1].start, slc[1].stop
        
        # Ensure bounds
        y_start, y_end = max(0, y_start), min(64, y_end)
        x_start, x_end = max(0, x_start), min(64, x_end)
        
        if y_end > y_start and x_end > x_start:
            # Map local mask to full frame
            local_h, local_w = local_mask.shape
            frame_h = y_end - y_start
            frame_w = x_end - x_start
            
            # Handle size mismatches
            if local_h == frame_h and local_w == frame_w:
                full_mask[y_start:y_end, x_start:x_end] = local_mask
            else:
                # Fallback: mark the entire bounding box
                full_mask[y_start:y_end, x_start:x_end] = True
        
        return full_mask
    
    def detect_object_scoped_change(self, current_frame_tensor: torch.Tensor) -> bool:
        """
        Detect object-scoped changes using the clicked object's mask.
        
        Args:
            current_frame_tensor: Current frame tensor
            
        Returns:
            True if significant change detected within the clicked object's mask
        """
        if (self.last_click['coords'] is None or 
            self.last_click['mask'] is None or 
            self.last_click['frame_tensor'] is None):
            return False
        
        try:
            # Convert tensors to numpy arrays (64, 64)
            if current_frame_tensor.dim() == 3:  # One-hot (16, 64, 64)
                curr_np = current_frame_tensor.argmax(dim=0).cpu().numpy()
            else:
                curr_np = current_frame_tensor.cpu().numpy()
            
            if self.last_click['frame_tensor'].dim() == 3:
                prev_np = self.last_click['frame_tensor'].argmax(dim=0).cpu().numpy()
            else:
                prev_np = self.last_click['frame_tensor'].cpu().numpy()
            
            # Get the clicked object's mask in full frame coordinates
            obj_idx = self.last_click['obj_idx']
            if obj_idx is None or obj_idx >= len(self.current_objects):
                return False
            
            # Use the stored mask from when the click was made
            mask = self.last_click['mask']
            
            # Count changes within the mask
            mask_pixels = mask.sum()
            if mask_pixels == 0:
                return False
            
            # Apply mask to both frames and compare
            prev_masked = prev_np[mask]
            curr_masked = curr_np[mask]
            
            changed_pixels = (prev_masked != curr_masked).sum()
            change_ratio = changed_pixels / mask_pixels
            
            # Success if ratio >= tau_ratio OR absolute pixels >= tau_pixels
            success = (change_ratio >= self.tau_ratio) or (changed_pixels >= self.tau_pixels)
            
            if os.getenv('RECON_DEBUG'):
                print(f"  🔍 Object-scoped change detection:")
                print(f"    Changed pixels: {changed_pixels}")
                print(f"    Change ratio: {change_ratio:.3f}")
                print(f"    Mask pixels: {mask_pixels}")
                print(f"    Success: {success} (tau_ratio={self.tau_ratio}, tau_pixels={self.tau_pixels})")
            
            return success
            
        except Exception as e:
            if os.getenv('RECON_DEBUG'):
                print(f"  ❌ Error in object-scoped change detection: {e}")
            return False
    
    def record_successful_click(self, coord: Tuple[int, int], obj_idx: int, 
                               frame_tensor: torch.Tensor, obj_mask: np.ndarray):
        """
        Record a successful click with object-scoped information.
        
        Args:
            coord: Click coordinates
            obj_idx: Index of clicked object
            frame_tensor: Current frame tensor
            obj_mask: Full-frame boolean mask of the clicked object
        """
        self.last_click = {
            'coords': coord,
            'obj_idx': obj_idx,
            'mask': obj_mask.copy(),
            'frame_tensor': frame_tensor.clone() if isinstance(frame_tensor, torch.Tensor) else frame_tensor
        }
        self.stickiness_strength = 1.0
        self.sticky_attempts = 0
        
        if os.getenv('RECON_DEBUG'):
            print(f"  🎯 Recorded successful click: coord={coord}, obj_idx={obj_idx}, mask_size={obj_mask.sum()}")
    
    def update_stickiness(self, current_frame_tensor: torch.Tensor, 
                         masked_max_cnn_prob: float, action6_available: bool = True):
        """
        Update stickiness based on object-scoped change detection.
        
        Args:
            current_frame_tensor: Current frame tensor
            masked_max_cnn_prob: CNN probability for the sticky object
            action6_available: Whether ACTION6 is available
        """
        if self.last_click['coords'] is None:
            return
        
        # Clear stickiness if ACTION6 unavailable or object vanished
        if not action6_available or masked_max_cnn_prob < self.p_min:
            if os.getenv('RECON_DEBUG'):
                print(f"  🚫 Clearing stickiness: ACTION6_available={action6_available}, CNN_prob={masked_max_cnn_prob:.3f}")
            self.clear_stickiness()
            return
        
        # Check for object-scoped changes
        change_detected = self.detect_object_scoped_change(current_frame_tensor)
        
        if change_detected:
            # Reset attempts on successful change
            self.sticky_attempts = 0
            if os.getenv('RECON_DEBUG'):
                print(f"  ✅ Object change detected, maintaining stickiness")
        else:
            # Increment stale attempts
            self.sticky_attempts += 1
            if os.getenv('RECON_DEBUG'):
                print(f"  ⏳ No change detected, sticky_attempts={self.sticky_attempts}/{self.max_sticky_attempts}")
            
            # Clear after max stale attempts
            if self.sticky_attempts >= self.max_sticky_attempts:
                if os.getenv('RECON_DEBUG'):
                    print(f"  🚫 Clearing stickiness after {self.max_sticky_attempts} stale attempts")
                self.clear_stickiness()
                return
        
        # Update frame tensor for next comparison
        self.last_click['frame_tensor'] = current_frame_tensor.clone() if isinstance(current_frame_tensor, torch.Tensor) else current_frame_tensor
    
    def clear_stickiness(self):
        """Clear all stickiness state"""
        self.last_click = {
            'coords': None,
            'obj_idx': None,
            'mask': None,
            'frame_tensor': None
        }
        self.stickiness_strength = 0.0
        self.sticky_attempts = 0
    
    def clear_cnn_cache_on_stale_click(self):
        """Clear CNN cache to enable fresh inference after stale clicks."""
        if hasattr(self.cnn_terminal, 'clear_cache'):
            self.cnn_terminal.clear_cache()
            if os.getenv('RECON_DEBUG'):
                print(f"  🔄 Cleared CNN cache to enable fresh inference")
        
        # Clear stale tries for disappeared objects
        current_obj_indices = set(range(len(self.current_objects)))
        self.stale_tries = {idx: tries for idx, tries in self.stale_tries.items() 
                           if idx in current_obj_indices}
    
    def record_stale_click(self, obj_idx: int):
        """Record a stale click (no change) for an object."""
        if obj_idx not in self.stale_tries:
            self.stale_tries[obj_idx] = 0
        self.stale_tries[obj_idx] += 1
        
        if os.getenv('RECON_DEBUG'):
            print(f"  📈 Recorded stale click for object_{obj_idx}: {self.stale_tries[obj_idx]} tries")
    
    def record_successful_object_click(self, obj_idx: int):
        """Record a successful click (change detected) for an object."""
        if obj_idx in self.stale_tries:
            self.stale_tries[obj_idx] = 0
            if os.getenv('RECON_DEBUG'):
                print(f"  ✅ Reset stale tries for object_{obj_idx} (successful change)")
    
    def get_stale_penalty(self, obj_idx: int) -> float:
        """Get stale penalty for an object based on failed attempts."""
        stale_count = self.stale_tries.get(obj_idx, 0)
        penalty = self.stale_penalty_lambda * stale_count
        return penalty
    
    def clear_cnn_cache_if_stale(self, obj_idx: int):
        """Clear CNN cache if object has stale clicks to enable exploration."""
        if self.stale_tries.get(obj_idx, 0) > 0:
            if hasattr(self.cnn_terminal, 'clear_cache'):
                self.cnn_terminal.clear_cache()
                if os.getenv('RECON_DEBUG'):
                    print(f"  🔄 Cleared CNN cache due to stale clicks on object_{obj_idx}")
    
    def _save_debug_visualization(self, frame_tensor: torch.Tensor, coords: Tuple[int, int], action_count: int):
        """Save debug visualization showing object masks and selected coordinate"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Convert tensor to numpy
            if frame_tensor.dim() == 3:  # (16, 64, 64) one-hot
                frame_np = frame_tensor.argmax(dim=0).cpu().numpy()
            else:  # (64, 64)
                frame_np = frame_tensor.cpu().numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Show frame
            im = ax.imshow(frame_np, cmap='tab20', vmin=0, vmax=15)
            
            # Overlay object masks with transparency
            for obj_idx, obj in enumerate(self.current_objects[:5]):  # Show first 5
                slc = obj["slice"]
                mask = obj["mask"]
                
                # Create overlay for this object
                overlay = np.zeros_like(frame_np, dtype=float)
                overlay[slc[0].start:slc[0].stop, slc[1].start:slc[1].stop][mask] = 0.3
                
                # Use different colors for different objects
                colors = ['red', 'blue', 'green', 'yellow', 'purple']
                color = colors[obj_idx % len(colors)]
                ax.imshow(overlay, alpha=0.3, cmap='Reds' if color == 'red' else 
                         'Blues' if color == 'blue' else 'Greens')
                
                # Add object label
                ax.text(slc[1].start, slc[0].start - 2, f'obj_{obj_idx}', 
                       color=color, fontweight='bold', fontsize=10)
            
            # Add crosshair at selected coordinate
            y, x = coords
            y, x = int(y), int(x)
            
            # Bold crosshair
            ax.axhline(y=y, color='white', linewidth=4, alpha=0.8)
            ax.axvline(x=x, color='white', linewidth=4, alpha=0.8)
            ax.axhline(y=y, color='black', linewidth=2, alpha=0.9)
            ax.axvline(x=x, color='black', linewidth=2, alpha=0.9)
            
            # Add coordinate text
            ax.text(x+2, y-2, f'({x},{y})', color='white', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Set title
            ax.set_title(f'Improved ReCoN ARC Angel - Action {action_count}\n'
                        f'Selected: action_click at ({x},{y}) with mask validation', 
                        fontsize=14, fontweight='bold')
            
            # Save
            debug_dir = '/workspace/recon_debug_frames'
            os.makedirs(debug_dir, exist_ok=True)
            filename = f'{debug_dir}/improved_frame_{action_count:04d}_coord_{x}_{y}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📸 Saved improved debug frame: {filename}")
            
        except Exception as e:
            print(f"Error saving debug visualization: {e}")
    
    # Compatibility methods for existing interface
    def build_structure(self):
        """Compatibility method - redirect to improved structure"""
        return self.build_improved_structure()
    
    def update_weights_from_frame(self, frame: torch.Tensor):
        """Compatibility method"""
        return self.update_weights_from_cnn_improved(frame)
    
    def get_best_action(self, available_actions: Optional[List[str]] = None):
        """Compatibility method"""
        action, coords, obj_idx = self.get_best_action_with_improved_scoring(available_actions)
        return action, coords  # Return only action and coords for backward compatibility
    
    def apply_availability_mask(self, available_actions: List[str]):
        """Apply availability mask to script nodes"""
        action_mapping = {
            "ACTION1": "action_1", "ACTION2": "action_2", "ACTION3": "action_3",
            "ACTION4": "action_4", "ACTION5": "action_5", "ACTION6": "action_click"
        }
        
        # Set link weights to near-zero for unavailable actions (pure ReCoN approach)
        for harness_action, node_id in action_mapping.items():
            if harness_action not in available_actions:
                # Set sub link weight from root to unavailable action to near zero
                for link in self.graph.get_links(source="frame_change_hypothesis", target=node_id):
                    if link.type == "sub":
                        link.weight = 1e-6  # Near zero but not exactly zero
                
                # For ACTION6, also set subtree weights to near zero
                if harness_action == "ACTION6":
                    # Set click_cnn and click_objects weights to near zero
                    for subtree_node in ["click_cnn", "click_objects"]:
                        for link in self.graph.get_links(source="action_click", target=subtree_node):
                            if link.type == "sub":
                                link.weight = 1e-6
                    
                    # Set terminal weights to near zero
                    for terminal_node in ["cnn_terminal", "resnet_terminal"]:
                        for parent_node in ["click_cnn", "click_objects"]:
                            for link in self.graph.get_links(source=parent_node, target=terminal_node):
                                if link.type == "sub":
                                    link.weight = 1e-6
                    
                    # Set object weights to near zero
                    for obj_idx in range(len(self.current_objects)):
                        object_id = f"object_{obj_idx}"
                        for link in self.graph.get_links(source="click_objects", target=object_id):
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
        """Get statistics about the improved hierarchy"""
        if not self._built:
            return {"built": False}
        
        return {
            "built": True,
            "total_nodes": len(self.graph.nodes),
            "total_links": len(self.graph.links),
            "current_objects": len(self.current_objects),
            "cnn_threshold": self.cnn_threshold,
            "max_objects": self.max_objects,
            "step_count": self.graph.step_count,
            "stickiness_strength": self.stickiness_strength,
            "last_click": self.last_click,
            "sticky_attempts": self.sticky_attempts,
            "tau_ratio": self.tau_ratio,
            "tau_pixels": self.tau_pixels,
            "p_min": self.p_min,
            "max_sticky_attempts": self.max_sticky_attempts,
            "stale_tries": dict(self.stale_tries),
            "stale_penalty_lambda": self.stale_penalty_lambda,
            "improvement": "Mask-aware CNN, background suppression, stickiness, proper ReCoN sequences"
        }
