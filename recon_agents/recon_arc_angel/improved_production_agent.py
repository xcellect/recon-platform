"""
Improved Production ReCoN ARC Angel Agent

Uses the ImprovedHierarchicalHypothesisManager with all the enhancements:
1. Proper ReCoN graph structure with por/ret sequences for ACTION6
2. Mask-aware CNN coupling using masked max instead of bounding-box max
3. Background suppression with area fraction and border penalties
4. Improved selection scoring with comprehensive object evaluation
5. Stickiness mechanism for successful clicks that cause frame changes
6. Pure ReCoN execution semantics

This is the MAIN production agent that fixes the coordinate selection issues.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Any
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, "/workspace/recon-platform")

try:
    # Try module-qualified imports first (for harness)
    from recon_agents.recon_arc_angel.improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager
    from recon_agents.recon_arc_angel.learning_manager import LearningManager
except ImportError:
    # Fallback to local imports (for development/testing)
    from improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager
    from learning_manager import LearningManager


class ImprovedProductionReCoNArcAngel:
    """
    Improved Production ReCoN ARC Angel agent using ImprovedHierarchicalHypothesisManager.
    
    This is the MAIN improved agent that fixes all the coordinate selection issues:
    - Proper ReCoN sequences with por/ret constraints
    - Mask-aware CNN coupling (no more out-of-object clicks)
    - Background suppression (no more clicking on borders/strips)
    - Comprehensive object scoring (regularity + CNN + penalties)
    - Stickiness for frame changes (persistence after successful clicks)
    - GPU acceleration and dual training
    """
    
    def __init__(self, game_id: str = "default", cnn_threshold: float = 0.1, max_objects: int = 50):
        """
        Initialize the improved production ReCoN ARC Angel agent.
        
        Args:
            game_id: Identifier for the game instance
            cnn_threshold: User-definable threshold for CNN confidence usage
            max_objects: Maximum objects to track per frame (BlindSquirrel limit)
        """
        self.game_id = game_id
        self.cnn_threshold = cnn_threshold
        self.max_objects = max_objects
        
        # 🚀 Core components - improved hierarchy with all enhancements
        self.hypothesis_manager = ImprovedHierarchicalHypothesisManager(
            cnn_threshold=cnn_threshold,
            max_objects=max_objects
        )
        self.hypothesis_manager.build_improved_structure()
        
        self.learning_manager = LearningManager(
            buffer_size=200000,
            batch_size=64,
            train_frequency=5,
            learning_rate=0.0001
        )
        
        # Set up dual training system
        self.learning_manager.set_cnn_terminal(self.hypothesis_manager.cnn_terminal)
        self.learning_manager.set_resnet_terminal(self.hypothesis_manager.resnet_terminal)
        
        # State tracking
        self.prev_frame_tensor: Optional[torch.Tensor] = None
        self.prev_action_type: Optional[str] = None
        self.prev_coords: Optional[Tuple[int, int]] = None
        self.prev_obj_idx: Optional[int] = None
        self.action_count = 0
        self.current_score = -1
        self.prev_frame_hash = None
        
        # Statistics
        self.stats = {
            'total_actions': 0,
            'score_resets': 0,
            'cnn_training_steps': 0,
            'resnet_training_steps': 0,
            'unique_experiences': 0,
            'objects_detected': 0,
            'successful_clicks': 0,
            'frame_changes_detected': 0,
            'sticky_selections': 0,
            'action6_coord_none_prevented': 0
        }
    
    def _convert_frame_to_tensor(self, frame_data: Any) -> torch.Tensor:
        """Convert frame data to tensor format for neural processing"""
        frame = np.array(frame_data.frame, dtype=np.int64)
        
        # Handle animation frames (take last frame)
        if len(frame.shape) == 3:
            frame = frame[-1]
        
        if frame.shape != (64, 64):
            raise ValueError(f"Expected frame shape (64, 64), got {frame.shape}")
        
        # One-hot encode: (64, 64) -> (16, 64, 64)
        tensor = torch.zeros(16, 64, 64, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        return tensor
    
    def _log_coordinate_heatmap(self, frame_tensor: torch.Tensor):
        """Log coordinate probability heatmap for debugging CNN exploration."""
        try:
            # Get CNN coordinate probabilities
            measurement = self.hypothesis_manager.cnn_terminal.measure(frame_tensor)
            result = self.hypothesis_manager.cnn_terminal._process_measurement(measurement)
            coord_probs = result["coordinate_probabilities"]
            
            # Find argmax coordinate
            flat_probs = coord_probs.flatten()
            argmax_idx = torch.argmax(flat_probs).item()
            argmax_y = argmax_idx // 64
            argmax_x = argmax_idx % 64
            max_prob = flat_probs[argmax_idx].item()
            
            # Calculate entropy for exploration measure
            entropy = -(coord_probs * torch.log(coord_probs + 1e-8)).sum().item()
            
            print(f"  📊 Coordinate Heatmap Analysis:")
            print(f"    Argmax: ({argmax_x}, {argmax_y}) with prob={max_prob:.4f}")
            print(f"    Entropy: {entropy:.3f} (higher = more exploration)")
            print(f"    Temperature: {getattr(self.hypothesis_manager.cnn_terminal, 'coord_temp', 'N/A')}")
            
            # Log top-5 coordinates for diversity check
            top_5_indices = torch.topk(flat_probs, 5).indices
            top_5_coords = [(idx.item() // 64, idx.item() % 64) for idx in top_5_indices]
            top_5_probs = [flat_probs[idx].item() for idx in top_5_indices]
            
            print(f"    Top-5 coords: {list(zip(top_5_coords, [f'{p:.4f}' for p in top_5_probs]))}")
            
        except Exception as e:
            print(f"  ❌ Error in coordinate heatmap logging: {e}")
    
    
    def _convert_to_game_action(self, action_type: str, coords: Optional[Tuple[int, int]] = None) -> Any:
        """Convert internal action to GameAction for harness"""
        try:
            from agents.structs import GameAction
            
            action_mapping = {
                "action_1": GameAction.ACTION1,
                "action_2": GameAction.ACTION2,
                "action_3": GameAction.ACTION3,
                "action_4": GameAction.ACTION4,
                "action_5": GameAction.ACTION5,
                "action_click": GameAction.ACTION6
            }
            
            if action_type not in action_mapping:
                raise ValueError(f"Unknown action type: {action_type}")
            
            game_action = action_mapping[action_type]
            
            if action_type == "action_click" and coords is not None:
                y, x = coords
                game_action.set_data({"x": x, "y": y})
                game_action.reasoning = f"Improved ReCoN ARC Angel ACTION6 at ({x}, {y}) - mask-aware selection with background suppression"
            else:
                game_action.reasoning = f"Improved ReCoN ARC Angel {action_type.upper()}"
            
            return game_action
            
        except ImportError:
            # Fallback for testing
            class MockGameAction:
                def __init__(self, name):
                    self.name = name
                    self.action_type = name.upper()
                    self.reasoning = ""
                    self.data = {}
                def set_data(self, data):
                    self.data = data
            
            game_action = MockGameAction(action_type)
            
            if action_type == "action_click" and coords is not None:
                y, x = coords
                game_action.set_data({"x": x, "y": y})
                game_action.reasoning = f"Improved ReCoN ARC Angel ACTION6 at ({x}, {y})"
                game_action.action_type = "ACTION6"
            else:
                action_num = action_type.split('_')[1] if '_' in action_type else '1'
                game_action.action_type = f"ACTION{action_num}"
                game_action.reasoning = f"Improved ReCoN ARC Angel {action_type.upper()}"
            
            return game_action
    
    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """
        Main action selection method implementing the complete improved workflow.
        
        Implements all the fixes from the agent's suggestions:
        - Proper ReCoN sequences with por/ret constraints
        - Mask-aware CNN coupling
        - Background suppression
        - Comprehensive object scoring
        - Stickiness mechanism
        """
        
        # Handle score changes (triggers ResNet training)
        if hasattr(latest_frame, 'score') and latest_frame.score != self.current_score:
            score_increased = self.learning_manager.on_score_change(latest_frame.score, self.game_id)
            if score_increased:
                self.hypothesis_manager.reset()
                self.stats['score_resets'] += 1
                if latest_frame.score > 0:
                    self.stats['resnet_training_steps'] += 1
            self.current_score = latest_frame.score
        
        # Handle game reset states
        if hasattr(latest_frame, 'state') and latest_frame.state in ["NOT_PLAYED", "GAME_OVER"]:
            self.prev_frame_tensor = None
            self.prev_action_type = None
            self.prev_coords = None
            self.prev_obj_idx = None
            self.prev_frame_hash = None
            
            try:
                from agents.structs import GameAction
                action = GameAction.RESET
                action.reasoning = "Improved ReCoN ARC Angel game reset"
                return action
            except ImportError:
                class MockReset:
                    def __init__(self):
                        self.name = "RESET"
                        self.action_type = "RESET"
                        self.reasoning = "Improved ReCoN ARC Angel game reset"
                return MockReset()
        
        # Convert frame to tensor with GPU acceleration
        try:
            current_frame_tensor = self._convert_frame_to_tensor(latest_frame)
        except Exception as e:
            print(f"Improved ReCoN ARC Angel: Error converting frame: {e}")
            return self._convert_to_game_action("action_1")
        
        # Object-scoped stickiness management (replaced global frame change detection)
        if (self.prev_action_type == "action_click" and 
            self.prev_coords is not None and
            hasattr(self, 'prev_obj_idx') and self.prev_obj_idx is not None):
            
            # Get CNN probability for the previously clicked object to check if it still exists
            prev_obj_cnn_prob = 0.0
            if self.prev_obj_idx < len(self.hypothesis_manager.current_objects):
                prev_obj_id = f"object_{self.prev_obj_idx}"
                for link in self.hypothesis_manager.graph.get_links(source="click_objects", target=prev_obj_id):
                    if link.type == "sub":
                        prev_obj_cnn_prob = float(link.weight)
                        break
            
            # Update stickiness with object-scoped change detection
            action6_available = hasattr(latest_frame, 'available_actions') and any(
                getattr(a, 'name', str(a)) == 'ACTION6' for a in latest_frame.available_actions
            )
            
            # Check for object-scoped changes and update stale tracking
            change_detected = self.hypothesis_manager.detect_object_scoped_change(current_frame_tensor)
            
            if change_detected:
                # Successful click - reset stale tries for this object
                self.hypothesis_manager.record_successful_object_click(self.prev_obj_idx)
                self.stats['frame_changes_detected'] += 1
            else:
                # Stale click - increment stale tries and clear CNN cache
                self.hypothesis_manager.record_stale_click(self.prev_obj_idx)
                self.hypothesis_manager.clear_cnn_cache_on_stale_click()
            
            self.hypothesis_manager.update_stickiness(
                current_frame_tensor, 
                prev_obj_cnn_prob, 
                action6_available
            )
        
        # Add experience and state transition for dual training (with validation)
        if (self.prev_frame_tensor is not None and 
            self.prev_action_type is not None and 
            self.action_count > 0):
            
            try:
                # Validate coordinates for ACTION6 before training
                valid_for_training = True
                if self.prev_action_type == "action_click" and self.prev_coords is None:
                    valid_for_training = False
                    if os.getenv('RECON_DEBUG'):
                        print(f"  ⚠️  Skipping training for action_click with coords=None")
                
                if valid_for_training:
                    # CNN experience (StochasticGoose style)
                    added = self.learning_manager.add_experience(
                        self.prev_frame_tensor, 
                        current_frame_tensor,
                        self.prev_action_type,
                        self.prev_coords
                    )
                    if added:
                        self.stats['unique_experiences'] += 1
                    
                    # ResNet state transition (BlindSquirrel style)
                    self.learning_manager.add_state_transition(
                        self.prev_frame_tensor,
                        current_frame_tensor, 
                        self.prev_action_type,
                        self.prev_coords,
                        self.game_id,
                        latest_frame.score if hasattr(latest_frame, 'score') else 0
                    )
                
            except Exception as e:
                print(f"Improved ReCoN ARC Angel: Error in dual training: {e}")
        
        # 🔍 Improved object segmentation + mask-aware CNN inference
        try:
            self.hypothesis_manager.update_weights_from_cnn_improved(current_frame_tensor)
            
            # Track objects detected
            stats = self.hypothesis_manager.get_stats()
            self.stats['objects_detected'] = stats.get('current_objects', 0)
            
        except Exception as e:
            print(f"Improved ReCoN ARC Angel: Error updating weights: {e}")
        
        # 🎯 Improved ReCoN execution with proper sequences
        self.hypothesis_manager.reset()
        
        # Apply availability mask
        if hasattr(latest_frame, 'available_actions'):
            available_names = []
            for action in latest_frame.available_actions:
                if hasattr(action, 'name'):
                    available_names.append(action.name)
                elif hasattr(action, 'value'):
                    available_names.append(f"ACTION{action.value}")
                else:
                    available_names.append(str(action))
            
            self.hypothesis_manager.apply_availability_mask(available_names)
        
        # ReCoN propagation with proper por/ret sequences
        self.hypothesis_manager.request_frame_change()
        
        for _ in range(10):  # Sufficient steps for the improved sequence: action_click -> click_cnn -> click_objects
            self.hypothesis_manager.propagate_step()
        
        # 🎯 Improved action selection with comprehensive scoring
        best_action, best_coords, best_obj_idx = self.hypothesis_manager.get_best_action_with_improved_scoring(
            available_actions=available_names if 'available_names' in locals() else None
        )
        
        # CRITICAL FIX: Prevent ACTION6 with coords=None (causes "Invalid action" errors)
        if best_action == "action_click" and best_coords is None:
            # Fall back to first available action
            if hasattr(latest_frame, 'available_actions') and latest_frame.available_actions:
                first_available = latest_frame.available_actions[0]
                if hasattr(first_available, 'name'):
                    action_name = first_available.name.lower()
                    if 'action' in action_name and action_name != 'action6':
                        best_action = action_name.replace('action', 'action_')
                    else:
                        best_action = "action_1"
                else:
                    best_action = "action_1"
            else:
                best_action = "action_1"
            
            best_coords = None
            best_obj_idx = None
            self.stats['action6_coord_none_prevented'] = self.stats.get('action6_coord_none_prevented', 0) + 1
        
        # Track if stickiness influenced the selection and record successful clicks
        if best_action == "action_click" and best_coords is not None and best_obj_idx is not None:
            # Check if stickiness influenced this selection
            if (self.hypothesis_manager.last_click['coords'] is not None and
                self.hypothesis_manager.last_click['obj_idx'] == best_obj_idx):
                self.stats['sticky_selections'] += 1
            
            # Store object info for next iteration's stickiness detection
            self.prev_obj_idx = best_obj_idx
            
            # If this is a different action from last time, record potential success
            # (actual success will be determined by object-scoped change detection next frame)
            if (self.prev_action_type != "action_click" or 
                self.prev_coords != best_coords or
                getattr(self, 'prev_obj_idx', None) != best_obj_idx):
                
                # Create full-frame mask for the selected object
                full_mask = self.hypothesis_manager._create_full_frame_mask(best_obj_idx)
                if full_mask is not None:
                    self.hypothesis_manager.record_successful_click(
                        best_coords, best_obj_idx, current_frame_tensor, full_mask
                    )
                    self.stats['successful_clicks'] += 1
        
        if best_action is None:
            # Fallback to first available action
            if hasattr(latest_frame, 'available_actions') and latest_frame.available_actions:
                first_available = latest_frame.available_actions[0]
                if hasattr(first_available, 'name'):
                    action_name = first_available.name.lower()
                    if 'action' in action_name:
                        if action_name == 'action6':
                            best_action = "action_click"
                        else:
                            best_action = action_name.replace('action', 'action_')
                    else:
                        best_action = "action_1"
                else:
                    best_action = "action_1"
            else:
                best_action = "action_1"
        
        # Debug log for verification
        if os.getenv('RECON_DEBUG'):
            print(f"🎯 Improved Production Agent Action Selection:")
            print(f"  Available names: {available_names if 'available_names' in locals() else 'N/A'}")
            print(f"  Selected action: {best_action}")
            print(f"  Selected coords: {best_coords}")
            print(f"  Frame score: {latest_frame.score if hasattr(latest_frame, 'score') else 'N/A'}")
            print(f"  Action count: {self.action_count}")
            print(f"  Object-scoped stickiness: {self.hypothesis_manager.stickiness_strength:.3f}")
            print(f"  Objects detected: {self.stats['objects_detected']}")
            
            # Add coordinate probability heatmap logging
            self._log_coordinate_heatmap(current_frame_tensor)
            
            # Visual debug: save frame with improved visualization
            if best_action == "action_click" and best_coords is not None:
                self.hypothesis_manager._save_debug_visualization(
                    current_frame_tensor, best_coords, self.action_count)
        
        # Convert to game action
        try:
            game_action = self._convert_to_game_action(best_action, best_coords)
        except Exception as e:
            print(f"Improved ReCoN ARC Angel: Error converting action: {e}")
            game_action = self._convert_to_game_action("action_1")
        
        # Store state for next iteration
        self.prev_frame_tensor = current_frame_tensor
        self.prev_action_type = best_action
        self.prev_coords = best_coords
        self.prev_obj_idx = best_obj_idx
        
        # Update counters and perform dual training
        self.action_count += 1
        self.stats['total_actions'] += 1
        self.learning_manager.step()
        
        # 🎓 CNN training (StochasticGoose style - every N actions)
        if self.learning_manager.should_train():
            try:
                metrics = self.learning_manager.train_step()
                if metrics:
                    self.stats['cnn_training_steps'] += 1
            except Exception as e:
                print(f"Improved ReCoN ARC Angel: Error in CNN training: {e}")
        
        return game_action
    
    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Check if agent is done playing"""
        if hasattr(latest_frame, 'state'):
            return latest_frame.state in ["WIN", "GAME_OVER"]
        return False
    
    def get_stats(self) -> dict:
        """Get comprehensive agent statistics"""
        stats = {
            **self.stats,
            'current_score': self.current_score,
            'action_count': self.action_count,
            'cnn_threshold': self.cnn_threshold,
            'max_objects': self.max_objects,
            'improvements': [
                'Proper ReCoN por/ret sequences',
                'Mask-aware CNN coupling',
                'Background suppression',
                'Comprehensive object scoring',
                'Stickiness mechanism',
                'Pure ReCoN execution'
            ],
            'hypothesis_manager': self.hypothesis_manager.get_stats(),
            'learning_manager': self.learning_manager.get_stats()
        }
        return stats
    
    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """
        Check if agent is done playing.
        
        Args:
            frames: List of previous frames
            latest_frame: Current frame data
            
        Returns:
            True only if game is actually won (not on GAME_OVER)
        """
        if hasattr(latest_frame, 'state'):
            # Only stop on WIN, continue playing even on GAME_OVER until MAX_ACTIONS
            return latest_frame.state == "WIN"
        return False
    
    def reset(self):
        """Reset agent state"""
        self.prev_frame_tensor = None
        self.prev_action_type = None
        self.prev_coords = None
        self.prev_obj_idx = None
        self.action_count = 0
        self.current_score = -1
        self.prev_frame_hash = None
        
        self.hypothesis_manager.reset()
        self.learning_manager.clear()
        
        self.stats = {
            'total_actions': 0,
            'score_resets': 0,
            'cnn_training_steps': 0,
            'resnet_training_steps': 0,
            'unique_experiences': 0,
            'objects_detected': 0,
            'successful_clicks': 0,
            'frame_changes_detected': 0,
            'sticky_selections': 0
        }
