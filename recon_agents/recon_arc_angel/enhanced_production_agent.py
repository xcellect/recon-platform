"""
Enhanced Production ReCoN ARC Angel Agent

Implements the complete systematic solution to prevent getting stuck:
1. Stable object identity across frames with IoU-based matching
2. ReCoN hypothesis system with systematic reduction (test-and-prune)
3. Causal object-scoped verification (clicked object only)
4. Deterministic evidence-led coordinate selection
5. Fixed CNN probability scaling and gating
6. Principled exploration with hypothesis scheduling
7. Never emits ACTION6 without valid coordinates
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
    from recon_agents.recon_arc_angel.enhanced_hierarchy_manager import EnhancedHierarchicalHypothesisManager
    from recon_agents.recon_arc_angel.learning_manager import LearningManager
    from recon_agents.recon_arc_angel.stable_object_tracker import StableObjectTracker
except ImportError:
    # Fallback to local imports (for development/testing)
    from enhanced_hierarchy_manager import EnhancedHierarchicalHypothesisManager
    from learning_manager import LearningManager
    from stable_object_tracker import StableObjectTracker


class EnhancedProductionReCoNArcAngel:
    """
    Enhanced Production ReCoN ARC Angel agent with systematic hypothesis reduction.
    
    This agent implements the complete solution to prevent getting stuck:
    - Stable object identity prevents bookkeeping errors
    - Causal object-scoped verification (clicked object only)
    - Systematic hypothesis reduction (test-and-prune)
    - Deterministic coordinate selection
    - Fixed CNN probability scaling
    - Never emits ACTION6 without valid coordinates
    """
    
    def __init__(self, game_id: str = "default", cnn_threshold: float = 0.1, max_objects: int = 50):
        """
        Initialize the enhanced production agent.
        
        Args:
            game_id: Identifier for the game instance
            cnn_threshold: User-definable threshold for CNN confidence usage
            max_objects: Maximum objects to track per frame
        """
        self.game_id = game_id
        self.cnn_threshold = cnn_threshold
        self.max_objects = max_objects
        
        # Enhanced hierarchy with stable object tracking
        self.hypothesis_manager = EnhancedHierarchicalHypothesisManager(
            cnn_threshold=cnn_threshold,
            max_objects=max_objects
        )
        self.hypothesis_manager.build_enhanced_structure()
        
        self.learning_manager = LearningManager(
            buffer_size=200000,
            batch_size=64,
            train_frequency=5,
            learning_rate=0.0001
        )
        
        # Set up dual training system
        self.learning_manager.set_cnn_terminal(self.hypothesis_manager.cnn_terminal)
        self.learning_manager.set_resnet_terminal(self.hypothesis_manager.resnet_terminal)
        
        # State tracking for causal verification
        self.prev_frame_tensor: Optional[torch.Tensor] = None
        self.prev_action_type: Optional[str] = None
        self.prev_coords: Optional[Tuple[int, int]] = None
        self.prev_object_id: Optional[str] = None  # Stable persistent ID
        self.action_count = 0
        self.current_score = -1
        
        # Statistics
        self.stats = {
            'total_actions': 0,
            'score_resets': 0,
            'cnn_training_steps': 0,
            'resnet_training_steps': 0,
            'unique_experiences': 0,
            'causal_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'hypothesis_confirmations': 0,
            'hypothesis_failures': 0,
            'coordinate_scaling_fixes': 0,
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
    
    def _convert_to_game_action(self, action_type: str, coords: Optional[Tuple[int, int]] = None) -> Any:
        """Convert internal action to GameAction for harness with safety checks."""
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
            
            if action_type == "action_click":
                if coords is not None:
                    y, x = coords
                    game_action.set_data({"x": x, "y": y})
                    game_action.reasoning = f"Enhanced ReCoN ARC Angel ACTION6 at ({x}, {y}) - systematic hypothesis reduction"
                else:
                    # CRITICAL FIX: Never emit ACTION6 without coordinates
                    # Fall back to ACTION1 instead
                    self.stats['action6_coord_none_prevented'] += 1
                    game_action = GameAction.ACTION1
                    game_action.reasoning = "Enhanced ReCoN ARC Angel fallback to ACTION1 (prevented ACTION6 with coords=None)"
            else:
                game_action.reasoning = f"Enhanced ReCoN ARC Angel {action_type.upper()}"
            
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
            
            if action_type == "action_click":
                if coords is not None:
                    y, x = coords
                    game_action.set_data({"x": x, "y": y})
                    game_action.reasoning = f"Enhanced ReCoN ARC Angel ACTION6 at ({x}, {y})"
                    game_action.action_type = "ACTION6"
                else:
                    # Fallback to ACTION1
                    game_action.action_type = "ACTION1"
                    game_action.reasoning = "Enhanced ReCoN ARC Angel fallback (prevented ACTION6 with coords=None)"
                    self.stats['action6_coord_none_prevented'] += 1
            else:
                action_num = action_type.split('_')[1] if '_' in action_type else '1'
                game_action.action_type = f"ACTION{action_num}"
                game_action.reasoning = f"Enhanced ReCoN ARC Angel {action_type.upper()}"
            
            return game_action
    
    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """
        Main action selection method with systematic hypothesis reduction.
        
        Implements the complete solution:
        - Stable object identity across frames
        - Causal object-scoped verification
        - Systematic hypothesis reduction
        - Deterministic coordinate selection
        - Fixed CNN probability scaling
        """
        
        # Handle score changes (triggers ResNet training)
        if hasattr(latest_frame, 'score') and latest_frame.score != self.current_score:
            score_increased = self.learning_manager.on_score_change(latest_frame.score, self.game_id)
            if score_increased:
                self.hypothesis_manager.reset()
                # Reset object tracker on new level
                self.hypothesis_manager.object_tracker = StableObjectTracker()
                self.stats['score_resets'] += 1
                if latest_frame.score > 0:
                    self.stats['resnet_training_steps'] += 1
            self.current_score = latest_frame.score
        
        # Handle game reset states
        if hasattr(latest_frame, 'state') and latest_frame.state in ["NOT_PLAYED", "GAME_OVER"]:
            self.prev_frame_tensor = None
            self.prev_action_type = None
            self.prev_coords = None
            self.prev_object_id = None
            
            try:
                from agents.structs import GameAction
                action = GameAction.RESET
                action.reasoning = "Enhanced ReCoN ARC Angel game reset"
                return action
            except ImportError:
                class MockReset:
                    def __init__(self):
                        self.name = "RESET"
                        self.action_type = "RESET"
                        self.reasoning = "Enhanced ReCoN ARC Angel game reset"
                return MockReset()
        
        # Convert frame to tensor
        try:
            current_frame_tensor = self._convert_frame_to_tensor(latest_frame)
        except Exception as e:
            print(f"Enhanced ReCoN ARC Angel: Error converting frame: {e}")
            return self._convert_to_game_action("action_1")
        
        # Causal object-scoped verification (only for clicked object)
        if (self.prev_frame_tensor is not None and 
            self.prev_action_type == "action_click" and 
            self.prev_coords is not None):
            
            self.stats['causal_verifications'] += 1
            
            # Verify if the clicked object caused the change
            success = self.hypothesis_manager.verify_clicked_object(
                self.prev_coords,
                self.prev_frame_tensor,
                current_frame_tensor
            )
            
            if success:
                self.stats['successful_verifications'] += 1
            else:
                self.stats['failed_verifications'] += 1
        
        # Add experience for dual training
        if (self.prev_frame_tensor is not None and 
            self.prev_action_type is not None and 
            self.action_count > 0):
            
            try:
                added = self.learning_manager.add_experience(
                    self.prev_frame_tensor, 
                    current_frame_tensor,
                    self.prev_action_type,
                    self.prev_coords
                )
                if added:
                    self.stats['unique_experiences'] += 1
                
                self.learning_manager.add_state_transition(
                    self.prev_frame_tensor,
                    current_frame_tensor, 
                    self.prev_action_type,
                    self.prev_coords,
                    self.game_id,
                    latest_frame.score if hasattr(latest_frame, 'score') else 0
                )
                
            except Exception as e:
                print(f"Enhanced ReCoN ARC Angel: Error in dual training: {e}")
        
        # Enhanced object segmentation with stable identity
        try:
            self.hypothesis_manager.update_weights_from_cnn_enhanced(current_frame_tensor)
            
            # Track statistics
            tracker_stats = self.hypothesis_manager.object_tracker.get_stats()
            self.stats['persistent_objects'] = tracker_stats['total_persistent_objects']
            self.stats['hypothesis_confirmations'] = tracker_stats['status_counts'].get('CONFIRMED', 0)
            self.stats['hypothesis_failures'] = tracker_stats['status_counts'].get('FAILED', 0)
            
        except Exception as e:
            print(f"Enhanced ReCoN ARC Angel: Error updating weights: {e}")
        
        # ReCoN execution with hypothesis system
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
        
        # ReCoN propagation
        self.hypothesis_manager.request_frame_change()
        
        for _ in range(12):  # More steps for hypothesis system
            self.hypothesis_manager.propagate_step()
        
        # Systematic hypothesis reduction
        best_action, best_coords, best_object_id = self.hypothesis_manager.get_best_hypothesis_with_systematic_reduction(
            available_actions=available_names if 'available_names' in locals() else None
        )
        
        # CRITICAL: Prevent ACTION6 with coords=None
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
            best_object_id = None
            self.stats['action6_coord_none_prevented'] += 1
        
        if best_action is None:
            # Final fallback
            best_action = "action_1"
            best_coords = None
            best_object_id = None
        
        # Debug logging with health checks
        if os.getenv('RECON_DEBUG'):
            print(f"🎯 Enhanced Production Agent Action Selection:")
            print(f"  Available names: {available_names if 'available_names' in locals() else 'N/A'}")
            print(f"  Selected action: {best_action}")
            print(f"  Selected coords: {best_coords}")
            print(f"  Selected object_id: {best_object_id}")
            print(f"  Frame score: {latest_frame.score if hasattr(latest_frame, 'score') else 'N/A'}")
            print(f"  Action count: {self.action_count}")
            print(f"  Causal verifications: {self.stats['causal_verifications']}")
            print(f"  Successful verifications: {self.stats['successful_verifications']}")
            print(f"  ACTION6 coord=None prevented: {self.stats['action6_coord_none_prevented']}")
            
            # Log coordinate heatmap analysis
            self._log_coordinate_heatmap_analysis(current_frame_tensor, best_coords, best_object_id)
        
        # Convert to game action with safety checks
        try:
            game_action = self._convert_to_game_action(best_action, best_coords)
        except Exception as e:
            print(f"Enhanced ReCoN ARC Angel: Error converting action: {e}")
            game_action = self._convert_to_game_action("action_1")
        
        # Store state for next iteration's causal verification
        self.prev_frame_tensor = current_frame_tensor
        self.prev_action_type = best_action
        self.prev_coords = best_coords
        self.prev_object_id = best_object_id
        
        # Update counters and perform dual training
        self.action_count += 1
        self.stats['total_actions'] += 1
        self.learning_manager.step()
        
        # CNN training
        if self.learning_manager.should_train():
            try:
                metrics = self.learning_manager.train_step()
                if metrics:
                    self.stats['cnn_training_steps'] += 1
            except Exception as e:
                print(f"Enhanced ReCoN ARC Angel: Error in CNN training: {e}")
        
        return game_action
    
    def _log_coordinate_heatmap_analysis(self, frame_tensor: torch.Tensor, 
                                       selected_coords: Optional[Tuple[int, int]], 
                                       selected_object_id: Optional[str]):
        """Log coordinate heatmap analysis for debugging CNN behavior."""
        try:
            # Get CNN coordinate probabilities
            measurement = self.hypothesis_manager.cnn_terminal.measure(frame_tensor)
            result = self.hypothesis_manager.cnn_terminal._process_measurement(measurement)
            coord_probs = result["coordinate_probabilities"]
            
            # Find global argmax
            flat_probs = coord_probs.flatten()
            argmax_idx = torch.argmax(flat_probs).item()
            argmax_y = argmax_idx // 64
            argmax_x = argmax_idx % 64
            max_prob = flat_probs[argmax_idx].item()
            
            # Calculate entropy
            entropy = -(coord_probs * torch.log(coord_probs + 1e-8)).sum().item()
            
            print(f"  📊 Coordinate Heatmap Analysis:")
            print(f"    Global argmax: ({argmax_x}, {argmax_y}) with prob={max_prob:.6f}")
            print(f"    Entropy: {entropy:.3f} (higher = more exploration)")
            print(f"    CNN coord temperature: {getattr(self.hypothesis_manager.cnn_terminal, 'coord_temp', 'N/A')}")
            
            # If we selected coordinates, show if they match CNN argmax
            if selected_coords is not None:
                sel_y, sel_x = selected_coords
                sel_prob = coord_probs[sel_y, sel_x].item()
                print(f"    Selected coord: ({sel_x}, {sel_y}) with prob={sel_prob:.6f}")
                
                if (sel_x, sel_y) == (argmax_x, argmax_y):
                    print(f"    ✅ Selected coord matches CNN argmax (deterministic)")
                else:
                    print(f"    📍 Selected coord differs from argmax (object-constrained)")
            
            # Show object-specific masked probabilities
            if selected_object_id:
                persistent_obj = self.hypothesis_manager.object_tracker.persistent_objects.get(selected_object_id)
                if persistent_obj:
                    scaled_prob = self.hypothesis_manager.calculate_scaled_masked_cnn_probability(coord_probs, persistent_obj)
                    print(f"    Object {selected_object_id} scaled masked prob: {scaled_prob:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error in coordinate heatmap analysis: {e}")
    
    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Check if agent is done playing"""
        if hasattr(latest_frame, 'state'):
            return latest_frame.state in ["WIN", "GAME_OVER"]
        return False
    
    def get_stats(self) -> dict:
        """Get comprehensive agent statistics with hypothesis tracking."""
        stats = {
            **self.stats,
            'current_score': self.current_score,
            'action_count': self.action_count,
            'cnn_threshold': self.cnn_threshold,
            'max_objects': self.max_objects,
            'enhancements': [
                'Stable object identity with IoU matching',
                'ReCoN hypothesis system with reduction',
                'Causal object-scoped verification',
                'Deterministic evidence-led selection',
                'Fixed CNN probability scaling',
                'Principled exploration scheduling',
                'ACTION6 coords=None prevention'
            ],
            'hypothesis_manager': self.hypothesis_manager.get_stats(),
            'learning_manager': self.learning_manager.get_stats()
        }
        return stats
    
    def reset(self):
        """Reset agent state"""
        self.prev_frame_tensor = None
        self.prev_action_type = None
        self.prev_coords = None
        self.prev_object_id = None
        self.action_count = 0
        self.current_score = -1
        
        self.hypothesis_manager.reset()
        self.hypothesis_manager.object_tracker = StableObjectTracker()
        self.learning_manager.clear()
        
        self.stats = {
            'total_actions': 0,
            'score_resets': 0,
            'cnn_training_steps': 0,
            'resnet_training_steps': 0,
            'unique_experiences': 0,
            'causal_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'hypothesis_confirmations': 0,
            'hypothesis_failures': 0,
            'coordinate_scaling_fixes': 0,
            'action6_coord_none_prevented': 0
        }
