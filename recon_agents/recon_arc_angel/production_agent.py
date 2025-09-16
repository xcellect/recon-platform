"""
Production ReCoN ARC Angel Agent

Uses the EfficientHierarchicalHypothesisManager with dual training.
This is the MAIN production agent that combines:
- StochasticGoose (1st place): CNN training + action prediction
- BlindSquirrel (2nd place): Object segmentation + ResNet training  
- REFINED_PLAN: Script nodes + pure ReCoN execution
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Any
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, "/workspace/recon-platform")

from .efficient_hierarchy_manager import EfficientHierarchicalHypothesisManager
from .learning_manager import LearningManager


class ProductionReCoNArcAngel:
    """
    Production ReCoN ARC Angel agent using EfficientHierarchicalHypothesisManager.
    
    This is the MAIN agent that combines all the winning approaches:
    - REFINED_PLAN compliance (script nodes, user thresholds, pure ReCoN)
    - StochasticGoose integration (CNN training, frame change prediction)
    - BlindSquirrel integration (object segmentation, ResNet training)
    - GPU acceleration (RTX A4500 optimization)
    - Dual training (CNN every N actions, ResNet on score increase)
    """
    
    def __init__(self, game_id: str = "default", cnn_threshold: float = 0.1, max_objects: int = 50):
        """
        Initialize the production ReCoN ARC Angel agent.
        
        Args:
            game_id: Identifier for the game instance
            cnn_threshold: User-definable threshold for CNN confidence usage
            max_objects: Maximum objects to track per frame (BlindSquirrel limit)
        """
        self.game_id = game_id
        self.cnn_threshold = cnn_threshold
        self.max_objects = max_objects
        
        # 🚀 Core components - efficient hierarchy with dual training
        self.hypothesis_manager = EfficientHierarchicalHypothesisManager(
            cnn_threshold=cnn_threshold,
            max_objects=max_objects
        )
        self.hypothesis_manager.build_structure()
        
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
        self.action_count = 0
        self.current_score = -1
        
        # Statistics
        self.stats = {
            'total_actions': 0,
            'score_resets': 0,
            'cnn_training_steps': 0,
            'resnet_training_steps': 0,
            'unique_experiences': 0,
            'objects_detected': 0
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
                game_action.reasoning = f"Production ReCoN ARC Angel ACTION6 at ({x}, {y}) via object segmentation"
            else:
                game_action.reasoning = f"Production ReCoN ARC Angel {action_type.upper()}"
            
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
                game_action.reasoning = f"Production ReCoN ARC Angel ACTION6 at ({x}, {y})"
                game_action.action_type = "ACTION6"
            else:
                action_num = action_type.split('_')[1] if '_' in action_type else '1'
                game_action.action_type = f"ACTION{action_num}"
                game_action.reasoning = f"Production ReCoN ARC Angel {action_type.upper()}"
            
            return game_action
    
    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """
        Main action selection method implementing the complete production workflow.
        
        Combines:
        - StochasticGoose CNN training and action prediction
        - BlindSquirrel object segmentation and ResNet training
        - REFINED_PLAN pure ReCoN execution with script nodes
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
            
            try:
                from agents.structs import GameAction
                action = GameAction.RESET
                action.reasoning = "Production ReCoN ARC Angel game reset"
                return action
            except ImportError:
                class MockReset:
                    def __init__(self):
                        self.name = "RESET"
                        self.action_type = "RESET"
                        self.reasoning = "Production ReCoN ARC Angel game reset"
                return MockReset()
        
        # Convert frame to tensor with GPU acceleration
        try:
            current_frame_tensor = self._convert_frame_to_tensor(latest_frame)
        except Exception as e:
            print(f"Production ReCoN ARC Angel: Error converting frame: {e}")
            return self._convert_to_game_action("action_1")
        
        # Add experience and state transition for dual training
        if (self.prev_frame_tensor is not None and 
            self.prev_action_type is not None and 
            self.action_count > 0):
            
            try:
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
                print(f"Production ReCoN ARC Angel: Error in dual training: {e}")
        
        # 🔍 Object segmentation + CNN inference (BlindSquirrel + StochasticGoose)
        try:
            self.hypothesis_manager.update_weights_from_cnn(current_frame_tensor)
            
            # Track objects detected
            stats = self.hypothesis_manager.get_stats()
            self.stats['objects_detected'] = stats.get('current_objects', 0)
            
        except Exception as e:
            print(f"Production ReCoN ARC Angel: Error updating weights: {e}")
        
        # 🎯 ReCoN execution (REFINED_PLAN compliance)
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
        
        for _ in range(8):  # Sufficient steps for script→terminal→script flow
            self.hypothesis_manager.propagate_step()
        
        # 🎯 Action selection with object-based coordinates
        best_action, best_coords = self.hypothesis_manager.get_best_action_with_object_coordinates(
            available_actions=latest_frame.available_actions if hasattr(latest_frame, 'available_actions') else None
        )
        
        if best_action is None:
            # Fallback to first available action
            if hasattr(latest_frame, 'available_actions') and latest_frame.available_actions:
                first_available = latest_frame.available_actions[0]
                if hasattr(first_available, 'name'):
                    action_name = first_available.name.lower()
                    if 'action' in action_name:
                        best_action = action_name.replace('action', 'action_')
                    else:
                        best_action = "action_1"
                else:
                    best_action = "action_1"
            else:
                best_action = "action_1"
        
        # Convert to game action
        try:
            game_action = self._convert_to_game_action(best_action, best_coords)
        except Exception as e:
            print(f"Production ReCoN ARC Angel: Error converting action: {e}")
            game_action = self._convert_to_game_action("action_1")
        
        # Store state for next iteration
        self.prev_frame_tensor = current_frame_tensor
        self.prev_action_type = best_action
        self.prev_coords = best_coords
        
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
                print(f"Production ReCoN ARC Angel: Error in CNN training: {e}")
        
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
            'hypothesis_manager': self.hypothesis_manager.get_stats(),
            'learning_manager': self.learning_manager.get_stats()
        }
        return stats
    
    def reset(self):
        """Reset agent state"""
        self.prev_frame_tensor = None
        self.prev_action_type = None
        self.prev_coords = None
        self.action_count = 0
        self.current_score = -1
        
        self.hypothesis_manager.reset()
        self.learning_manager.clear()
        
        self.stats = {
            'total_actions': 0,
            'score_resets': 0,
            'cnn_training_steps': 0,
            'resnet_training_steps': 0,
            'unique_experiences': 0,
            'objects_detected': 0
        }
