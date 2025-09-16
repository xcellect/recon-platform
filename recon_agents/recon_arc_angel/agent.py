"""
ReCoN ARC Angel Agent

Main agent class that implements the StochasticGoose parity approach using ReCoN principles.
Interfaces with the ARC-AGI harness and coordinates all components:
- HypothesisManager for ReCoN graph structure
- LearningManager for experience collection and training  
- Region aggregation for hierarchical coordinates
- Pure ReCoN execution with continuous sur magnitudes
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Any
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, "/workspace/recon-platform")

from .hypothesis_manager import HypothesisManager
from .learning_manager import LearningManager


class ReCoNArcAngel:
    """
    Main ReCoN ARC Angel agent that interfaces with the ARC-AGI harness.
    
    This agent implements the approach described in the refined plan:
    - Uses CNNValidActionTerminal for action/coordinate prediction
    - Builds hierarchical ReCoN graph (root -> actions -> regions)
    - Applies continuous sur magnitudes for probability propagation
    - Learns via deduplicated experience buffer with frame change detection
    - Resets on score increase (new level) like StochasticGoose
    """
    
    def __init__(self, game_id: str = "default"):
        """
        Initialize the ReCoN ARC Angel agent.
        
        Args:
            game_id: Identifier for the game instance
        """
        self.game_id = game_id
        
        # Core components
        self.hypothesis_manager = HypothesisManager()
        self.hypothesis_manager.build_structure()
        
        self.learning_manager = LearningManager(
            buffer_size=200000,
            batch_size=64,
            train_frequency=5,
            learning_rate=0.0001
        )
        self.learning_manager.set_cnn_terminal(self.hypothesis_manager.cnn_terminal)
        
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
            'training_steps': 0,
            'unique_experiences': 0
        }
    
    def _convert_frame_to_tensor(self, frame_data: Any) -> torch.Tensor:
        """
        Convert frame data to tensor format for CNN.
        
        Args:
            frame_data: Frame data from harness (with .frame attribute)
            
        Returns:
            Tensor of shape (16, 64, 64) - one-hot encoded colors
        """
        frame = np.array(frame_data.frame, dtype=np.int64)
        
        # Handle animation frames (take last frame)
        if len(frame.shape) == 3:
            frame = frame[-1]
        
        if frame.shape != (64, 64):
            raise ValueError(f"Expected frame shape (64, 64), got {frame.shape}")
        
        # One-hot encode: (64, 64) -> (16, 64, 64)
        tensor = torch.zeros(16, 64, 64, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        
        return tensor
    
    def _apply_available_actions_mask(self, available_actions: List[Any]):
        """
        Apply availability mask to hypothesis manager.
        
        Args:
            available_actions: List of available GameAction enums from harness
        """
        # Convert GameAction enums to string names
        available_names = []
        for action in available_actions:
            if hasattr(action, 'name'):
                available_names.append(action.name)
            elif hasattr(action, 'value'):
                available_names.append(f"ACTION{action.value}")
            else:
                available_names.append(str(action))
        
        # Map harness action names to internal node IDs
        action_mapping = {
            "ACTION1": "action_1",
            "ACTION2": "action_2", 
            "ACTION3": "action_3",
            "ACTION4": "action_4",
            "ACTION5": "action_5",
            "ACTION6": "action_click"
        }
        
        from recon_engine.node import ReCoNState
        
        # PURE RECON AVAILABILITY: Set root→child sub weight to ≈0 for unavailable actions
        # This defers requests rather than forcing FAILED state, maintaining pure ReCoN semantics
        for harness_action, node_id in action_mapping.items():
            if harness_action not in available_names:
                # Set sub link weight from root to unavailable action to near zero
                for link in self.hypothesis_manager.graph.get_links(source="frame_change_hypothesis", target=node_id):
                    if link.type == "sub":
                        link.weight = 1e-6  # Near zero but not exactly zero to avoid division issues
                
                # Also set region weights to near zero if ACTION6 not available
                if harness_action == "ACTION6":
                    for region_y in range(self.hypothesis_manager.regions_per_dim):
                        for region_x in range(self.hypothesis_manager.regions_per_dim):
                            region_id = f"region_{region_y}_{region_x}"
                            for link in self.hypothesis_manager.graph.get_links(source="action_click", target=region_id):
                                if link.type == "sub":
                                    link.weight = 1e-6
    
    def _convert_to_game_action(self, action_type: str, coords: Optional[Tuple[int, int]] = None) -> Any:
        """
        Convert internal action to GameAction for harness.
        
        Args:
            action_type: Internal action type ("action_1", "action_click", etc.)
            coords: Coordinates if action_click
            
        Returns:
            GameAction object for harness
        """
        # Import GameAction from harness (this will be replaced with real import in deployment)
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
                game_action.reasoning = f"ReCoN ARC Angel ACTION6 at ({x}, {y})"
            else:
                game_action.reasoning = f"ReCoN ARC Angel {action_type.upper()}"
            
            return game_action
            
        except ImportError:
            # Fallback for testing - create mock object
            class MockGameAction:
                def __init__(self, action_type):
                    self.name = action_type
                    # Convert internal action names to harness format
                    if action_type.startswith("action_"):
                        action_num = action_type.split("_")[1]
                        self.action_type = f"ACTION{action_num}"
                    elif action_type == "action_click":
                        self.action_type = "ACTION6"
                    else:
                        self.action_type = action_type.upper()
                    self.reasoning = ""
                    self.data = {}
                def set_data(self, data):
                    self.data = data
            
            game_action = MockGameAction(action_type)
            
            if action_type == "action_click" and coords is not None:
                y, x = coords
                game_action.set_data({"x": x, "y": y})
                game_action.reasoning = f"ReCoN ARC Angel ACTION6 at ({x}, {y})"
            else:
                game_action.reasoning = f"ReCoN ARC Angel {action_type.upper()}"
            
            return game_action
    
    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """
        Main action selection method called by harness.
        
        Args:
            frames: List of previous frames
            latest_frame: Current frame data
            
        Returns:
            GameAction for harness
        """
        # Handle score changes (new level detection)
        if hasattr(latest_frame, 'score') and latest_frame.score != self.current_score:
            score_increased = self.learning_manager.on_score_change(latest_frame.score)
            if score_increased:
                # Reset hypothesis manager on score increase
                self.hypothesis_manager.reset()
                self.stats['score_resets'] += 1
            self.current_score = latest_frame.score
        
        # Handle game reset states
        if hasattr(latest_frame, 'state') and latest_frame.state in ["NOT_PLAYED", "GAME_OVER"]:
            self.prev_frame_tensor = None
            self.prev_action_type = None
            self.prev_coords = None
            
            # Return RESET action
            try:
                from agents.structs import GameAction
                action = GameAction.RESET
                action.reasoning = "ReCoN ARC Angel game reset"
                return action
            except ImportError:
                # Fallback for testing
                class MockReset:
                    def __init__(self):
                        self.name = "RESET"
                        self.reasoning = "ReCoN ARC Angel game reset"
                return MockReset()
        
        # Convert frame to tensor
        try:
            current_frame_tensor = self._convert_frame_to_tensor(latest_frame)
        except Exception as e:
            print(f"ReCoN ARC Angel: Error converting frame: {e}")
            # Return fallback action
            try:
                from agents.structs import GameAction
                return GameAction.ACTION1
            except ImportError:
                return self._convert_to_game_action("action_1")
        
        # Add experience from previous action if available
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
            except Exception as e:
                print(f"ReCoN ARC Angel: Error adding experience: {e}")
        
        # Update hypothesis weights from current frame
        try:
            self.hypothesis_manager.update_weights_from_frame(current_frame_tensor)
        except Exception as e:
            print(f"ReCoN ARC Angel: Error updating weights: {e}")
        
        # Reset and request frame change
        self.hypothesis_manager.reset()
        
        # Apply availability mask AFTER reset
        if hasattr(latest_frame, 'available_actions'):
            self._apply_available_actions_mask(latest_frame.available_actions)
        
        # Request frame change and run ReCoN propagation
        self.hypothesis_manager.request_frame_change()
        
        # Run several propagation steps
        for _ in range(5):
            self.hypothesis_manager.propagate_step()
        
        # Get best action with availability enforcement
        best_action, best_coords = self.hypothesis_manager.get_best_action(
            available_actions=latest_frame.available_actions if hasattr(latest_frame, 'available_actions') else None
        )
        
        
        
        if best_action is None:
            # Fallback to first available action or ACTION1
            if hasattr(latest_frame, 'available_actions') and latest_frame.available_actions:
                # Convert first available action
                first_available = latest_frame.available_actions[0]
                if hasattr(first_available, 'name'):
                    action_name = first_available.name.lower()
                    if action_name.startswith('action'):
                        best_action = action_name.replace('action', 'action_')
                    else:
                        best_action = "action_1"
                elif isinstance(first_available, str):
                    # Handle string action names like "ACTION2"
                    if first_available.startswith('ACTION'):
                        action_num = first_available.replace('ACTION', '')
                        best_action = f"action_{action_num}"
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
            print(f"ReCoN ARC Angel: Error converting action: {e}")
            # Fallback
            try:
                from agents.structs import GameAction
                game_action = GameAction.ACTION1
                game_action.reasoning = "ReCoN ARC Angel fallback"
            except ImportError:
                game_action = self._convert_to_game_action("action_1")
        
        # Store current state for next iteration
        self.prev_frame_tensor = current_frame_tensor
        self.prev_action_type = best_action
        self.prev_coords = best_coords
        
        # Update counters
        self.action_count += 1
        self.stats['total_actions'] += 1
        self.learning_manager.step()
        
        # Train if needed
        if self.learning_manager.should_train():
            try:
                metrics = self.learning_manager.train_step()
                if metrics:
                    self.stats['training_steps'] += 1
            except Exception as e:
                print(f"ReCoN ARC Angel: Error training: {e}")
        
        return game_action
    
    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """
        Check if agent is done playing.
        
        Args:
            frames: List of previous frames
            latest_frame: Current frame data
            
        Returns:
            True if game is complete
        """
        if hasattr(latest_frame, 'state'):
            return latest_frame.state in ["WIN", "GAME_OVER"]
        return False
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        stats = {
            **self.stats,
            'current_score': self.current_score,
            'action_count': self.action_count,
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
        
        # Reset stats
        self.stats = {
            'total_actions': 0,
            'score_resets': 0,
            'training_steps': 0,
            'unique_experiences': 0
        }
