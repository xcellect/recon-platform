"""
Test harness adapter phase - 4:45-5:30

Test thin agent wrapper to extract frame, call agent, handle available_actions,
return proper GameAction with (x,y) for ACTION6.
"""
import pytest
import sys
import os
import torch
import numpy as np
from typing import List, Optional

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

# Mock the ARC-AGI structures for testing
class MockFrameData:
    """Mock FrameData for testing"""
    def __init__(self, frame: np.ndarray, available_actions: List[str], score: int = 0, state: str = "PLAYING"):
        self.frame = frame
        self.available_actions = available_actions
        self.score = score
        self.state = state

class MockGameAction:
    """Mock GameAction for testing"""
    def __init__(self, action_type: str):
        self.action_type = action_type
        self.reasoning = ""
        self.data = {}
    
    def set_data(self, data: dict):
        self.data = data

# Mock the actual GameAction enum values
class MockGameActionEnum:
    ACTION1 = MockGameAction("ACTION1")
    ACTION2 = MockGameAction("ACTION2") 
    ACTION3 = MockGameAction("ACTION3")
    ACTION4 = MockGameAction("ACTION4")
    ACTION5 = MockGameAction("ACTION5")
    ACTION6 = MockGameAction("ACTION6")
    RESET = MockGameAction("RESET")

def test_frame_conversion():
    """Test conversion from harness frame format to tensor"""
    # Create mock frame data (64x64 grid with color indices)
    frame_array = np.zeros((64, 64), dtype=int)
    frame_array[10:20, 10:20] = 1  # Color 1 square
    frame_array[30:40, 30:40] = 3  # Color 3 square
    
    def convert_frame_to_tensor(frame_data: MockFrameData) -> torch.Tensor:
        """Convert frame data to one-hot tensor"""
        frame = np.array(frame_data.frame, dtype=np.int64)
        
        # Handle animation frames (take last frame)
        if len(frame.shape) == 3:
            frame = frame[-1]
        
        assert frame.shape == (64, 64), f"Expected (64, 64), got {frame.shape}"
        
        # One-hot encode: (64, 64) -> (16, 64, 64)
        tensor = torch.zeros(16, 64, 64, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        
        return tensor
    
    frame_data = MockFrameData(frame_array, ["ACTION1", "ACTION2", "ACTION6"])
    tensor = convert_frame_to_tensor(frame_data)
    
    # Check shape
    assert tensor.shape == (16, 64, 64)
    
    # Check one-hot encoding
    assert tensor[0, 0, 0] == 1  # Background (color 0)
    assert tensor[1, 15, 15] == 1  # Color 1 region
    assert tensor[3, 35, 35] == 1  # Color 3 region
    assert tensor.sum() == 64 * 64  # Exactly one hot per pixel

def test_action_conversion():
    """Test conversion from internal actions to GameAction"""
    
    def convert_to_game_action(action_type: str, coords: Optional[tuple] = None) -> MockGameAction:
        """Convert internal action to GameAction"""
        action_mapping = {
            "action_1": MockGameActionEnum.ACTION1,
            "action_2": MockGameActionEnum.ACTION2,
            "action_3": MockGameActionEnum.ACTION3,
            "action_4": MockGameActionEnum.ACTION4,
            "action_5": MockGameActionEnum.ACTION5,
            "action_click": MockGameActionEnum.ACTION6
        }
        
        if action_type not in action_mapping:
            raise ValueError(f"Unknown action type: {action_type}")
        
        game_action = action_mapping[action_type]
        
        if action_type == "action_click" and coords is not None:
            y, x = coords
            game_action.set_data({"x": x, "y": y})
            game_action.reasoning = f"ACTION6 at ({x}, {y})"
        else:
            game_action.reasoning = f"{action_type.upper()}"
        
        return game_action
    
    # Test individual actions
    action1 = convert_to_game_action("action_1")
    assert action1.action_type == "ACTION1"
    assert action1.reasoning == "ACTION_1"
    
    # Test click action with coordinates
    action6 = convert_to_game_action("action_click", (25, 35))
    assert action6.action_type == "ACTION6"
    assert action6.data["x"] == 35
    assert action6.data["y"] == 25
    assert "35, 25" in action6.reasoning  # Shows (x, y) in reasoning

def test_available_actions_filtering():
    """Test filtering based on available actions"""
    from recon_agents.recon_arc_angel.hypothesis_manager import HypothesisManager
    
    def apply_available_actions_mask(manager: HypothesisManager, available_actions: List[str]):
        """Apply availability mask to hypothesis manager"""
        # Map harness action names to internal node IDs
        action_mapping = {
            "ACTION1": "action_1",
            "ACTION2": "action_2", 
            "ACTION3": "action_3",
            "ACTION4": "action_4",
            "ACTION5": "action_5",
            "ACTION6": "action_click"
        }
        
        # Set unavailable actions to FAILED
        from recon_engine.node import ReCoNState
        
        for harness_action, node_id in action_mapping.items():
            if harness_action not in available_actions:
                if node_id in manager.graph.nodes:
                    manager.graph.nodes[node_id].state = ReCoNState.FAILED
                    manager.graph.nodes[node_id].activation = -1.0
                    
                    # Also fail all regions if ACTION6 not available
                    if harness_action == "ACTION6":
                        for region_y in range(manager.regions_per_dim):
                            for region_x in range(manager.regions_per_dim):
                                region_id = f"region_{region_y}_{region_x}"
                                if region_id in manager.graph.nodes:
                                    manager.graph.nodes[region_id].state = ReCoNState.FAILED
                                    manager.graph.nodes[region_id].activation = -1.0
    
    manager = HypothesisManager()
    manager.build_structure()
    
    # Set all actions to be viable initially
    from recon_engine.node import ReCoNState
    for i in range(1, 6):
        action_id = f"action_{i}"
        manager.graph.nodes[action_id].state = ReCoNState.TRUE
        manager.graph.nodes[action_id].activation = 0.5
    
    manager.graph.nodes["action_click"].state = ReCoNState.TRUE
    manager.graph.nodes["action_click"].activation = 0.5
    
    # Apply mask: only ACTION1 and ACTION3 available
    apply_available_actions_mask(manager, ["ACTION1", "ACTION3"])
    
    # Get best action
    best_action, _ = manager.get_best_action()
    assert best_action in ["action_1", "action_3"]

class ReCoNArcAngel:
    """Main agent class that interfaces with the ARC-AGI harness"""
    
    def __init__(self, game_id: str = "test"):
        self.game_id = game_id
        
        # Core components
        from recon_agents.recon_arc_angel.hypothesis_manager import HypothesisManager
        from recon_agents.recon_arc_angel.learning_manager import LearningManager
        
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
        self.action_count = 0
        self.current_score = -1
    
    def _convert_frame_to_tensor(self, frame_data: MockFrameData) -> torch.Tensor:
        """Convert frame data to tensor format"""
        frame = np.array(frame_data.frame, dtype=np.int64)
        
        # Handle animation frames (take last frame)
        if len(frame.shape) == 3:
            frame = frame[-1]
        
        assert frame.shape == (64, 64), f"Expected (64, 64), got {frame.shape}"
        
        # One-hot encode: (64, 64) -> (16, 64, 64)
        tensor = torch.zeros(16, 64, 64, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        
        return tensor
    
    def _apply_available_actions_mask(self, available_actions: List[str]):
        """Apply availability mask to hypothesis manager"""
        action_mapping = {
            "ACTION1": "action_1",
            "ACTION2": "action_2", 
            "ACTION3": "action_3",
            "ACTION4": "action_4",
            "ACTION5": "action_5",
            "ACTION6": "action_click"
        }
        
        from recon_engine.node import ReCoNState
        
        # Set unavailable actions to FAILED
        for harness_action, node_id in action_mapping.items():
            if harness_action not in available_actions:
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
    
    def _convert_to_game_action(self, action_type: str, coords: Optional[tuple] = None) -> MockGameAction:
        """Convert internal action to GameAction"""
        action_mapping = {
            "action_1": MockGameActionEnum.ACTION1,
            "action_2": MockGameActionEnum.ACTION2,
            "action_3": MockGameActionEnum.ACTION3,
            "action_4": MockGameActionEnum.ACTION4,
            "action_5": MockGameActionEnum.ACTION5,
            "action_click": MockGameActionEnum.ACTION6
        }
        
        if action_type not in action_mapping:
            raise ValueError(f"Unknown action type: {action_type}")
        
        game_action = action_mapping[action_type]
        
        if action_type == "action_click" and coords is not None:
            y, x = coords
            game_action.set_data({"x": x, "y": y})
            game_action.reasoning = f"ACTION6 at ({x}, {y})"
        else:
            game_action.reasoning = f"{action_type.upper()}"
        
        return game_action
    
    def choose_action(self, frames: List[MockFrameData], latest_frame: MockFrameData) -> MockGameAction:
        """Main action selection method called by harness"""
        
        # Handle score changes
        if latest_frame.score != self.current_score:
            score_increased = self.learning_manager.on_score_change(latest_frame.score)
            if score_increased:
                # Reset hypothesis manager on score increase
                self.hypothesis_manager.reset()
            self.current_score = latest_frame.score
        
        # Handle game reset
        if latest_frame.state in ["NOT_PLAYED", "GAME_OVER"]:
            self.prev_frame_tensor = None
            return MockGameActionEnum.RESET
        
        # Convert frame to tensor
        current_frame_tensor = self._convert_frame_to_tensor(latest_frame)
        
        # Add experience from previous action if available
        if self.prev_frame_tensor is not None and self.action_count > 0:
            # Get the previous action from learning manager or hypothesis manager
            # For now, we'll skip this step in the test
            pass
        
        # Update hypothesis weights from current frame
        self.hypothesis_manager.update_weights_from_frame(current_frame_tensor)
        
        # Reset and request frame change
        self.hypothesis_manager.reset()
        
        # Apply availability mask AFTER reset
        self._apply_available_actions_mask(latest_frame.available_actions)
        
        self.hypothesis_manager.request_frame_change()
        
        # Run ReCoN propagation
        for _ in range(5):  # Run several steps
            self.hypothesis_manager.propagate_step()
        
        # Get best action with availability enforcement
        best_action, best_coords = self.hypothesis_manager.get_best_action(
            available_actions=latest_frame.available_actions
        )
        
        if best_action is None:
            # Fallback to first available action
            if latest_frame.available_actions:
                # Convert first available action like "ACTION2" to "action_2"
                first_available = latest_frame.available_actions[0]
                if first_available.startswith('ACTION'):
                    action_num = first_available.replace('ACTION', '')
                    best_action = f"action_{action_num}"
                else:
                    best_action = "action_1"
            else:
                return MockGameActionEnum.RESET
        
        # Convert to game action
        game_action = self._convert_to_game_action(best_action, best_coords)
        
        # Store current frame for next iteration
        self.prev_frame_tensor = current_frame_tensor
        self.action_count += 1
        self.learning_manager.step()
        
        # Train if needed
        if self.learning_manager.should_train():
            self.learning_manager.train_step()
        
        return game_action
    
    def is_done(self, frames: List[MockFrameData], latest_frame: MockFrameData) -> bool:
        """Check if agent is done"""
        return latest_frame.state in ["WIN", "GAME_OVER"]

def test_recon_arc_angel_basic():
    """Test basic ReCoNArcAngel functionality"""
    agent = ReCoNArcAngel("test_game")
    
    # Create test frame
    frame_array = np.zeros((64, 64), dtype=int)
    frame_array[20:30, 20:30] = 2  # Color 2 square
    
    frame_data = MockFrameData(
        frame=frame_array,
        available_actions=["ACTION1", "ACTION2", "ACTION6"],
        score=0,
        state="PLAYING"
    )
    
    # Choose action
    action = agent.choose_action([frame_data], frame_data)
    
    # Should return a valid action
    assert action is not None
    assert hasattr(action, 'action_type')
    assert hasattr(action, 'reasoning')

def test_recon_arc_angel_click_action():
    """Test that click actions include coordinates"""
    agent = ReCoNArcAngel("test_game")
    
    # Create frame that might favor clicking
    frame_array = np.zeros((64, 64), dtype=int)
    frame_array[40:50, 40:50] = 1  # Pattern in region (5,5)
    
    frame_data = MockFrameData(
        frame=frame_array,
        available_actions=["ACTION6"],  # Only allow clicking
        score=0,
        state="PLAYING"
    )
    
    # Choose action
    action = agent.choose_action([frame_data], frame_data)
    
    # If ACTION6 chosen, should have coordinates
    if action.action_type == "ACTION6":
        assert "x" in action.data
        assert "y" in action.data
        assert 0 <= action.data["x"] < 64
        assert 0 <= action.data["y"] < 64

def test_recon_arc_angel_score_change():
    """Test handling of score changes"""
    agent = ReCoNArcAngel("test_game")
    
    frame_array = np.zeros((64, 64), dtype=int)
    
    # First frame with score 0
    frame1 = MockFrameData(frame_array, ["ACTION1"], score=0)
    action1 = agent.choose_action([frame1], frame1)
    
    buffer_size_before = len(agent.learning_manager.experience_buffer)
    
    # Second frame with score 1 (increased)
    frame2 = MockFrameData(frame_array, ["ACTION1"], score=1)
    action2 = agent.choose_action([frame1, frame2], frame2)
    
    # Buffer should be cleared on score increase
    buffer_size_after = len(agent.learning_manager.experience_buffer)
    assert buffer_size_after == 0  # Should be cleared

def test_recon_arc_angel_availability_masking():
    """Test that availability masking works in full agent"""
    agent = ReCoNArcAngel("test_game")
    
    frame_array = np.zeros((64, 64), dtype=int)
    frame_array[10:20, 10:20] = 3
    
    # Test with only ACTION2 and ACTION4 available
    frame_data = MockFrameData(
        frame=frame_array,
        available_actions=["ACTION2", "ACTION4"],
        score=0,
        state="PLAYING"
    )
    
    action = agent.choose_action([frame_data], frame_data)
    
    # With airtight availability masking, should only select from available actions
    assert action.action_type in ["ACTION2", "ACTION4"], f"Expected ACTION2 or ACTION4, got {action.action_type}"
    
    # Test edge case: no actions available
    frame_data_no_actions = MockFrameData(
        frame=frame_array,
        available_actions=[],
        score=0,
        state="PLAYING"
    )
    
    action_no_actions = agent.choose_action([frame_data_no_actions], frame_data_no_actions)
    
    # Should fallback gracefully when no actions available
    assert hasattr(action_no_actions, 'action_type')
    assert hasattr(action_no_actions, 'reasoning')
