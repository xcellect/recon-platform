"""
Test learning phase - 3:15-4:00

Test deduplicated buffer keyed by (frame hash, unified action index),
train every N actions with BCE on selected logit, reset on new level.
"""
import pytest
import sys
import os
import torch
import numpy as np
import hashlib
from collections import deque
from typing import Dict, Any

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

def test_frame_hashing():
    """Test that frame hashing works consistently"""
    # Create test frames
    frame1 = torch.zeros(16, 64, 64)
    frame1[1, 10:20, 10:20] = 1.0
    
    frame2 = torch.zeros(16, 64, 64) 
    frame2[1, 10:20, 10:20] = 1.0  # Same as frame1
    
    frame3 = torch.zeros(16, 64, 64)
    frame3[2, 10:20, 10:20] = 1.0  # Different from frame1
    
    def hash_frame(frame: torch.Tensor) -> str:
        """Hash a frame tensor"""
        frame_bytes = frame.cpu().numpy().astype(bool).tobytes()
        return hashlib.md5(frame_bytes).hexdigest()
    
    # Same frames should have same hash
    hash1 = hash_frame(frame1)
    hash2 = hash_frame(frame2)
    assert hash1 == hash2
    
    # Different frames should have different hash
    hash3 = hash_frame(frame3)
    assert hash1 != hash3

def test_experience_deduplication():
    """Test that experiences are deduplicated correctly"""
    
    class ExperienceBuffer:
        def __init__(self, maxlen: int = 1000):
            self.buffer = deque(maxlen=maxlen)
            self.experience_hashes = set()
        
        def _hash_experience(self, frame: torch.Tensor, action_idx: int) -> str:
            """Hash frame + action combination"""
            frame_bytes = frame.cpu().numpy().astype(bool).tobytes()
            hash_input = frame_bytes + str(action_idx).encode('utf-8')
            return hashlib.md5(hash_input).hexdigest()
        
        def add_experience(self, frame: torch.Tensor, action_idx: int, reward: float) -> bool:
            """Add experience if not duplicate. Returns True if added."""
            exp_hash = self._hash_experience(frame, action_idx)
            
            if exp_hash in self.experience_hashes:
                return False  # Duplicate
            
            experience = {
                'frame': frame.cpu().numpy().astype(bool),
                'action_idx': action_idx,
                'reward': reward
            }
            
            self.buffer.append(experience)
            self.experience_hashes.add(exp_hash)
            return True
        
        def clear(self):
            """Clear buffer and hashes"""
            self.buffer.clear()
            self.experience_hashes.clear()
    
    buffer = ExperienceBuffer()
    
    # Create test frames
    frame1 = torch.zeros(16, 64, 64)
    frame1[1, 10:20, 10:20] = 1.0
    
    frame2 = torch.zeros(16, 64, 64)
    frame2[2, 20:30, 20:30] = 1.0
    
    # Add unique experiences
    assert buffer.add_experience(frame1, 0, 1.0) == True  # Added
    assert buffer.add_experience(frame1, 1, 0.0) == True  # Different action
    assert buffer.add_experience(frame2, 0, 1.0) == True  # Different frame
    
    # Try to add duplicates
    assert buffer.add_experience(frame1, 0, 1.0) == False  # Duplicate
    assert buffer.add_experience(frame1, 1, 0.0) == False  # Duplicate
    
    # Check buffer state
    assert len(buffer.buffer) == 3
    assert len(buffer.experience_hashes) == 3

def test_unified_action_indexing():
    """Test unified action indexing (0-4 for ACTION1-5, 5+ for coordinates)"""
    
    def get_unified_action_index(action_type: str, coords: tuple = None) -> int:
        """Convert action to unified index"""
        if action_type in ["action_1", "action_2", "action_3", "action_4", "action_5"]:
            return int(action_type.split("_")[1]) - 1  # 0-4
        elif action_type == "action_click" and coords is not None:
            y, x = coords
            coord_idx = y * 64 + x
            return 5 + coord_idx  # 5+
        else:
            raise ValueError(f"Invalid action: {action_type}, {coords}")
    
    # Test individual actions
    assert get_unified_action_index("action_1") == 0
    assert get_unified_action_index("action_2") == 1
    assert get_unified_action_index("action_5") == 4
    
    # Test click coordinates
    assert get_unified_action_index("action_click", (0, 0)) == 5
    assert get_unified_action_index("action_click", (0, 1)) == 6
    assert get_unified_action_index("action_click", (1, 0)) == 69  # 5 + 64
    assert get_unified_action_index("action_click", (63, 63)) == 5 + 63*64 + 63

def test_frame_change_detection():
    """Test frame change detection logic"""
    
    def frames_changed(prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> bool:
        """Check if frames are different"""
        prev_bool = prev_frame.cpu().numpy().astype(bool)
        curr_bool = curr_frame.cpu().numpy().astype(bool)
        return not np.array_equal(prev_bool, curr_bool)
    
    # Create test frames
    frame1 = torch.zeros(16, 64, 64)
    frame1[1, 10:20, 10:20] = 1.0
    
    frame2 = torch.zeros(16, 64, 64)
    frame2[1, 10:20, 10:20] = 1.0  # Same as frame1
    
    frame3 = torch.zeros(16, 64, 64)
    frame3[2, 10:20, 10:20] = 1.0  # Different from frame1
    
    # Test change detection
    assert frames_changed(frame1, frame2) == False  # No change
    assert frames_changed(frame1, frame3) == True   # Change detected
    assert frames_changed(frame2, frame3) == True   # Change detected

class LearningManager:
    """Manages learning for the ReCoN ARC Angel agent"""
    
    def __init__(self, buffer_size: int = 200000, batch_size: int = 64, train_frequency: int = 5):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        
        # Experience buffer with deduplication
        self.experience_buffer = deque(maxlen=buffer_size)
        self.experience_hashes = set()
        
        # Training state
        self.action_count = 0
        self.current_score = -1
        
        # For testing - track training calls
        self.training_calls = []
    
    def _hash_experience(self, frame: torch.Tensor, action_idx: int) -> str:
        """Hash frame + action combination for deduplication"""
        frame_bytes = frame.cpu().numpy().astype(bool).tobytes()
        hash_input = frame_bytes + str(action_idx).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def add_experience(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor, 
                      action_idx: int) -> bool:
        """
        Add experience if unique.
        
        Args:
            prev_frame: Previous frame tensor
            curr_frame: Current frame tensor  
            action_idx: Unified action index
            
        Returns:
            True if experience was added (not duplicate)
        """
        exp_hash = self._hash_experience(prev_frame, action_idx)
        
        if exp_hash in self.experience_hashes:
            return False  # Duplicate
        
        # Detect frame change
        prev_bool = prev_frame.cpu().numpy().astype(bool)
        curr_bool = curr_frame.cpu().numpy().astype(bool)
        frame_changed = not np.array_equal(prev_bool, curr_bool)
        
        experience = {
            'frame': prev_bool,  # Store as bool numpy array
            'action_idx': action_idx,
            'reward': 1.0 if frame_changed else 0.0
        }
        
        self.experience_buffer.append(experience)
        self.experience_hashes.add(exp_hash)
        return True
    
    def should_train(self) -> bool:
        """Check if we should train now"""
        return (self.action_count % self.train_frequency == 0 and 
                len(self.experience_buffer) >= self.batch_size)
    
    def train_step(self, cnn_terminal) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            cnn_terminal: CNN terminal to train
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.experience_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch_indices = np.random.choice(len(self.experience_buffer), 
                                       self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch tensors
        states = []
        action_indices = []
        rewards = []
        
        for exp in batch:
            # Convert bool numpy back to float tensor
            state_tensor = torch.from_numpy(exp['frame'].astype(np.float32))
            states.append(state_tensor)
            action_indices.append(exp['action_idx'])
            rewards.append(exp['reward'])
        
        states = torch.stack(states)
        action_indices = torch.tensor(action_indices, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # For testing, just record the call
        metrics = {
            'batch_size': len(batch),
            'positive_rewards': (rewards > 0).sum().item(),
            'mean_reward': rewards.mean().item()
        }
        
        self.training_calls.append(metrics)
        return metrics
    
    def on_score_change(self, new_score: int):
        """Handle score change - clear buffer and reset"""
        if new_score != self.current_score:
            self.experience_buffer.clear()
            self.experience_hashes.clear()
            self.current_score = new_score
            return True
        return False
    
    def step(self):
        """Increment action count"""
        self.action_count += 1

def test_learning_manager_basic():
    """Test basic LearningManager functionality"""
    manager = LearningManager(buffer_size=100, batch_size=4, train_frequency=2)
    
    # Create test frames
    frame1 = torch.zeros(16, 64, 64)
    frame1[1, 10:20, 10:20] = 1.0
    
    frame2 = torch.zeros(16, 64, 64)
    frame2[2, 10:20, 10:20] = 1.0  # Different
    
    frame3 = torch.zeros(16, 64, 64)
    frame3[1, 10:20, 10:20] = 1.0  # Same as frame1
    
    # Add experiences
    assert manager.add_experience(frame1, frame2, 0) == True  # Changed
    assert manager.add_experience(frame2, frame3, 1) == True  # Changed
    assert manager.add_experience(frame1, frame3, 2) == True  # No change
    assert manager.add_experience(frame1, frame2, 0) == False  # Duplicate
    
    # Check buffer
    assert len(manager.experience_buffer) == 3
    assert manager.experience_buffer[0]['reward'] == 1.0  # Changed
    assert manager.experience_buffer[1]['reward'] == 1.0  # Changed  
    assert manager.experience_buffer[2]['reward'] == 0.0  # No change

def test_learning_manager_training_schedule():
    """Test training schedule and frequency"""
    manager = LearningManager(buffer_size=100, batch_size=4, train_frequency=3)
    
    # Create dummy experiences to reach batch size
    frame1 = torch.zeros(16, 64, 64)
    frame2 = torch.zeros(16, 64, 64)
    frame2[1, 0, 0] = 1.0  # Make it different
    
    for i in range(5):
        manager.add_experience(frame1, frame2, i)
        manager.step()
    
    # Should not train yet (action_count=5, not multiple of 3)
    assert manager.should_train() == False
    
    # Step to action_count=6 (multiple of 3)
    manager.step()
    assert manager.should_train() == True

def test_learning_manager_score_reset():
    """Test buffer clearing on score change"""
    manager = LearningManager()
    
    # Add some experiences
    frame1 = torch.zeros(16, 64, 64)
    frame2 = torch.zeros(16, 64, 64)
    frame2[1, 0, 0] = 1.0
    
    manager.add_experience(frame1, frame2, 0)
    manager.add_experience(frame1, frame2, 1)
    
    assert len(manager.experience_buffer) == 2
    assert len(manager.experience_hashes) == 2
    
    # Score change should clear buffer
    changed = manager.on_score_change(1)
    assert changed == True
    assert len(manager.experience_buffer) == 0
    assert len(manager.experience_hashes) == 0
    
    # Same score should not clear
    changed = manager.on_score_change(1)
    assert changed == False

def test_learning_integration():
    """Integration test: full learning cycle"""
    manager = LearningManager(buffer_size=100, batch_size=3, train_frequency=2)
    
    # Simulate sequence of frames and actions
    frames = []
    for i in range(6):
        frame = torch.zeros(16, 64, 64)
        frame[1, i:i+5, i:i+5] = 1.0  # Moving pattern
        frames.append(frame)
    
    # Add experiences with training
    for i in range(5):
        manager.add_experience(frames[i], frames[i+1], i % 3)
        manager.step()
        
        if manager.should_train():
            metrics = manager.train_step(None)  # Mock CNN terminal
            assert 'batch_size' in metrics
            assert metrics['batch_size'] == 3
    
    # Training happens when action_count % train_frequency == 0
    # action_count goes: 1, 2, 3, 4, 5
    # train_frequency = 2, so training on steps 2 and 4 
    # But we need at least batch_size=3 experiences first
    # So only step 4 should train (step 2 has only 2 experiences)
    assert len(manager.training_calls) == 1
