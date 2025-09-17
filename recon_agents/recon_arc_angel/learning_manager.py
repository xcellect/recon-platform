"""
Learning Manager

Manages experience collection, deduplication, and training for the ReCoN ARC Angel agent.
Implements the StochasticGoose-style learning approach:
- Deduplicated experience buffer keyed by (frame hash, unified action index)
- Binary classification: predict if action causes frame change
- Train every N actions with BCE on selected logit
- Reset on score increase (new level)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hashlib
from collections import deque
from typing import Dict, Any, Optional, Tuple

import sys
import os
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.neural_terminal import CNNValidActionTerminal


class LearningManager:
    """
    Manages learning for the ReCoN ARC Angel agent.
    
    Implements the StochasticGoose learning approach:
    - Experience buffer with deduplication
    - Frame change detection as supervision signal
    - Periodic training with BCE loss
    - Buffer reset on score increase
    """
    
    def __init__(self, buffer_size: int = 200000, batch_size: int = 64, 
                 train_frequency: int = 5, learning_rate: float = 0.0001):
        """
        Initialize the learning manager.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Batch size for training
            train_frequency: Train every N actions
            learning_rate: Learning rate for optimizer
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.learning_rate = learning_rate
        
        # Experience buffer with deduplication
        self.experience_buffer = deque(maxlen=buffer_size)
        self.experience_hashes = set()
        
        # Training state
        self.action_count = 0
        self.current_score = -1
        
        # CNN terminal and optimizer (set externally)
        self.cnn_terminal: Optional[CNNValidActionTerminal] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        # ResNet terminal and training (BlindSquirrel style)
        self.resnet_terminal = None
        self.resnet_optimizer = None
        self.state_graph = {}  # Track states for ResNet training
        self.transitions = {}  # Track transitions for distance-to-goal training
        
        # Training statistics
        self.training_stats = {
            'total_training_steps': 0,
            'total_experiences': 0,
            'buffer_resets': 0,
            'last_loss': 0.0
        }
    
    def set_cnn_terminal(self, cnn_terminal: CNNValidActionTerminal):
        """Set the CNN terminal and create optimizer"""
        self.cnn_terminal = cnn_terminal
        self.optimizer = optim.Adam(cnn_terminal.model.parameters(), lr=self.learning_rate)
    
    def set_resnet_terminal(self, resnet_terminal):
        """Set the ResNet terminal and create optimizer (BlindSquirrel style)"""
        self.resnet_terminal = resnet_terminal
        if torch.cuda.is_available():
            resnet_terminal.to_device('cuda')
        self.resnet_optimizer = optim.Adam(resnet_terminal.model.parameters(), lr=self.learning_rate)
    
    def _hash_experience(self, frame: torch.Tensor, action_idx: int) -> str:
        """Hash frame + action combination for deduplication"""
        frame_bytes = frame.cpu().numpy().astype(bool).tobytes()
        hash_input = frame_bytes + str(action_idx).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def get_unified_action_index(self, action_type: str, coords: Optional[Tuple[int, int]] = None) -> int:
        """
        Convert action to unified index for experience storage.
        
        Args:
            action_type: "action_1" through "action_5" or "action_click"
            coords: (y, x) coordinates if action_click
            
        Returns:
            Unified action index: 0-4 for ACTION1-5, 5+ for coordinates
        """
        if action_type in ["action_1", "action_2", "action_3", "action_4", "action_5"]:
            return int(action_type.split("_")[1]) - 1  # 0-4
        elif action_type == "action_click" and coords is not None:
            y, x = coords
            coord_idx = y * 64 + x
            return 5 + coord_idx  # 5+
        else:
            raise ValueError(f"Invalid action: {action_type}, {coords}")
    
    def add_experience(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor, 
                      action_type: str, coords: Optional[Tuple[int, int]] = None) -> bool:
        """
        Add experience if unique.
        
        Args:
            prev_frame: Previous frame tensor (16, 64, 64)
            curr_frame: Current frame tensor (16, 64, 64)
            action_type: Action that was taken
            coords: Coordinates if action_click
            
        Returns:
            True if experience was added (not duplicate)
        """
        action_idx = self.get_unified_action_index(action_type, coords)
        exp_hash = self._hash_experience(prev_frame, action_idx)
        
        if exp_hash in self.experience_hashes:
            return False  # Duplicate
        
        # Detect frame change
        prev_bool = prev_frame.cpu().numpy().astype(bool)
        curr_bool = curr_frame.cpu().numpy().astype(bool)
        frame_changed = not np.array_equal(prev_bool, curr_bool)
        
        experience = {
            'frame': prev_bool,  # Store as bool numpy array to save memory
            'action_idx': action_idx,
            'reward': 1.0 if frame_changed else 0.0
        }
        
        self.experience_buffer.append(experience)
        self.experience_hashes.add(exp_hash)
        self.training_stats['total_experiences'] += 1
        return True
    
    def should_train(self) -> bool:
        """Check if we should train now"""
        return (self.action_count % self.train_frequency == 0 and 
                len(self.experience_buffer) >= self.batch_size and
                self.cnn_terminal is not None and
                self.optimizer is not None)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics
        """
        if not self.should_train():
            return {}
        
        # Sample batch
        batch_indices = np.random.choice(len(self.experience_buffer), 
                                       self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch tensors
        states = []
        action_indices = []
        rewards = []
        
        device = next(self.cnn_terminal.model.parameters()).device
        
        for exp in batch:
            # Convert bool numpy back to float tensor
            state_tensor = torch.from_numpy(exp['frame'].astype(np.float32))
            states.append(state_tensor)
            action_indices.append(exp['action_idx'])
            rewards.append(exp['reward'])
        
        states = torch.stack(states).to(device)
        action_indices = torch.tensor(action_indices, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Forward pass
        self.optimizer.zero_grad()
        combined_logits = self.cnn_terminal.model(states)  # (batch, 4101)
        
        # BCE loss on selected actions only (StochasticGoose style)
        selected_logits = combined_logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)
        
        # Light entropy regularization (encourage exploration)
        all_probs = torch.sigmoid(combined_logits)
        action_probs = all_probs[:, :5]
        coord_probs = all_probs[:, 5:]
        
        action_entropy = action_probs.mean()
        coord_entropy = coord_probs.mean()
        
        # Total loss with entropy regularization
        total_loss = main_loss - 0.0001 * action_entropy - 0.00001 * coord_entropy
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Update statistics
        self.training_stats['total_training_steps'] += 1
        self.training_stats['last_loss'] = total_loss.item()
        
        # Training metrics
        accuracy = ((torch.sigmoid(selected_logits) > 0.5) == rewards).float().mean()
        
        metrics = {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'action_entropy': action_entropy.item(),
            'coord_entropy': coord_entropy.item(),
            'accuracy': accuracy.item(),
            'batch_size': len(batch),
            'positive_rewards': (rewards > 0).sum().item(),
            'mean_reward': rewards.mean().item()
        }
        
        return metrics
    
    def add_state_transition(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor,
                           action_type: str, coords, game_id: str, score: int):
        """
        Add state transition for ResNet training (BlindSquirrel style).
        
        This tracks states and transitions for distance-to-goal training.
        """
        if self.resnet_terminal is None:
            return
        
        # Create state keys
        prev_key = (game_id, score, hash(prev_frame.cpu().numpy().tobytes()))
        curr_key = (game_id, score, hash(curr_frame.cpu().numpy().tobytes()))
        
        # Store states
        self.state_graph[prev_key] = {'frame': prev_frame, 'score': score}
        self.state_graph[curr_key] = {'frame': curr_frame, 'score': score}
        
        # Store transition
        action_idx = self.get_unified_action_index(action_type, coords)
        self.transitions[(prev_key, action_idx)] = curr_key
    
    def on_score_change(self, new_score: int, game_id: str = "default") -> bool:
        """
        Handle score change - reset CNN and train ResNet on increase.
        
        Args:
            new_score: New score value
            game_id: Game identifier
            
        Returns:
            True if score increased and reset occurred
        """
        if new_score > self.current_score:
            # Train ResNet on score increase (BlindSquirrel style)
            if self.resnet_terminal is not None and new_score > 0:
                self._train_resnet_on_level_completion(game_id, new_score)
            
            # Reset CNN (StochasticGoose style)
            self.experience_buffer.clear()
            self.experience_hashes.clear()
            
            # Reset CNN model and optimizer
            if self.cnn_terminal is not None:
                self.cnn_terminal.model = self.cnn_terminal._create_action_model()
                if hasattr(self.cnn_terminal.model, 'to'):
                    device = next(iter(self.cnn_terminal.model.parameters())).device
                    self.cnn_terminal.model = self.cnn_terminal.model.to(device)
                
                self.optimizer = optim.Adam(self.cnn_terminal.model.parameters(), 
                                          lr=self.learning_rate)
                self.cnn_terminal.clear_cache()
            
            self.current_score = new_score
            self.training_stats['buffer_resets'] += 1
            return True
        
        self.current_score = new_score
        return False
    
    def _train_resnet_on_level_completion(self, game_id: str, completed_score: int):
        """Train ResNet when level is completed (BlindSquirrel approach)"""
        print(f"🎓 Training ResNet for {game_id} level {completed_score}")
        
        # Prepare training data from state transitions
        training_data = self._prepare_resnet_training_data(game_id, completed_score)
        
        if len(training_data) < self.batch_size:
            print(f"⚠️  Not enough ResNet training data: {len(training_data)} < {self.batch_size}")
            return
        
        # Train ResNet
        self.resnet_terminal.model.train()
        criterion = torch.nn.MSELoss()
        
        import time
        start_time = time.time()
        max_train_time = 15 * 60  # 15 minutes like BlindSquirrel
        
        for epoch in range(10):  # BlindSquirrel uses 10 epochs
            if time.time() - start_time > max_train_time:
                print(f"⚠️  Reached max ResNet training time")
                break
            
            # Sample batch
            if len(training_data) >= self.batch_size:
                batch_indices = np.random.choice(len(training_data), self.batch_size, replace=False)
                batch = [training_data[i] for i in batch_indices]
                
                # Prepare batch tensors
                states = torch.stack([item['state'] for item in batch])
                actions = torch.stack([item['action'] for item in batch])
                values = torch.stack([item['value'] for item in batch])
                
                device = next(self.resnet_terminal.model.parameters()).device
                states = states.to(device)
                actions = actions.to(device)
                values = values.to(device)
                
                # Forward pass (ResNet expects (state, action) as tuple)
                self.resnet_optimizer.zero_grad()
                predictions = self.resnet_terminal.model((states, actions))
                loss = criterion(predictions.squeeze(), values)
                
                # Backward pass
                loss.backward()
                self.resnet_optimizer.step()
                
                if epoch % 2 == 0:
                    print(f"  ResNet Epoch {epoch+1}/10: Loss {loss.item():.4f}")
        
        self.resnet_terminal.model.eval()
        print(f"✅ ResNet training completed for level {completed_score}")
    
    def _prepare_resnet_training_data(self, game_id: str, target_score: int):
        """Prepare ResNet training data using distance-to-goal"""
        # This is a simplified version of BlindSquirrel's approach
        training_data = []
        
        # For now, use simple heuristic: frame_changed = positive value
        for (prev_key, action_idx), next_key in self.transitions.items():
            if prev_key in self.state_graph and next_key in self.state_graph:
                prev_frame = self.state_graph[prev_key]['frame']
                next_frame = self.state_graph[next_key]['frame']
                
                # Simple value: 1.0 if frame changed, 0.0 if not
                frame_changed = not torch.equal(prev_frame, next_frame)
                value = 1.0 if frame_changed else 0.0
                
                # Create action tensor (match ResNet model input size)
                action_tensor = torch.zeros(10)  # ResNet model expects 10-dim action
                if action_idx < 5:
                    action_tensor[action_idx] = 1.0
                else:
                    action_tensor[5] = 1.0  # ACTION6
                
                training_data.append({
                    'state': prev_frame,
                    'action': action_tensor,
                    'value': torch.tensor(value, dtype=torch.float32)
                })
        
        return training_data
    
    def step(self):
        """Increment action count"""
        self.action_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        stats = {
            'action_count': self.action_count,
            'current_score': self.current_score,
            'buffer_size': len(self.experience_buffer),
            'unique_experiences': len(self.experience_hashes),
            'buffer_utilization': len(self.experience_buffer) / self.buffer_size,
            'should_train': self.should_train(),
            **self.training_stats
        }
        
        if self.cnn_terminal is not None:
            stats['cnn_cache_size'] = len(self.cnn_terminal._output_cache)
        
        return stats
    
    def clear(self):
        """Clear all learning state"""
        self.experience_buffer.clear()
        self.experience_hashes.clear()
        self.action_count = 0
        self.current_score = -1
        
        if self.cnn_terminal is not None:
            self.cnn_terminal.clear_cache()
        
        # Reset training stats
        self.training_stats = {
            'total_training_steps': 0,
            'total_experiences': 0,
            'buffer_resets': 0,
            'last_loss': 0.0
        }
