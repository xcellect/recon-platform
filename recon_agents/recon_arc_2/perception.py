"""
CNN Change Predictor for ReCoN ARC-2

Simple CNN that predicts which actions will cause frame changes.
Based on proven approach from ARC3-solution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class ChangePredictor(nn.Module):
    """
    CNN that predicts probability each action will change the frame.

    Input: 64x64 grid with 16 channels (one-hot encoded colors 0-15)
    Output: 6 probabilities for ACTION1-ACTION6 (including click actions)
    """

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = 6  # ACTION1-ACTION6

        # Simple CNN backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Global average pooling for efficiency
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, self.num_actions)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 16, 64, 64) - one-hot encoded grid

        Returns:
            logits: Tensor of shape (batch, 6) - change probability per action
        """
        # CNN feature extraction
        x = F.relu(self.conv1(x))  # (batch, 32, 64, 64)
        x = F.relu(self.conv2(x))  # (batch, 64, 64, 64)
        x = F.relu(self.conv3(x))  # (batch, 128, 64, 64)

        # Global pooling
        x = self.global_pool(x)    # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)

        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)       # (batch, 6)

        return logits

    def predict_change_probabilities(self, frame: np.ndarray) -> np.ndarray:
        """
        Predict probability each action will change the frame.

        Args:
            frame: numpy array of shape (height, width) with values 0-15

        Returns:
            probs: numpy array of shape (6,) with change probabilities
        """
        # Convert to one-hot encoding
        one_hot = self._frame_to_onehot(frame)

        # Add batch dimension and convert to tensor
        tensor = torch.from_numpy(one_hot).float().unsqueeze(0)  # (1, 16, H, W)

        # Resize to expected input size if needed
        if tensor.shape[-2:] != (self.grid_size, self.grid_size):
            tensor = F.interpolate(tensor, size=(self.grid_size, self.grid_size),
                                 mode='nearest')

        # Forward pass
        with torch.no_grad():
            logits = self.forward(tensor)  # (1, 6)
            probs = torch.sigmoid(logits).squeeze(0).numpy()  # (6,)

        return probs

    def _frame_to_onehot(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to one-hot encoding.

        Args:
            frame: (H, W) array with values 0-15

        Returns:
            one_hot: (16, H, W) array
        """
        H, W = frame.shape
        one_hot = np.zeros((16, H, W), dtype=np.float32)

        for i in range(16):
            one_hot[i] = (frame == i).astype(np.float32)

        return one_hot


class ChangePredictorTrainer:
    """
    Trainer for the ChangePredictor model.
    """

    def __init__(self, model: ChangePredictor, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        # Experience buffer for training
        self.experience_buffer = []
        self.max_buffer_size = 1000

    def add_experience(self, frame: np.ndarray, action_idx: int, changed: bool):
        """
        Add experience to training buffer.

        Args:
            frame: The input frame
            action_idx: Action taken (0-5)
            changed: Whether the action changed the frame
        """
        # Create target vector (0s except for the action taken)
        target = np.zeros(6, dtype=np.float32)
        target[action_idx] = 1.0 if changed else 0.0

        experience = {
            'frame': frame.copy(),
            'target': target
        }

        self.experience_buffer.append(experience)

        # Keep buffer size manageable
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            loss: Training loss if sufficient data available, None otherwise
        """
        if len(self.experience_buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)

        frames = []
        targets = []

        for idx in indices:
            exp = self.experience_buffer[idx]
            frames.append(self.model._frame_to_onehot(exp['frame']))
            targets.append(exp['target'])

        # Convert to tensors
        frames_tensor = torch.from_numpy(np.stack(frames)).float()  # (batch, 16, H, W)
        targets_tensor = torch.from_numpy(np.stack(targets)).float()  # (batch, 6)

        # Resize frames if needed
        if frames_tensor.shape[-2:] != (self.model.grid_size, self.model.grid_size):
            frames_tensor = F.interpolate(frames_tensor,
                                        size=(self.model.grid_size, self.model.grid_size),
                                        mode='nearest')

        # Forward pass
        logits = self.model(frames_tensor)
        loss = self.criterion(logits, targets_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()