"""
ReCoN ARC-1 Neural Models

Implements neural components adapted for ReCoN architecture.
Faithful reproduction of BlindSquirrel's neural models as ReCoN terminals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models as torchvision_models
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class ReCoNArc1ActionModel(nn.Module):
    """
    Action value model adapted for ReCoN Neural Terminal.

    Faithful reproduction of BlindSquirrel's ResNet-18 architecture.
    """

    def __init__(self, game_id: str = "default"):
        super().__init__()
        self.game_id = game_id

        # Grid symbol embedding (16 colors -> 16-dim embedding)
        self.grid_symbol_embedding = nn.Embedding(16, 16)

        # Custom stem for 16-channel input
        self.stem = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ResNet-18 backbone (replace initial layers)
        weights = torchvision_models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision_models.resnet18(weights=weights)
        backbone.conv1 = nn.Identity()
        backbone.bn1 = nn.Identity()
        backbone.relu = nn.Identity()
        backbone.maxpool = nn.Identity()

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # State processing head
        self.state_fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 64),
            nn.ReLU(inplace=True)
        )

        # Action processing head (26-dim action encoding)
        self.action_fc = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(inplace=True)
        )

        # Final value prediction head
        self.head_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0,1] range for ReCoN compatibility
        )

    def forward(self, state, action):
        """Forward pass matching original BlindSquirrel architecture."""
        # Embed grid symbols
        x = self.grid_symbol_embedding(state)
        x = x.permute(0, 3, 1, 2)  # (B, 64, 64, 16) -> (B, 16, 64, 64)

        # Custom stem
        x = self.stem(x)

        # ResNet-18 backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # State processing
        x = self.state_fc(x)

        # Action processing
        x_a = self.action_fc(action)

        # Combine state and action features
        x = torch.cat([x, x_a], dim=1)

        # Final value prediction
        x = self.head_fc(x)

        return x


class ActionModelDataset(Dataset):
    """Dataset for training the ActionModel."""

    def __init__(self, examples: List[Tuple]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, action_idx, target = self.examples[idx]

        # Convert state frame to tensor
        if isinstance(state.frame, str):
            # Handle special states
            grid = torch.zeros((64, 64), dtype=torch.long)
        else:
            grid = torch.tensor(state.frame, dtype=torch.long)

        # Encode action
        action_encoder = ActionEncoder()
        action_encoding = action_encoder.encode_action(state, action_idx)

        return {
            "state": grid,
            "action": action_encoding,
            "score": torch.tensor([target], dtype=torch.float32)
        }

    def collate(self, batch):
        """Collate function for DataLoader."""
        state = torch.stack([b["state"] for b in batch], dim=0)
        action = torch.stack([b["action"] for b in batch], dim=0)
        score = torch.stack([b["score"] for b in batch], dim=0)
        return {"state": state, "action": action, "score": score}


class ReCoNArc1Trainer:
    """Training utilities for ReCoN ARC-1 models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training hyperparameters (from original BlindSquirrel)
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 10
        self.max_train_time = 15

    def train(self, training_data: List[Tuple], action_encoder: 'ActionEncoder') -> Optional[ReCoNArc1ActionModel]:
        """Train a new model on provided data."""
        if not training_data or len(training_data) < 10:
            return None

        # Create model
        model = ReCoNArc1ActionModel()
        model.to(self.device)

        # Setup training
        dataset = ActionModelDataset(training_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                # Move to device
                states = batch["state"].to(self.device)
                actions = batch["action"].to(self.device)
                targets = batch["score"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                predictions = model(states, actions)
                loss = criterion(predictions, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"ReCoN ARC-1 Training | Epoch {epoch+1}/{self.num_epochs} | Loss: {avg_loss:.4f}")

        model.eval()
        return model


class ActionEncoder:
    """
    Encodes actions into 26-dimensional tensors matching BlindSquirrel format.

    Action encoding structure:
    - 6 dimensions for action type (ACTION1-5, CLICK)
    - 16 dimensions for color (one-hot)
    - 4 dimensions for object features (regularity, size, y_centroid, x_centroid)
    """

    @staticmethod
    def encode_action(state, action_idx: int) -> torch.Tensor:
        """Encode action index into 26-dimensional tensor."""
        # Initialize encoding components
        action_type = torch.zeros(6)
        colour = torch.zeros(16)
        regularity = torch.zeros(1)
        size = torch.zeros(1)
        y_centroid = torch.zeros(1)
        x_centroid = torch.zeros(1)

        if action_idx <= 4:
            # Basic actions (ACTION1-5)
            action_type[action_idx] = 1
            regularity[0] = 1
            size[0] = 1
            y_centroid[0] = -1
            x_centroid[0] = -1
        else:
            # Click action (ACTION6)
            action_type[5] = 1
            object_idx = action_idx - 5

            if hasattr(state, 'object_data') and object_idx < len(state.object_data):
                action_obj = state.object_data[object_idx]
                colour[action_obj['colour']] = 1
                regularity[0] = action_obj['regularity']
                size[0] = action_obj['size']
                y_centroid[0] = action_obj['y_centroid'] / 64.0  # Normalize to [0,1]
                x_centroid[0] = action_obj['x_centroid'] / 64.0  # Normalize to [0,1]
            else:
                # Fallback for invalid click action
                regularity[0] = 0.5
                size[0] = 0.5
                y_centroid[0] = 0.5  # Center of grid (normalized)
                x_centroid[0] = 0.5

        # Combine all components
        combined = torch.cat([
            action_type, colour, regularity, size, y_centroid, x_centroid
        ])

        return combined.float()

    @staticmethod
    def decode_action_type(action_tensor: torch.Tensor) -> int:
        """Decode action type from tensor."""
        action_type_part = action_tensor[:6]
        return torch.argmax(action_type_part).item()

    @staticmethod
    def get_action_features(action_tensor: torch.Tensor) -> Dict[str, float]:
        """Extract action features from encoded tensor."""
        action_type = torch.argmax(action_tensor[:6]).item()
        colour = torch.argmax(action_tensor[6:22]).item() if action_type == 5 else -1
        regularity = action_tensor[22].item()
        size = action_tensor[23].item()
        y_centroid = action_tensor[24].item()
        x_centroid = action_tensor[25].item()

        return {
            'action_type': action_type,
            'colour': colour,
            'regularity': regularity,
            'size': size,
            'y_centroid': y_centroid,
            'x_centroid': x_centroid
        }


class ReCoNArc1NeuralTerminal:
    """
    Neural Terminal implementation for ReCoN ARC-1.

    Wraps the action model as a ReCoN neural terminal with measurement function.
    """

    def __init__(self, model: Optional[ReCoNArc1ActionModel] = None):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model:
            self.model.to(self.device)

    def measurement_fn(self, state_data: Any, action_data: Any) -> float:
        """
        Measurement function for ReCoN Neural Terminal.

        Returns action value in [0,1] range suitable for ReCoN.
        """
        if not self.model:
            return 0.5  # Default neutral value

        try:
            # Convert state to tensor
            if hasattr(state_data, 'frame') and isinstance(state_data.frame, tuple):
                grid = torch.tensor(state_data.frame, dtype=torch.long)
            else:
                grid = torch.zeros((64, 64), dtype=torch.long)

            # Encode action
            action_encoder = ActionEncoder()
            action_encoding = action_encoder.encode_action(state_data, action_data)

            # Add batch dimensions
            grid = grid.unsqueeze(0).to(self.device)
            action_encoding = action_encoding.unsqueeze(0).to(self.device)

            # Predict value
            self.model.eval()
            with torch.no_grad():
                value = self.model(grid, action_encoding)
                return float(value.item())

        except Exception as e:
            print(f"Error in neural terminal measurement: {e}")
            return 0.5

    def update_model(self, new_model: ReCoNArc1ActionModel):
        """Update the underlying model."""
        self.model = new_model
        if self.model:
            self.model.to(self.device)