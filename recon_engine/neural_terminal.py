"""
Neural Terminal Node Implementation

Wraps PyTorch models as ReCoN terminal nodes.
Supports different model types (CNN, ResNet, etc.) and output modes.
"""

from typing import Dict, Any, Union, List, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .hybrid_node import HybridReCoNNode, NodeMode
from .messages import ReCoNMessage


class NeuralOutputMode:
    """Output processing modes for neural terminals."""
    VALUE = "value"              # Single value prediction (BlindSquirrel style)
    PROBABILITY = "probability"  # Probability distribution (For specialized CNN terminal)
    CLASSIFICATION = "classification"  # Class prediction
    EMBEDDING = "embedding"      # Feature embedding
    MULTI_HEAD = "multi_head"    # Multiple outputs


class NeuralTerminal(HybridReCoNNode):
    """
    Terminal node that wraps a PyTorch neural network.
    
    Integrates neural models into ReCoN graphs while maintaining
    theoretical correctness of message passing.
    """
    
    def __init__(self, node_id: str, model: nn.Module, 
                 output_mode: str = NeuralOutputMode.VALUE,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 threshold: float = 0.8):
        super().__init__(node_id, "terminal", NodeMode.NEURAL)
        
        self.model = model
        self.output_mode = output_mode
        self.input_shape = input_shape
        self.threshold = threshold
        
        # Cache for avoiding redundant computation
        self._input_cache = {}
        self._output_cache = {}
        
        # Input preprocessing
        self.input_normalizer = None
        self.input_transformer = None
        
        # Output postprocessing
        self.output_transformer = None
        self.confidence_estimator = None
        
        # Model metadata for visualization
        self.model_info = self._extract_model_info()
        
    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract model information for visualization."""
        info = {
            "model_type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": str(next(self.model.parameters()).device) if list(self.model.parameters()) else "cpu"
        }
        
        # Try to extract architecture details
        if hasattr(self.model, 'features'):
            info["has_features"] = True
        if hasattr(self.model, 'classifier'):
            info["has_classifier"] = True
        if hasattr(self.model, 'conv1'):
            info["is_cnn"] = True
            
        return info
    
    def set_input_preprocessing(self, normalizer: Optional[Callable] = None,
                              transformer: Optional[Callable] = None):
        """Set input preprocessing functions."""
        self.input_normalizer = normalizer
        self.input_transformer = transformer
    
    def set_output_postprocessing(self, transformer: Optional[Callable] = None,
                                confidence_estimator: Optional[Callable] = None):
        """Set output postprocessing functions."""
        self.output_transformer = transformer
        self.confidence_estimator = confidence_estimator
    
    def process_messages(self, messages: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Process messages through neural network.
        
        For terminal nodes, we primarily receive 'sub' requests
        and respond with 'sur' confirmations/failures.
        """
        # Check for request (sub message)
        if not messages.get("sub"):
            return {}
        
        # Extract environment/input from sub message
        sub_messages = messages["sub"]
        environment = sub_messages[-1] if sub_messages else None
        
        # Perform neural measurement
        try:
            measurement = self.measure(environment)
            return self._process_measurement(measurement)
        except Exception as e:
            print(f"Neural terminal {self.id} failed: {e}")
            return {"sur": "fail"}
    
    def measure(self, environment: Any = None) -> Union[float, torch.Tensor]:
        """
        Perform neural measurement.
        
        Converts environment to model input and runs inference.
        """
        if environment is None:
            return 0.0
        
        # Check cache first
        cache_key = self._get_cache_key(environment)
        if cache_key in self._output_cache:
            return self._output_cache[cache_key]
        
        # Convert environment to model input
        model_input = self._prepare_input(environment)
        
        # Run model inference
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(model_input)
        
        # Process output based on mode
        processed_output = self._process_output(raw_output)
        
        # Cache result
        self._output_cache[cache_key] = processed_output
        
        return processed_output
    
    def _prepare_input(self, environment: Any) -> torch.Tensor:
        """Convert environment to model input tensor."""
        # Convert to tensor
        if isinstance(environment, torch.Tensor):
            input_tensor = environment
        elif isinstance(environment, np.ndarray):
            input_tensor = torch.from_numpy(environment).float()
        elif isinstance(environment, (list, tuple)):
            input_tensor = torch.tensor(environment).float()
        elif hasattr(environment, 'to_tensor'):
            input_tensor = environment.to_tensor()
        else:
            # Try to extract features from object
            input_tensor = self._extract_features(environment)
        
        # Apply input preprocessing
        if self.input_normalizer:
            input_tensor = self.input_normalizer(input_tensor)
        if self.input_transformer:
            input_tensor = self.input_transformer(input_tensor)
        
        # Ensure correct device
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Add batch dimension if needed
        if len(input_tensor.shape) == len(self.input_shape or []):
            input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    def _extract_features(self, environment: Any) -> torch.Tensor:
        """Extract features from unknown environment type."""
        # Default feature extraction for various types
        if hasattr(environment, '__dict__'):
            # Object with attributes - extract numeric values
            values = []
            for key, value in environment.__dict__.items():
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, (list, tuple)):
                    values.extend([float(x) for x in value if isinstance(x, (int, float))])
            return torch.tensor(values if values else [0.0])
        else:
            # Fallback to zero tensor
            return torch.tensor([0.0])
    
    def _process_output(self, raw_output: torch.Tensor) -> Union[float, torch.Tensor]:
        """Process model output based on output mode."""
        if self.output_mode == NeuralOutputMode.VALUE:
            # Single value (e.g., BlindSquirrel value prediction)
            if raw_output.numel() == 1:
                return raw_output.item()
            else:
                # Multiple outputs - take mean or max
                return raw_output.mean().item()
        
        elif self.output_mode == NeuralOutputMode.PROBABILITY:
            # Probability distribution (e.g., specialized CNN terminal action probabilities)
            if raw_output.dim() > 1:
                # Apply softmax to get probabilities
                probabilities = F.softmax(raw_output, dim=-1)
                return probabilities.squeeze()
            else:
                # Single output - apply sigmoid
                return torch.sigmoid(raw_output)
        
        elif self.output_mode == NeuralOutputMode.CLASSIFICATION:
            # Classification - return class probabilities
            return F.softmax(raw_output, dim=-1)
        
        elif self.output_mode == NeuralOutputMode.EMBEDDING:
            # Feature embedding - return as-is
            return raw_output.squeeze()
        
        elif self.output_mode == NeuralOutputMode.MULTI_HEAD:
            # Multiple heads - return full tensor
            return raw_output
        
        else:
            # Default - return first element
            return raw_output.flatten()[0].item()
    
    def _process_measurement(self, measurement: Union[float, torch.Tensor]) -> Dict[str, Any]:
        """Convert measurement to ReCoN messages."""
        if isinstance(measurement, torch.Tensor):
            if measurement.numel() == 1:
                # Single value measurement
                value = measurement.item()
                if value >= self.threshold:
                    return {"sur": "confirm", "activation": value}
                else:
                    return {"sur": "fail", "activation": value}
            else:
                # Vector measurement - use in implicit mode
                self.activation = measurement
                # Check if any element exceeds threshold
                if (measurement >= self.threshold).any():
                    return {"sur": "confirm", "activation": measurement}
                else:
                    return {"sur": "fail", "activation": measurement}
        else:
            # Scalar measurement
            if measurement >= self.threshold:
                return {"sur": "confirm", "activation": measurement}
            else:
                return {"sur": "fail", "activation": measurement}
    
    def _get_cache_key(self, environment: Any) -> str:
        """Generate cache key for environment."""
        if isinstance(environment, torch.Tensor):
            # Handle CUDA tensors by moving to CPU first
            tensor_cpu = environment.detach().cpu()
            return str(tensor_cpu.numpy().tobytes())
        elif isinstance(environment, np.ndarray):
            return str(environment.tobytes())
        elif isinstance(environment, (list, tuple)):
            return str(tuple(environment))
        else:
            return str(hash(str(environment)))
    
    def clear_cache(self):
        """Clear input/output caches."""
        self._input_cache.clear()
        self._output_cache.clear()
    
    def train_mode(self, mode: bool = True):
        """Set model training mode."""
        self.model.train(mode)
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def to_device(self, device: Union[str, torch.device]):
        """Move model to device."""
        self.model = self.model.to(device)
        self.model_info["device"] = str(device)
    
    def get_model_summary(self) -> str:
        """Get human-readable model summary."""
        summary = f"Model: {self.model_info['model_type']}\n"
        summary += f"Parameters: {self.model_info['parameters']:,}\n"
        summary += f"Trainable: {self.model_info['trainable_parameters']:,}\n"
        summary += f"Device: {self.model_info['device']}\n"
        summary += f"Output mode: {self.output_mode}\n"
        summary += f"Threshold: {self.threshold}\n"
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Export node configuration for visualization."""
        base_dict = super().to_dict()
        base_dict.update({
            "model_info": self.model_info,
            "output_mode": self.output_mode,
            "threshold": self.threshold,
            "input_shape": self.input_shape,
            "cache_size": len(self._output_cache)
        })
        return base_dict


class ResNetActionValueTerminal(NeuralTerminal):
    """
    Specialized terminal for BlindSquirrel-style value prediction.
    
    Uses ResNet architecture to predict action values.
    """
    
    def __init__(self, node_id: str, game_id: str = "default"):
        # Create ResNet-based value model (simplified version)
        model = self._create_value_model()
        super().__init__(node_id, model, NeuralOutputMode.VALUE, (64, 64, 16))
        self.game_id = game_id
    
    def _create_value_model(self) -> nn.Module:
        """Create ResNet-based value prediction model."""
        return ValuePredictionModel()
    
    def _prepare_input(self, environment: Any) -> torch.Tensor:
        """Prepare state and action for value prediction."""
        # Expected input: (state_tensor, action_tensor)
        if isinstance(environment, tuple) and len(environment) == 2:
            state, action = environment
            
            # Convert state to tensor if needed
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state).long()
            
            # Convert action to tensor if needed  
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action).float()
            
            # Combine state and action for model input
            return super()._prepare_input((state, action))
        else:
            return super()._prepare_input(environment)


class CNNValidActionTerminal(NeuralTerminal):
    """
    Specialized terminal for action prediction.
    
    Uses CNN architecture to predict action probabilities.
    """
    
    def __init__(self, node_id: str, use_gpu: bool = True):
        # Create CNN-based action model
        model = self._create_action_model()
        super().__init__(node_id, model, NeuralOutputMode.PROBABILITY, (16, 64, 64))
        
        # Move to GPU if available and requested
        if use_gpu and torch.cuda.is_available():
            self.to_device('cuda')
            print(f"CNNValidActionTerminal {node_id} using GPU: {next(self.model.parameters()).device}")
    
    def _create_action_model(self) -> nn.Module:
        """Create CNN-based action prediction model."""
        return ActionPredictionModel()
    
    def _process_measurement(self, measurement: torch.Tensor) -> Dict[str, Any]:
        """Process action probabilities for hierarchical sampling."""
        if measurement.numel() == 4101:  # 5 actions + 4096 coordinates
            action_probs = measurement[:5]
            coord_probs = measurement[5:].reshape(64, 64)
            
            return {
                "sur": "confirm",
                "action_probabilities": action_probs,
                "coordinate_probabilities": coord_probs,
                "activation": measurement
            }
        else:
            return super()._process_measurement(measurement)


class ValuePredictionModel(nn.Module):
    """Simplified ResNet-based value prediction model."""
    
    def __init__(self):
        super().__init__()
        # Grid symbol embedding
        self.grid_embedding = nn.Embedding(16, 16)
        
        # Simplified ResNet layers
        self.conv1 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Action embedding
        self.action_embedding = nn.Linear(10, 64)  # Placeholder action dim
        
        # Combined processing
        self.fc1 = nn.Linear(256 * 8 * 8 + 64, 512)
        self.fc2 = nn.Linear(512, 64)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, x):
        if isinstance(x, tuple):
            state, action = x
            
            # Process state (64x64 grid)
            if state.dtype == torch.long:
                state = self.grid_embedding(state).permute(0, 3, 1, 2)
            
            # CNN processing
            h = F.relu(self.conv1(state))
            h = F.max_pool2d(h, 2)
            h = F.relu(self.conv2(h))
            h = F.max_pool2d(h, 2)
            h = F.relu(self.conv3(h))
            h = F.max_pool2d(h, 2)
            h = h.flatten(1)
            
            # Process action
            action_h = F.relu(self.action_embedding(action))
            
            # Combine
            combined = torch.cat([h, action_h], dim=1)
            combined = F.relu(self.fc1(combined))
            combined = F.relu(self.fc2(combined))
            
            return self.value_head(combined)
        else:
            # Fallback for single input
            return torch.tensor([0.0])


class ActionPredictionModel(nn.Module):
    """Simplified CNN-based action prediction model."""
    
    def __init__(self):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Action head (5 actions)
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, 5)
        
        # Coordinate head (64x64 grid)
        self.coord_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, 1)
        self.coord_conv4 = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        # Shared backbone
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        
        # Action head
        action_h = self.action_pool(h)
        action_h = action_h.flatten(1)
        action_h = F.relu(self.action_fc(action_h))
        action_logits = self.action_head(action_h)
        
        # Coordinate head
        coord_h = F.relu(self.coord_conv1(h))
        coord_h = F.relu(self.coord_conv2(coord_h))
        coord_h = F.relu(self.coord_conv3(coord_h))
        coord_logits = self.coord_conv4(coord_h).flatten(1)
        
        # Combine outputs
        combined_logits = torch.cat([action_logits, coord_logits], dim=1)
        return combined_logits