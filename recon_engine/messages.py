"""
ReCoN Message Types and Definitions

Enhanced message passing semantics supporting both discrete and continuous values.
Maintains compatibility with Table 1 of the ReCoN paper while enabling hybrid modes.
"""

from enum import Enum
from typing import Dict, Any, Union, Optional
import torch
import numpy as np


class MessageType(Enum):
    """Message types passed between ReCoN nodes."""
    INHIBIT_REQUEST = "inhibit_request"
    INHIBIT_CONFIRM = "inhibit_confirm" 
    WAIT = "wait"
    CONFIRM = "confirm"
    FAIL = "fail"
    REQUEST = "request"
    # Extended types for hybrid mode
    CONTINUOUS = "continuous"
    PROBABILITY = "probability"
    EMBEDDING = "embedding"


class HybridMessage:
    """
    Enhanced message supporting both discrete and continuous values.
    
    Automatically converts between discrete ReCoN messages and continuous
    neural network activations while preserving theoretical correctness.
    """
    
    def __init__(self, 
                 value: Union[str, float, torch.Tensor, MessageType],
                 source_node: str,
                 target_node: str,
                 link_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 threshold: float = 0.8):
        
        self.source = source_node
        self.target = target_node
        self.link_type = link_type
        self.threshold = threshold
        self.metadata = metadata or {}
        
        # Store both representations
        self._discrete_value = None
        self._continuous_value = None
        
        # Parse input value
        if isinstance(value, str):
            self._discrete_value = value
            self._continuous_value = self._discrete_to_continuous(value)
        elif isinstance(value, MessageType):
            self._discrete_value = value.value
            self._continuous_value = self._discrete_to_continuous(value.value)
        elif isinstance(value, (int, float)):
            self._continuous_value = float(value)
            self._discrete_value = self._continuous_to_discrete(value)
        elif isinstance(value, torch.Tensor):
            self._continuous_value = value
            self._discrete_value = self._tensor_to_discrete(value)
        elif isinstance(value, np.ndarray):
            self._continuous_value = torch.from_numpy(value).float()
            self._discrete_value = self._tensor_to_discrete(self._continuous_value)
        else:
            # Fallback
            self._continuous_value = 0.0
            self._discrete_value = "wait"
    
    @property
    def discrete(self) -> str:
        """Get discrete message representation."""
        return self._discrete_value
    
    @property
    def continuous(self) -> Union[float, torch.Tensor]:
        """Get continuous value representation."""
        return self._continuous_value
    
    @property
    def is_tensor(self) -> bool:
        """Check if continuous value is a tensor."""
        return isinstance(self._continuous_value, torch.Tensor)
    
    @property
    def magnitude(self) -> float:
        """Get magnitude of the message for comparison."""
        if isinstance(self._continuous_value, torch.Tensor):
            return self._continuous_value.abs().max().item()
        else:
            return abs(self._continuous_value)
    
    def as_activation(self) -> Union[float, torch.Tensor]:
        """Get message as activation value for neural processing."""
        return self._continuous_value
    
    def as_message_type(self) -> MessageType:
        """Get message as discrete MessageType."""
        try:
            return MessageType(self._discrete_value)
        except ValueError:
            # Fallback for non-standard discrete values
            return MessageType.WAIT
    
    def _discrete_to_continuous(self, discrete: str) -> float:
        """Convert discrete message to continuous value."""
        mapping = {
            "confirm": 1.0,
            "inhibit_confirm": 1.0,
            "request": 0.5,
            "wait": 0.0,
            "fail": -1.0,
            "inhibit_request": -1.0
        }
        return mapping.get(discrete, 0.0)
    
    def _continuous_to_discrete(self, continuous: Union[float, torch.Tensor]) -> str:
        """Convert continuous value to discrete message."""
        if isinstance(continuous, torch.Tensor):
            value = continuous.item() if continuous.numel() == 1 else continuous.max().item()
        else:
            value = continuous
        
        if value >= self.threshold:
            return "confirm"
        elif value <= -self.threshold:
            return "fail"
        elif value > 0:
            return "wait"
        else:
            return "wait"
    
    def _tensor_to_discrete(self, tensor: torch.Tensor) -> str:
        """Convert tensor to discrete message based on thresholding."""
        if tensor.numel() == 1:
            value = tensor.item()
        else:
            # For multi-dimensional tensors, use max activation
            value = tensor.max().item()
        
        return self._continuous_to_discrete(value)
    
    def to_explicit_mode(self) -> str:
        """Convert to explicit mode (discrete string)."""
        return self.discrete
    
    def to_implicit_mode(self) -> Union[float, torch.Tensor]:
        """Convert to implicit mode (continuous value)."""
        return self.continuous
    
    def threshold_at(self, threshold: float) -> 'HybridMessage':
        """Create new message with different threshold."""
        new_msg = HybridMessage(
            self._continuous_value,
            self.source,
            self.target, 
            self.link_type,
            self.metadata.copy(),
            threshold
        )
        return new_msg
    
    def clone(self) -> 'HybridMessage':
        """Create a copy of this message."""
        return HybridMessage(
            self._continuous_value,
            self.source,
            self.target,
            self.link_type, 
            self.metadata.copy(),
            self.threshold
        )
    
    def __repr__(self):
        cont_repr = f"{self._continuous_value:.3f}" if isinstance(self._continuous_value, float) else f"tensor({self._continuous_value.shape})"
        return f"HybridMessage({self._discrete_value}|{cont_repr}, {self.source}->{self.target} via {self.link_type})"
    
    def __str__(self):
        return self._discrete_value
    
    def __float__(self):
        if isinstance(self._continuous_value, torch.Tensor):
            return self._continuous_value.item() if self._continuous_value.numel() == 1 else self._continuous_value.max().item()
        return float(self._continuous_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "discrete": self._discrete_value,
            "continuous": self._continuous_value.tolist() if isinstance(self._continuous_value, torch.Tensor) else self._continuous_value,
            "source": self.source,
            "target": self.target,
            "link_type": self.link_type,
            "threshold": self.threshold,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HybridMessage':
        """Create from dictionary."""
        continuous = data["continuous"]
        if isinstance(continuous, list):
            continuous = torch.tensor(continuous)
        
        msg = cls(
            continuous,
            data["source"],
            data["target"], 
            data["link_type"],
            data.get("metadata", {}),
            data.get("threshold", 0.8)
        )
        # Override discrete value if provided
        if "discrete" in data:
            msg._discrete_value = data["discrete"]
        
        return msg


class ReCoNMessage:
    """
    Legacy ReCoN message for backward compatibility.
    
    Maintains original discrete semantics while supporting conversion
    to hybrid messages when needed.
    """
    
    def __init__(self, 
                 message_type: MessageType,
                 source_node: str,
                 target_node: str,
                 link_type: str,
                 activation: Union[float, torch.Tensor] = 0.0):
        self.type = message_type
        self.source = source_node
        self.target = target_node
        self.link_type = link_type
        self.activation = activation
        
    def to_hybrid(self) -> HybridMessage:
        """Convert to hybrid message."""
        return HybridMessage(
            self.type.value,
            self.source,
            self.target,
            self.link_type
        )
        
    def __repr__(self):
        return f"ReCoNMessage({self.type.value}, {self.source}->{self.target} via {self.link_type})"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "source": self.source,
            "target": self.target, 
            "link_type": self.link_type,
            "activation": self.activation.tolist() if isinstance(self.activation, torch.Tensor) else self.activation
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReCoNMessage':
        """Create from dictionary."""
        activation = data["activation"]
        if isinstance(activation, list):
            activation = torch.tensor(activation)
            
        return cls(
            MessageType(data["type"]),
            data["source"], 
            data["target"],
            data["link_type"],
            activation
        )


# Utility functions for message conversion
def auto_convert_message(message: Any, target_mode: str = "auto") -> Union[HybridMessage, ReCoNMessage]:
    """
    Automatically convert various message types to appropriate format.
    
    Args:
        message: Input message (string, number, tensor, etc.)
        target_mode: 'explicit', 'implicit', 'hybrid', or 'auto'
    
    Returns:
        Converted message object
    """
    if isinstance(message, (HybridMessage, ReCoNMessage)):
        return message
    
    if target_mode == "hybrid" or target_mode == "auto":
        return HybridMessage(message, "unknown", "unknown", "unknown")
    elif target_mode == "explicit":
        if isinstance(message, str):
            return ReCoNMessage(MessageType(message), "unknown", "unknown", "unknown")
        else:
            # Convert to discrete first
            hybrid = HybridMessage(message, "unknown", "unknown", "unknown")
            return ReCoNMessage(hybrid.as_message_type(), "unknown", "unknown", "unknown")
    else:
        return HybridMessage(message, "unknown", "unknown", "unknown")


def ensure_compatible(msg1: Any, msg2: Any) -> tuple:
    """
    Ensure two messages are compatible for processing.
    
    Returns tuple of converted messages in compatible format.
    """
    # Convert both to hybrid format
    h1 = auto_convert_message(msg1, "hybrid")
    h2 = auto_convert_message(msg2, "hybrid")
    
    return h1, h2