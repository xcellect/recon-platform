"""
Hybrid ReCoN Node Implementation

Supports both explicit state machines and implicit activation levels.
Enables mode switching to support different agent architectures.
"""

from enum import Enum
from typing import Dict, Any, Union, List, Optional, Callable
import torch
import numpy as np
from .node import ReCoNNode, ReCoNState
# Import CompactReCoNNode when needed to avoid circular imports
from .messages import MessageType, ReCoNMessage


class NodeMode(Enum):
    """Supported node operation modes."""
    EXPLICIT = "explicit"      # Discrete state machine
    IMPLICIT = "implicit"      # Continuous activation levels  
    NEURAL = "neural"          # Neural network terminal
    HYBRID = "hybrid"          # Can switch between modes


class HybridReCoNNode(ReCoNNode):
    """
    Hybrid ReCoN node supporting multiple operation modes.
    
    Can switch between:
    - Explicit: Discrete state machine (BlindSquirrel style)
    - Implicit: Continuous activations (StochasticGoose style)
    - Neural: Neural network terminal
    - Hybrid: Dynamic mode switching based on context
    """
    
    def __init__(self, node_id: str, node_type: str = "script", mode: NodeMode = NodeMode.EXPLICIT):
        # Initialize all properties first to avoid issues with property setters
        self.mode = mode
        self._explicit_state = ReCoNState.INACTIVE
        self._implicit_activation = 0.0
        self.activation_dim = 1
        self.mode_history = [mode]
        self.switch_threshold = 0.8
        
        # Neural components (if applicable)
        self.neural_model = None
        self.input_processor = None
        self.output_processor = None
        
        # Message conversion functions
        self.discrete_to_continuous = self._default_d2c
        self.continuous_to_discrete = self._default_c2d
        
        # Now call super().__init__
        super().__init__(node_id, node_type)
        
    def set_mode(self, mode: NodeMode, preserve_state: bool = True):
        """
        Switch node operation mode.
        
        Args:
            mode: New operation mode
            preserve_state: Whether to preserve current state when switching
        """
        old_mode = self.mode
        self.mode = mode
        self.mode_history.append(mode)
        
        if preserve_state:
            self._convert_state(old_mode, mode)
    
    def _convert_state(self, from_mode: NodeMode, to_mode: NodeMode):
        """Convert state representation between modes."""
        if from_mode == NodeMode.EXPLICIT and to_mode == NodeMode.IMPLICIT:
            # Convert discrete state to activation level
            if self._explicit_state == ReCoNState.CONFIRMED:
                self._implicit_activation = 1.0
            elif self._explicit_state == ReCoNState.FAILED:
                self._implicit_activation = -1.0
            elif self._explicit_state == ReCoNState.ACTIVE:
                self._implicit_activation = 0.5
            else:
                self._implicit_activation = 0.0
                
        elif from_mode == NodeMode.IMPLICIT and to_mode == NodeMode.EXPLICIT:
            # Convert activation level to discrete state
            if self._implicit_activation >= self.switch_threshold:
                self._explicit_state = ReCoNState.CONFIRMED
            elif self._implicit_activation <= -self.switch_threshold:
                self._explicit_state = ReCoNState.FAILED
            elif self._implicit_activation > 0:
                self._explicit_state = ReCoNState.ACTIVE
            else:
                self._explicit_state = ReCoNState.INACTIVE
    
    @property
    def state(self) -> ReCoNState:
        """Get current state based on mode."""
        if self.mode == NodeMode.EXPLICIT:
            return self._explicit_state
        elif self.mode == NodeMode.IMPLICIT:
            # Derive state from activation level
            if isinstance(self._implicit_activation, torch.Tensor):
                activation = self._implicit_activation.item() if self._implicit_activation.numel() == 1 else self._implicit_activation.max().item()
            else:
                activation = self._implicit_activation
                
            if activation >= self.switch_threshold:
                return ReCoNState.CONFIRMED
            elif activation <= -self.switch_threshold:
                return ReCoNState.FAILED
            elif activation > 0:
                return ReCoNState.ACTIVE
            else:
                return ReCoNState.INACTIVE
        else:
            return self._explicit_state
    
    @state.setter
    def state(self, value: ReCoNState):
        """Set state based on mode."""
        if self.mode == NodeMode.EXPLICIT:
            self._explicit_state = value
        elif self.mode == NodeMode.IMPLICIT:
            # Convert discrete state to activation
            if value == ReCoNState.CONFIRMED:
                self._implicit_activation = 1.0
            elif value == ReCoNState.FAILED:
                self._implicit_activation = -1.0
            elif value == ReCoNState.ACTIVE:
                self._implicit_activation = 0.5
            else:
                self._implicit_activation = 0.0
    
    @property
    def activation(self) -> Union[float, torch.Tensor]:
        """Get current activation based on mode."""
        if self.mode == NodeMode.IMPLICIT:
            return self._implicit_activation
        elif self.mode == NodeMode.EXPLICIT:
            # Convert discrete state to activation
            if self._explicit_state == ReCoNState.CONFIRMED:
                return 1.0
            elif self._explicit_state == ReCoNState.FAILED:
                return -1.0
            elif self._explicit_state == ReCoNState.ACTIVE:
                return 0.5
            else:
                return 0.0
        else:
            return self._implicit_activation
    
    @activation.setter  
    def activation(self, value: Union[float, torch.Tensor]):
        """Set activation based on mode."""
        if self.mode == NodeMode.IMPLICIT:
            self._implicit_activation = value
        elif self.mode == NodeMode.EXPLICIT:
            # Convert activation to discrete state
            if isinstance(value, torch.Tensor):
                val = value.item() if value.numel() == 1 else value.max().item()
            else:
                val = value
                
            if val >= self.switch_threshold:
                self._explicit_state = ReCoNState.CONFIRMED
            elif val <= -self.switch_threshold:
                self._explicit_state = ReCoNState.FAILED
            elif val > 0:
                self._explicit_state = ReCoNState.ACTIVE
            else:
                self._explicit_state = ReCoNState.INACTIVE
    
    def process_messages(self, messages: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Process incoming messages based on current mode.
        
        Args:
            messages: Incoming messages by link type
            
        Returns:
            Outgoing messages to send
        """
        if self.mode == NodeMode.EXPLICIT:
            return self._process_explicit(messages)
        elif self.mode == NodeMode.IMPLICIT:
            return self._process_implicit(messages)
        elif self.mode == NodeMode.NEURAL:
            return self._process_neural(messages)
        elif self.mode == NodeMode.HYBRID:
            return self._process_hybrid(messages)
        else:
            raise ValueError(f"Unknown node mode: {self.mode}")
    
    def _process_explicit(self, messages: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Process messages using explicit state machine."""
        # Use parent class explicit processing
        return super().process_messages(messages)
    
    def _process_implicit(self, messages: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Process messages using implicit activation levels."""
        # Convert messages to continuous values
        z = {}
        for link_type, msg_list in messages.items():
            if msg_list:
                # Take most recent message
                msg = msg_list[-1]
                z[link_type] = self._message_to_activation(msg)
            else:
                z[link_type] = 0.0
        
        # Apply compact rules (similar to CompactReCoNNode)
        return self._apply_compact_rules(z)
    
    def _process_neural(self, messages: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Process messages using neural network."""
        if self.neural_model is None:
            return {}
        
        # Convert messages to model input
        if self.input_processor:
            model_input = self.input_processor(messages)
        else:
            model_input = self._default_input_processing(messages)
        
        # Run neural model
        with torch.no_grad():
            output = self.neural_model(model_input)
        
        # Convert output to messages
        if self.output_processor:
            return self.output_processor(output)
        else:
            return self._default_output_processing(output)
    
    def _process_hybrid(self, messages: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Process messages with dynamic mode switching."""
        # Analyze messages to determine best mode
        if self._should_use_neural(messages):
            self.set_mode(NodeMode.NEURAL)
            return self._process_neural(messages)
        elif self._should_use_implicit(messages):
            self.set_mode(NodeMode.IMPLICIT)
            return self._process_implicit(messages)
        else:
            self.set_mode(NodeMode.EXPLICIT)
            return self._process_explicit(messages)
    
    def _should_use_neural(self, messages: Dict[str, List[Any]]) -> bool:
        """Determine if neural processing is appropriate."""
        return (self.neural_model is not None and 
                any(isinstance(msg, torch.Tensor) for msg_list in messages.values() 
                    for msg in msg_list if msg_list))
    
    def _should_use_implicit(self, messages: Dict[str, List[Any]]) -> bool:
        """Determine if implicit processing is appropriate."""
        return any(isinstance(msg, (float, torch.Tensor)) for msg_list in messages.values() 
                  for msg in msg_list if msg_list)
    
    def _message_to_activation(self, message: Any) -> Union[float, torch.Tensor]:
        """Convert message to activation value."""
        if isinstance(message, (int, float)):
            return float(message)
        elif isinstance(message, torch.Tensor):
            return message
        elif isinstance(message, str):
            # Convert discrete messages to activations
            if message in ["confirm", "inhibit_confirm"]:
                return 1.0
            elif message in ["fail", "inhibit_request"]:
                return -1.0
            elif message == "wait":
                return 0.5
            else:
                return 0.0
        else:
            return 0.0
    
    def _apply_compact_rules(self, z: Dict[str, Union[float, torch.Tensor]]) -> Dict[str, Any]:
        """Apply compact arithmetic rules from paper section 3.1."""
        # Extract z values
        z_gen = z.get("gen", 0.0)
        z_por = z.get("por", 0.0)
        z_ret = z.get("ret", 0.0)  
        z_sub = z.get("sub", 0.0)
        z_sur = z.get("sur", 0.0)
        
        # Apply paper equations
        # f_node^por = 0 if z^por < 0, otherwise z^por
        f_por = 0.0 if self._is_negative(z_por) else z_por
        
        # f_node^sub = 0 if z^gen != 0 or (exists link^por and z^por <= 0), otherwise z^sub
        has_por_inhibition = self.has_por_link and self._is_non_positive(z_por)
        f_sub = 0.0 if (not self._is_zero(z_gen) or has_por_inhibition) else z_sub
        
        # f_node^sur with ret modulation
        if self._is_non_positive(z_sub) or has_por_inhibition:
            f_sur = 0.0
        elif self.has_ret_link:
            f_sur = self._multiply(self._add(z_sur, z_gen), z_ret)
        else:
            f_sur = self._add(z_sur, z_gen)
        
        # Update internal activation
        self._implicit_activation = f_sur if not self._is_zero(f_sur) else f_por
        
        # Return output messages
        output = {}
        if not self._is_zero(f_por):
            output["por"] = f_por
        if not self._is_zero(f_sub):
            output["sub"] = f_sub
        if not self._is_zero(f_sur):
            output["sur"] = f_sur
            
        return output
    
    def _is_negative(self, x: Union[float, torch.Tensor]) -> bool:
        """Check if value is negative."""
        if isinstance(x, torch.Tensor):
            return (x < 0).any().item()
        return x < 0
    
    def _is_non_positive(self, x: Union[float, torch.Tensor]) -> bool:
        """Check if value is non-positive."""
        if isinstance(x, torch.Tensor):
            return (x <= 0).any().item()
        return x <= 0
    
    def _is_zero(self, x: Union[float, torch.Tensor]) -> bool:
        """Check if value is zero."""
        if isinstance(x, torch.Tensor):
            return torch.allclose(x, torch.zeros_like(x))
        return abs(x) < 1e-8
    
    def _add(self, x: Union[float, torch.Tensor], y: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Add two values (handles mixed types)."""
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return torch.as_tensor(x) + torch.as_tensor(y)
        return x + y
    
    def _multiply(self, x: Union[float, torch.Tensor], y: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Multiply two values (handles mixed types)."""
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return torch.as_tensor(x) * torch.as_tensor(y)
        return x * y
    
    def _default_input_processing(self, messages: Dict[str, List[Any]]) -> torch.Tensor:
        """Default neural input processing."""
        # Simple concatenation of message values
        inputs = []
        for link_type in ["sub", "sur", "por", "ret", "gen"]:
            if messages.get(link_type):
                msg = messages[link_type][-1]
                if isinstance(msg, torch.Tensor):
                    inputs.append(msg.flatten())
                elif isinstance(msg, (int, float)):
                    inputs.append(torch.tensor([msg]))
                else:
                    inputs.append(torch.tensor([0.0]))
            else:
                inputs.append(torch.tensor([0.0]))
        
        return torch.cat(inputs) if inputs else torch.tensor([0.0])
    
    def _default_output_processing(self, output: torch.Tensor) -> Dict[str, Any]:
        """Default neural output processing."""
        # Simple thresholding 
        if output.numel() == 1:
            value = output.item()
            if value > self.switch_threshold:
                return {"sur": "confirm"}
            elif value < -self.switch_threshold:
                return {"sur": "fail"}
            else:
                return {"sur": "wait"}
        else:
            return {"sur": output}
    
    def _default_d2c(self, message: str) -> float:
        """Default discrete to continuous conversion."""
        return self._message_to_activation(message)
    
    def _default_c2d(self, value: Union[float, torch.Tensor]) -> str:
        """Default continuous to discrete conversion."""
        if isinstance(value, torch.Tensor):
            val = value.item() if value.numel() == 1 else value.max().item()
        else:
            val = value
            
        if val >= self.switch_threshold:
            return "confirm"
        elif val <= -self.switch_threshold:
            return "fail"
        else:
            return "wait"
    
    def set_neural_model(self, model: torch.nn.Module, 
                        input_processor: Optional[Callable] = None,
                        output_processor: Optional[Callable] = None):
        """Set neural network model and processing functions."""
        self.neural_model = model
        self.input_processor = input_processor
        self.output_processor = output_processor
    
    def measure(self, environment: Any = None) -> Union[float, torch.Tensor]:
        """Perform measurement based on current mode."""
        if self.mode == NodeMode.NEURAL and self.neural_model is not None:
            # Use neural model for measurement
            if environment is not None:
                input_tensor = self._environment_to_tensor(environment)
                with torch.no_grad():
                    return self.neural_model(input_tensor)
            else:
                return self.activation
        else:
            # Use parent measurement function
            return super().measure(environment)
    
    def _environment_to_tensor(self, environment: Any) -> torch.Tensor:
        """Convert environment to tensor for neural processing."""
        if isinstance(environment, torch.Tensor):
            return environment
        elif isinstance(environment, np.ndarray):
            return torch.from_numpy(environment).float()
        elif isinstance(environment, (list, tuple)):
            return torch.tensor(environment).float()
        else:
            return torch.tensor([0.0])
    
    def to_dict(self) -> Dict[str, Any]:
        """Export node configuration for visualization."""
        return {
            "id": self.id,
            "type": self.type,
            "mode": self.mode.value,
            "state": self.state.value,
            "activation": self.activation.tolist() if isinstance(self.activation, torch.Tensor) else self.activation,
            "has_neural_model": self.neural_model is not None,
            "activation_dim": self.activation_dim,
            "mode_history": [mode.value for mode in self.mode_history]
        }