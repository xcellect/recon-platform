"""
ReCoN Node Implementation

Core ReCoN node with state machine and message passing.
Implements the 8-state machine from the ReCoN paper.
"""

from enum import Enum
from typing import Dict, Any, Union, List, Optional
import torch
from .messages import MessageType, ReCoNMessage


class ReCoNState(Enum):
    """ReCoN node states as defined in the paper."""
    INACTIVE = "inactive"
    REQUESTED = "requested" 
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    WAITING = "waiting"
    TRUE = "true"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class ReCoNNode:
    """
    A Request Confirmation Network node.
    
    Implements the state machine and message passing semantics
    from "Request Confirmation Networks for Neuro-Symbolic Script Execution".
    """
    
    def __init__(self, node_id: str, node_type: str = "script"):
        self.id = node_id
        self.type = node_type  # "script" or "terminal"
        self.state = ReCoNState.INACTIVE
        self.activation = 0.0  # Can be float or torch.Tensor
        
        # Gate activations for link types  
        self.gates = {
            "sub": 0.0,
            "sur": 0.0, 
            "por": 0.0,
            "ret": 0.0,
            "gen": 0.0  # For self-loops in compact implementation
        }
        
        # Track incoming messages for this step
        self.incoming_messages: Dict[str, List[ReCoNMessage]] = {
            "sub": [], "sur": [], "por": [], "ret": [], "gen": []
        }
        
        # For terminal nodes - measurement function
        self.measurement_fn = None
        
    def reset(self):
        """Reset node to inactive state."""
        self.state = ReCoNState.INACTIVE
        self.activation = 0.0
        for gate in self.gates:
            self.gates[gate] = 0.0
        for link_type in self.incoming_messages:
            self.incoming_messages[link_type].clear()
    
    def add_incoming_message(self, message: ReCoNMessage):
        """Add incoming message for processing."""
        if message.link_type in self.incoming_messages:
            self.incoming_messages[message.link_type].append(message)
    
    def get_link_activation(self, link_type: str) -> Union[float, torch.Tensor]:
        """Get combined activation from all messages on a link type."""
        if not self.incoming_messages[link_type]:
            return 0.0
            
        # Sum all activations for this link type
        total = 0.0
        for msg in self.incoming_messages[link_type]:
            if msg.type == MessageType.INHIBIT_REQUEST and link_type == "por":
                total = -1.0  # Inhibition signal
            elif msg.type == MessageType.INHIBIT_CONFIRM and link_type == "ret":
                total = -1.0  # Inhibition signal
            elif msg.type == MessageType.REQUEST and link_type == "sub":
                total = 1.0  # Request signal
            elif msg.type == MessageType.CONFIRM and link_type == "sur":
                total = 1.0  # Confirm signal
            elif msg.type == MessageType.WAIT and link_type == "sur":
                total = 0.01  # Wait signal (small positive)
            else:
                # Regular activation
                if isinstance(msg.activation, torch.Tensor):
                    if isinstance(total, torch.Tensor):
                        total = total + msg.activation
                    else:
                        total = msg.activation.clone()
                else:
                    total += msg.activation
                    
        return total
    
    def update_state(self, inputs: Optional[Dict[str, Union[float, torch.Tensor]]] = None) -> Dict[str, str]:
        """
        Update node state based on incoming messages.
        Returns outgoing messages to send.
        """
        if inputs is None:
            inputs = {}
        
        # Get activation from each link type
        sub_activation = inputs.get("sub", self.get_link_activation("sub"))
        por_activation = inputs.get("por", self.get_link_activation("por")) 
        ret_activation = inputs.get("ret", self.get_link_activation("ret"))
        sur_activation = inputs.get("sur", self.get_link_activation("sur"))
        
        # Check if requested (sub > 0)
        is_requested = (isinstance(sub_activation, torch.Tensor) and sub_activation.sum() > 0) or \
                      (isinstance(sub_activation, (int, float)) and sub_activation > 0)
        
        # Check if inhibited by predecessor (por < 0)
        is_por_inhibited = (isinstance(por_activation, torch.Tensor) and por_activation.sum() < 0) or \
                          (isinstance(por_activation, (int, float)) and por_activation < 0)
        
        # Check if ret inhibited (ret < 0)
        is_ret_inhibited = (isinstance(ret_activation, torch.Tensor) and ret_activation.sum() < 0) or \
                          (isinstance(ret_activation, (int, float)) and ret_activation < 0)
        
        # Check if child confirmed (sur >= 1) 
        child_confirmed = (isinstance(sur_activation, torch.Tensor) and sur_activation.sum() >= 1) or \
                         (isinstance(sur_activation, (int, float)) and sur_activation >= 1)
        
        # Check if children still waiting (sur > 0 but < 1)
        children_waiting = (isinstance(sur_activation, torch.Tensor) and 0 < sur_activation.sum() < 1) or \
                          (isinstance(sur_activation, (int, float)) and 0 < sur_activation < 1)
        
        # No children waiting (sur <= 0)
        no_children_waiting = (isinstance(sur_activation, torch.Tensor) and sur_activation.sum() <= 0) or \
                             (isinstance(sur_activation, (int, float)) and sur_activation <= 0)
        
        # State transitions based on paper specification
        old_state = self.state
        
        # Terminal nodes have simplified state machine
        if self.type == "terminal":
            if not is_requested:
                self.state = ReCoNState.INACTIVE
                self.activation = 0.0
            elif old_state == ReCoNState.INACTIVE and is_requested:
                # Terminal goes directly to measurement when requested
                measurement = self.measure()
                if measurement > 0.8:  # Threshold from paper
                    self.state = ReCoNState.CONFIRMED
                else:
                    self.state = ReCoNState.FAILED
            # Terminal states persist until request ends
            elif old_state in [ReCoNState.CONFIRMED, ReCoNState.FAILED]:
                if not is_requested:
                    self.state = ReCoNState.INACTIVE
                    self.activation = 0.0
        
        # Script nodes follow full state machine
        else:
            if not is_requested:
                # Request terminated - reset to inactive
                self.state = ReCoNState.INACTIVE
                self.activation = 0.0
                
            elif old_state == ReCoNState.INACTIVE and is_requested:
                self.state = ReCoNState.REQUESTED
                
            elif old_state == ReCoNState.REQUESTED:
                if is_por_inhibited:
                    self.state = ReCoNState.SUPPRESSED
                else:
                    self.state = ReCoNState.ACTIVE
                    
            elif old_state == ReCoNState.SUPPRESSED:
                if not is_por_inhibited:
                    self.state = ReCoNState.ACTIVE
                    
            elif old_state == ReCoNState.ACTIVE:
                self.state = ReCoNState.WAITING
                
            elif old_state == ReCoNState.WAITING:
                if child_confirmed:
                    self.state = ReCoNState.TRUE
                elif no_children_waiting:
                    self.state = ReCoNState.FAILED
                    
            elif old_state == ReCoNState.TRUE:
                if not is_ret_inhibited:
                    self.state = ReCoNState.CONFIRMED
                # Stay in TRUE state if still ret inhibited
                    
            # Terminal states persist until request ends
            elif old_state in [ReCoNState.CONFIRMED, ReCoNState.FAILED]:
                if not is_requested:
                    self.state = ReCoNState.INACTIVE
                    self.activation = 0.0
        
        # Update activation for subsymbolic processing
        if self.state not in [ReCoNState.INACTIVE, ReCoNState.FAILED]:
            if isinstance(sur_activation, torch.Tensor):
                self.activation = sur_activation
            elif isinstance(self.activation, torch.Tensor):
                # Keep existing tensor activation
                pass
            else:
                self.activation = float(sur_activation) if sur_activation != 0 else self.activation
        
        # Generate outgoing messages
        return self.get_outgoing_messages(inputs)
    
    def get_outgoing_messages(self, inputs: Dict[str, Union[float, torch.Tensor]]) -> Dict[str, str]:
        """Get messages to send based on current state (Table 1 from paper)."""
        messages = {}
        
        if self.state == ReCoNState.INACTIVE:
            # Send nothing
            pass
            
        elif self.state == ReCoNState.REQUESTED:
            messages["por"] = "inhibit_request"
            messages["ret"] = "inhibit_confirm"
            messages["sur"] = "wait"
            
        elif self.state == ReCoNState.ACTIVE:
            messages["por"] = "inhibit_request" 
            messages["ret"] = "inhibit_confirm"
            messages["sub"] = "request"
            messages["sur"] = "wait"
            
        elif self.state == ReCoNState.SUPPRESSED:
            messages["por"] = "inhibit_request"
            messages["ret"] = "inhibit_confirm"
            
        elif self.state == ReCoNState.WAITING:
            messages["por"] = "inhibit_request"
            messages["ret"] = "inhibit_confirm" 
            messages["sub"] = "request"
            messages["sur"] = "wait"
            
        elif self.state == ReCoNState.TRUE:
            # Stop inhibiting por (successors can now activate)
            messages["ret"] = "inhibit_confirm"
            # Send wait to parent to prevent it from failing (micropsi2 logic)
            messages["sur"] = "wait"
            
        elif self.state == ReCoNState.CONFIRMED:
            messages["ret"] = "inhibit_confirm"  # Still inhibit predecessors per Table 1
            # Only send confirm if not ret inhibited (i.e., last in sequence)
            ret_activation = inputs.get("ret", self.get_link_activation("ret"))
            is_ret_inhibited = (isinstance(ret_activation, torch.Tensor) and ret_activation.sum() < 0) or \
                              (isinstance(ret_activation, (int, float)) and ret_activation < 0)
            
            if is_ret_inhibited:
                messages["sur"] = "wait"  # Send wait if inhibited (not last in sequence)
            else:
                messages["sur"] = "confirm"  # Send confirm only if last in sequence
            
        elif self.state == ReCoNState.FAILED:
            messages["por"] = "inhibit_request"
            messages["ret"] = "inhibit_confirm"
        
        # Terminal nodes have restricted message types and behavior
        if self.type == "terminal":
            # Terminals can only send sur messages, and have simpler state logic
            if self.state == ReCoNState.CONFIRMED:
                messages = {"sur": "confirm"}
            elif self.state == ReCoNState.FAILED:
                messages = {}  # Failed terminals send nothing
            else:
                messages = {}  # Inactive/other states send nothing
            
        return messages
    
    def can_confirm(self, inputs: Dict[str, Union[float, torch.Tensor]]) -> bool:
        """Check if node can confirm (not ret inhibited)."""
        ret_activation = inputs.get("ret", self.get_link_activation("ret"))
        is_ret_inhibited = (isinstance(ret_activation, torch.Tensor) and ret_activation.sum() < 0) or \
                          (isinstance(ret_activation, (int, float)) and ret_activation < 0)
        return not is_ret_inhibited and self.state == ReCoNState.TRUE
    
    def measure(self, environment: Any = None) -> Union[float, torch.Tensor]:
        """
        For terminal nodes - perform measurement.
        Returns activation value based on environment.
        """
        if self.type != "terminal":
            return 0.0
            
        if self.measurement_fn is not None:
            return self.measurement_fn(environment)
        else:
            # Default terminal behavior - confirm immediately
            return 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "state": self.state.value,
            "activation": self.activation.tolist() if isinstance(self.activation, torch.Tensor) else self.activation,
            "gates": {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in self.gates.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReCoNNode':
        """Deserialize from dictionary."""
        node = cls(data["id"], data["type"])
        node.state = ReCoNState(data["state"])
        
        activation = data["activation"]
        if isinstance(activation, list):
            node.activation = torch.tensor(activation)
        else:
            node.activation = activation
            
        for gate, value in data["gates"].items():
            if isinstance(value, list):
                node.gates[gate] = torch.tensor(value)
            else:
                node.gates[gate] = value
                
        return node
    
    def __repr__(self):
        return f"ReCoNNode(id={self.id}, type={self.type}, state={self.state.value})"