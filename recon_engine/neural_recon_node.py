"""
Section 2.2 Neural Definition Implementation

Implements the neural ReCoN node with 10 threshold elements as specified
in Section 2.2 of "Request Confirmation Networks for Neuro-Symbolic Script Execution".

This provides a neural implementation that should be equivalent to the 
discrete state machine (Table 1) and compact arithmetic rules (Section 3.1).
"""

from typing import Dict, Any, Union, List, Optional
import torch
import torch.nn as nn
from .node import ReCoNNode, ReCoNState
from .messages import MessageType, ReCoNMessage


class ThresholdElement(nn.Module):
    """
    Neural threshold element implementing Section 2.2's activation function.
    
    α_j = Σ(w_ij·α_i) if all w_ij·α_i ≥ 0, else 0
    """
    
    def __init__(self, element_id: str, input_size: int = 5):
        super().__init__()
        self.element_id = element_id
        self.weights = nn.Parameter(torch.randn(input_size))
        self.register_buffer('activations', torch.zeros(1))
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute activation using Section 2.2's threshold function.
        
        Args:
            inputs: Input activations [sub, sur, por, ret, gen]
            
        Returns:
            Output activation (0 if any w_ij·α_i < 0, else sum)
        """
        # Compute weighted inputs: w_ij · α_i
        weighted_inputs = self.weights * inputs
        
        # Check if any weighted input is negative (complete inhibition)
        if torch.any(weighted_inputs < 0):
            return torch.tensor(0.0)
        else:
            return torch.sum(weighted_inputs)


class NeuralReCoNNode(ReCoNNode):
    """
    Neural ReCoN node implementing Section 2.2's 10-element ensemble.
    
    Provides neural implementation equivalent to discrete state machine.
    """
    
    def __init__(self, node_id: str, node_type: str = "script"):
        super().__init__(node_id, node_type)
        
        # Create 10 threshold elements as per Figure 3
        self.neural_elements = nn.ModuleDict({
            "IC": ThresholdElement("IC"),  # Inhibit Confirm
            "IR": ThresholdElement("IR"),  # Inhibit Request
            "W": ThresholdElement("W"),    # Wait signal
            "C": ThresholdElement("C"),    # Confirm signal
            "R": ThresholdElement("R"),    # Request signal
            "F": ThresholdElement("F"),    # Fail signal
            "S1": ThresholdElement("S1"),  # State element 1
            "S2": ThresholdElement("S2"),  # State element 2
            "S3": ThresholdElement("S3"),  # State element 3
            "S4": ThresholdElement("S4")   # State element 4
        })
        
        # Initialize connectivity based on Figure 3 interpretation
        self._initialize_neural_connectivity()
        
        # Neural state tracking
        self.neural_state_vector = torch.zeros(10)
        
    def _initialize_neural_connectivity(self):
        """Initialize neural connectivity based on Section 2.2 description."""
        # Based on paper: "request activation is directly sent to neurons IC, IR, and W"
        
        # IC (Inhibit Confirm) - receives request, inhibits confirm signals
        self.neural_elements["IC"].weights.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # [sub, sur, por, ret, gen]
        
        # IR (Inhibit Request) - receives request, inhibits child requests  
        self.neural_elements["IR"].weights.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        
        # W (Wait) - receives request, sends wait to parent
        self.neural_elements["W"].weights.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        
        # C (Confirm) - receives child confirmations, inhibited by IC
        self.neural_elements["C"].weights.data = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])  # sur input
        
        # R (Request) - sends requests to children, inhibited by IR
        self.neural_elements["R"].weights.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # sub input
        
        # F (Fail) - detects failure conditions
        self.neural_elements["F"].weights.data = torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0])  # negative sur
        
        # State elements for internal processing
        for i, element_id in enumerate(["S1", "S2", "S3", "S4"]):
            # Initialize with small random weights
            self.neural_elements[element_id].weights.data = torch.randn(5) * 0.1
    
    def update_state_neural(self, inputs: Dict[str, Union[float, torch.Tensor]]) -> Dict[str, str]:
        """
        Update state using neural ensemble instead of discrete state machine.
        
        This implements Section 2.2's neural definition as an alternative to
        the discrete state machine from Table 1.
        """
        # Convert inputs to tensor
        input_vector = torch.tensor([
            float(inputs.get("sub", 0.0)),
            float(inputs.get("sur", 0.0)), 
            float(inputs.get("por", 0.0)),
            float(inputs.get("ret", 0.0)),
            float(inputs.get("gen", 0.0))
        ])
        
        # Compute all element activations
        element_outputs = {}
        for element_id, element in self.neural_elements.items():
            with torch.no_grad():
                element_outputs[element_id] = element(input_vector)
        
        # Extract message signals from neural ensemble
        messages = {}
        
        # Map neural outputs to ReCoN messages
        ic_activation = element_outputs["IC"].item()
        ir_activation = element_outputs["IR"].item()
        w_activation = element_outputs["W"].item()
        c_activation = element_outputs["C"].item()
        r_activation = element_outputs["R"].item()
        f_activation = element_outputs["F"].item()
        
        # Generate messages based on neural activations
        if ir_activation > 0.5:
            messages["por"] = "inhibit_request"
        if ic_activation > 0.5:
            messages["ret"] = "inhibit_confirm"
        if r_activation > 0.5:
            messages["sub"] = "request"
        if w_activation > 0.5:
            messages["sur"] = "wait"
        elif c_activation > 0.5:
            messages["sur"] = "confirm"
        elif f_activation > 0.5:
            messages["sur"] = "fail"
        
        # Update node's discrete state for compatibility
        self._derive_state_from_neural_outputs(element_outputs)
        
        return messages
    
    def _derive_state_from_neural_outputs(self, element_outputs: Dict[str, torch.Tensor]):
        """Derive discrete state from neural element activations for compatibility."""
        # This maps neural ensemble state back to discrete ReCoN states
        
        ic = element_outputs["IC"].item()
        ir = element_outputs["IR"].item() 
        w = element_outputs["W"].item()
        c = element_outputs["C"].item()
        r = element_outputs["R"].item()
        f = element_outputs["F"].item()
        
        # Derive state based on neural activation pattern
        if c > 0.5:
            self.state = ReCoNState.CONFIRMED
        elif f > 0.5:
            self.state = ReCoNState.FAILED
        elif r > 0.5 and w > 0.5:
            self.state = ReCoNState.WAITING  # Requesting children and waiting
        elif r > 0.5:
            self.state = ReCoNState.ACTIVE   # Requesting children
        elif ir > 0.5 and not r > 0.5:
            self.state = ReCoNState.SUPPRESSED  # Inhibited from requesting
        elif w > 0.5:
            self.state = ReCoNState.REQUESTED  # Sending wait but not requesting
        else:
            self.state = ReCoNState.INACTIVE
        
        # Update activation for visualization
        self.activation = max(ic, ir, w, c, r, f)
    
    def update_state(self, inputs: Optional[Dict[str, Union[float, torch.Tensor]]] = None) -> Dict[str, str]:
        """Override to use neural processing when enabled."""
        if inputs is None:
            inputs = {}
        
        # Use neural processing instead of discrete state machine
        return self.update_state_neural(inputs)
    
    def get_neural_state(self) -> Dict[str, float]:
        """Get internal neural element activations for debugging/visualization."""
        if not hasattr(self, 'neural_elements'):
            return {}
        
        # Get current activations from all elements
        state = {}
        for element_id, element in self.neural_elements.items():
            if hasattr(element, 'activations'):
                state[element_id] = element.activations.item()
            else:
                state[element_id] = 0.0
        
        return state
    
    def to_dict(self) -> Dict[str, Any]:
        """Include neural state in serialization."""
        data = super().to_dict()
        data["neural_state"] = self.get_neural_state()
        data["implementation"] = "neural_section_2_2"
        return data


def create_neural_recon_node(node_id: str, node_type: str = "script") -> NeuralReCoNNode:
    """
    Factory function to create a neural ReCoN node implementing Section 2.2.
    
    This provides a drop-in replacement for standard ReCoNNode that uses
    the neural definition instead of discrete state machine.
    """
    return NeuralReCoNNode(node_id, node_type)


# Integration with existing hybrid system
def add_neural_mode_to_hybrid_node():
    """
    Proposal: Extend HybridReCoNNode to support Section 2.2 neural mode.
    
    This adds NodeMode.NEURAL_THRESHOLD that uses the threshold
    element ensemble instead of generic neural models.
    """
    pass  # Implementation proposal documented in tests
