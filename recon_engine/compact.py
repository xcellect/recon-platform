"""
Compact ReCoN Implementation

Implements the arithmetic rules from section 3.1 of the paper.
No explicit state machine - derives state from activation levels.
Uses gen loops for persistent states and real-valued activations.
"""

from typing import Dict, Any, Union, Optional
import torch
import numpy as np
from .node import ReCoNNode, ReCoNState
# Import ReCoNGraph when needed to avoid circular imports


class CompactReCoNNode(ReCoNNode):
    """
    Compact ReCoN node implementation using arithmetic rules.
    
    States are derived from activation levels instead of explicit state machine.
    Implements the f_node functions from section 3.1 of the paper.
    """
    
    def __init__(self, node_id: str, node_type: str = "script"):
        super().__init__(node_id, node_type)
        
        # For compact implementation, activation encodes state
        # Use gen loop for persistent states
        self.has_por_link = False
        self.has_ret_link = False
        
    def set_link_existence(self, has_por: bool = False, has_ret: bool = False):
        """Set whether this node has por/ret links (needed for compact rules)."""
        self.has_por_link = has_por
        self.has_ret_link = has_ret
    
    def update_state_compact(self, z: Dict[str, Union[float, torch.Tensor]]) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Update using compact arithmetic rules from paper section 3.1.
        
        z contains incoming activations: z^gen, z^por, z^ret, z^sub, z^sur
        Returns new gate activations.
        """
        # Extract z values
        z_gen = z.get("gen", 0.0)
        z_por = z.get("por", 0.0) 
        z_ret = z.get("ret", 0.0)
        z_sub = z.get("sub", 0.0)
        z_sur = z.get("sur", 0.0)
        
        # Convert to float for arithmetic if needed
        def to_float(x):
            if isinstance(x, torch.Tensor):
                return x.item() if x.numel() == 1 else x
            return float(x) if not isinstance(x, torch.Tensor) else x
        
        z_gen = to_float(z_gen)
        z_por = to_float(z_por) 
        z_ret = to_float(z_ret)
        z_sub = to_float(z_sub)
        z_sur = to_float(z_sur)
        
        # Implement f_node functions from paper
        
        # f_node^gen - implements gen loop for persistent states
        if (z_gen * z_sub == 0) or (self.has_por_link and z_por == 0):
            f_gen = z_sur
        else:
            f_gen = z_gen * z_sub
        
        # f_node^por - controls successor inhibition
        if (z_sub <= 0) or (self.has_por_link and z_por <= 0):
            f_por = 0
        else:
            f_por = z_sur + z_gen
        
        # f_node^ret - controls predecessor inhibition  
        if z_por < 0:
            f_ret = 1
        else:
            f_ret = 0
        
        # f_node^sub - controls child requests
        if (z_gen != 0) or (self.has_por_link and z_por <= 0):
            f_sub = 0
        else:
            f_sub = z_sub
        
        # f_node^sur - controls parent confirmation
        if (z_sub <= 0) or (self.has_por_link and z_por <= 0):
            f_sur = 0
        elif self.has_ret_link:
            f_sur = (z_sur + z_gen) * z_ret
        else:
            f_sur = z_sur + z_gen
        
        # Update node's internal activation (gen loop)
        self.gates["gen"] = f_gen
        self.activation = f_gen
        
        # Derive state from activation level (for compatibility)
        self._derive_state_from_activation()
        
        return {
            "gen": f_gen,
            "por": f_por, 
            "ret": f_ret,
            "sub": f_sub,
            "sur": f_sur
        }
    
    def _derive_state_from_activation(self):
        """Derive discrete state from continuous activation for compatibility."""
        activation = float(self.activation) if not isinstance(self.activation, torch.Tensor) else self.activation.item()
        
        if activation < 0:
            self.state = ReCoNState.FAILED
        elif activation < 0.01:
            self.state = ReCoNState.INACTIVE
        elif activation < 0.3:
            self.state = ReCoNState.REQUESTED  # preparing
        elif activation < 0.5:
            self.state = ReCoNState.SUPPRESSED
        elif activation < 0.7:
            self.state = ReCoNState.ACTIVE  # requesting
        elif activation < 1.0:
            self.state = ReCoNState.WAITING  # pending
        else:
            self.state = ReCoNState.CONFIRMED
    
    def update_state(self, inputs: Optional[Dict[str, Union[float, torch.Tensor]]] = None) -> Dict[str, str]:
        """Override to use compact implementation."""
        if inputs is None:
            inputs = {}
        
        # Get z values from inputs and gen loop
        z = {
            "gen": self.gates.get("gen", 0.0),
            "por": inputs.get("por", self.get_link_activation("por")),
            "ret": inputs.get("ret", self.get_link_activation("ret")),
            "sub": inputs.get("sub", self.get_link_activation("sub")),
            "sur": inputs.get("sur", self.get_link_activation("sur"))
        }
        
        # Update using compact rules
        new_gates = self.update_state_compact(z)
        
        # Update gates
        for gate_type, activation in new_gates.items():
            self.gates[gate_type] = activation
        
        # Convert gate activations to messages
        return self._gates_to_messages(new_gates)
    
    def _gates_to_messages(self, gates: Dict[str, Union[float, torch.Tensor]]) -> Dict[str, str]:
        """Convert gate activations to discrete messages."""
        messages = {}
        
        # por gate -> inhibit_request if active
        if gates["por"] > 0:
            messages["por"] = "request"  # Positive por
        elif gates["por"] < 0:
            messages["por"] = "inhibit_request"
        
        # ret gate -> inhibit_confirm if active  
        if gates["ret"] > 0:
            messages["ret"] = "inhibit_confirm"
        
        # sub gate -> request if active
        if gates["sub"] > 0:
            messages["sub"] = "request"
        
        # sur gate -> confirm/wait based on level
        sur_level = float(gates["sur"]) if not isinstance(gates["sur"], torch.Tensor) else gates["sur"].item()
        if sur_level >= 1.0:
            messages["sur"] = "confirm"
        elif sur_level > 0:
            messages["sur"] = "wait"
        
        return messages


class CompactReCoNGraph(ReCoNGraph):
    """
    ReCoN Graph using compact node implementation.
    
    Addresses the "undesirable property" mentioned in the paper by
    making link existence explicit.
    """
    
    def add_node(self, node_id: str, node_type: str = "script") -> CompactReCoNNode:
        """Add a compact ReCoN node."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
            
        node = CompactReCoNNode(node_id, node_type)
        self.nodes[node_id] = node
        self.graph.add_node(node_id, node_obj=node)
        
        return node
    
    def add_link(self, source: str, target: str, link_type: str, weight: Union[float, torch.Tensor] = 1.0):
        """Add link and update node link existence flags."""
        link = super().add_link(source, target, link_type, weight)
        
        # Update compact nodes' link existence flags
        if isinstance(self.nodes[source], CompactReCoNNode):
            if link_type == "por":
                self.nodes[source].has_por_link = True
            elif link_type == "ret":
                self.nodes[target].has_ret_link = True
                
        if isinstance(self.nodes[target], CompactReCoNNode):
            if link_type == "por":
                self.nodes[target].has_ret_link = True  # Auto-created ret link
            elif link_type == "ret":
                self.nodes[source].has_por_link = True  # Auto-created por link
        
        return link
    
    def propagate_step(self):
        """
        Propagate using compact two-phase approach: propagation then calculation.
        
        Phase 1: z = W · a (propagation)
        Phase 2: f_gate(f_node(z)) (calculation)
        """
        self.step_count += 1
        
        # Phase 1: Propagation - collect weighted activations
        z_values = {}  # node_id -> {link_type -> activation}
        
        for node_id in self.nodes:
            z_values[node_id] = {"gen": 0, "por": 0, "ret": 0, "sub": 0, "sur": 0}
        
        # Add root requests
        for root_id in self.requested_roots:
            z_values[root_id]["sub"] = 1.0
        
        # Propagate along links: z = W · a
        for link in self.links:
            source_node = self.nodes[link.source]
            target_node = self.nodes[link.target]
            
            # Get source activation
            source_activation = source_node.activation
            if isinstance(source_activation, torch.Tensor):
                source_activation = source_activation.item() if source_activation.numel() == 1 else source_activation
            
            # Weight and propagate
            weighted_activation = source_activation * link.weight
            if isinstance(weighted_activation, torch.Tensor):
                weighted_activation = weighted_activation.item() if weighted_activation.numel() == 1 else weighted_activation
            
            # Add to target's z value for this link type
            current = z_values[link.target][link.type]
            if isinstance(current, torch.Tensor) or isinstance(weighted_activation, torch.Tensor):
                if not isinstance(current, torch.Tensor):
                    current = torch.tensor(current)
                if not isinstance(weighted_activation, torch.Tensor):
                    weighted_activation = torch.tensor(weighted_activation)
                z_values[link.target][link.type] = current + weighted_activation
            else:
                z_values[link.target][link.type] = current + weighted_activation
        
        # Phase 2: Calculation - f_gate(f_node(z))
        for node_id, node in self.nodes.items():
            if isinstance(node, CompactReCoNNode):
                # Use compact update
                z = z_values[node_id]
                node.update_state_compact(z)
            else:
                # Use regular update
                inputs = {k: v for k, v in z_values[node_id].items() if v != 0}
                node.update_state(inputs)
        
        # Handle terminal measurements
        for node in self.nodes.values():
            if node.type == "terminal" and node.state == ReCoNState.ACTIVE:
                measurement = node.measure()
                if isinstance(node, CompactReCoNNode):
                    # Set activation directly
                    node.activation = 1.0 if measurement > 0.8 else -1.0
                    node._derive_state_from_activation()
                else:
                    node.state = ReCoNState.CONFIRMED if measurement > 0.8 else ReCoNState.FAILED


def create_compact_example():
    """Create an example compact ReCoN for testing."""
    graph = CompactReCoNGraph()
    
    # Create simple sequence: A -> B -> T
    node_a = graph.add_node("A", "script")
    node_b = graph.add_node("B", "script") 
    node_t = graph.add_node("T", "terminal")
    
    graph.add_link("A", "B", "por")
    graph.add_link("B", "T", "sub")
    
    return graph


def test_compact_vs_regular():
    """Test that compact and regular implementations give same results."""
    
    # Create same network with both implementations
    regular_graph = ReCoNGraph()
    compact_graph = CompactReCoNGraph()
    
    for graph in [regular_graph, compact_graph]:
        graph.add_node("Root", "script")
        graph.add_node("A", "script") 
        graph.add_node("B", "script")
        graph.add_node("T", "terminal")
        
        graph.add_link("Root", "A", "sub")
        graph.add_link("A", "B", "por")
        graph.add_link("B", "T", "sub")
    
    # Execute both
    regular_result = regular_graph.execute_script("Root", max_steps=20)
    compact_result = compact_graph.execute_script("Root", max_steps=20)
    
    return regular_result, compact_result


if __name__ == "__main__":
    # Test compact implementation
    graph = create_compact_example()
    
    print("Compact ReCoN Example:")
    print(f"Initial: {graph}")
    
    result = graph.execute_script("A", max_steps=10)
    print(f"Result: {result}")
    print(f"Final states: {[(nid, node.state.value) for nid, node in graph.nodes.items()]}")
    
    # Compare implementations
    regular_result, compact_result = test_compact_vs_regular()
    print(f"\nComparison - Regular: {regular_result}, Compact: {compact_result}")
    print(f"Match: {regular_result == compact_result}")