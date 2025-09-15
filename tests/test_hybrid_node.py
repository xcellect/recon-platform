"""
Test Hybrid Node Architecture (Bonus Feature)

Tests the hybrid node system that extends the core ReCoN paper with:
- Multi-modal execution (explicit/implicit/neural)
- Seamless mode switching during execution
- Enhanced message protocol with auto-conversion
- Neural terminal integration

This is a bonus feature beyond the ReCoN paper while preserving its essence.
"""

import pytest
import torch
import torch.nn as nn
from recon_engine import ReCoNNode, ReCoNState, ReCoNGraph, MessageType


class MockNeuralModel(nn.Module):
    """Mock neural model for testing neural mode."""
    
    def __init__(self, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(1, output_dim)
        self.activation = nn.Sigmoid()
        # Initialize with positive bias to get higher outputs
        with torch.no_grad():
            self.linear.bias.fill_(2.0)  # Positive bias to push sigmoid output higher
            self.linear.weight.fill_(1.0)
    
    def forward(self, x):
        return self.activation(self.linear(x))


class TestHybridNodeArchitecture:
    """Test hybrid node capabilities beyond the core paper."""
    
    def test_hybrid_node_creation(self):
        """Hybrid nodes should support multiple execution modes."""
        node = ReCoNNode("hybrid_test", node_type="hybrid")
        
        # Should start in explicit mode (default)
        assert hasattr(node, 'execution_mode')
        assert node.execution_mode == "explicit"
        
        # Should support mode switching
        node.set_execution_mode("neural")
        assert node.execution_mode == "neural"
        
        node.set_execution_mode("implicit")
        assert node.execution_mode == "implicit"
    
    def test_explicit_mode_preserves_paper_behavior(self):
        """Explicit mode should behave exactly like original ReCoN paper."""
        hybrid_node = ReCoNNode("hybrid", node_type="hybrid")
        standard_node = ReCoNNode("standard", node_type="script")
        
        hybrid_node.set_execution_mode("explicit")
        
        # Both should follow identical state transitions
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        
        hybrid_messages = hybrid_node.update_state(inputs)
        standard_messages = standard_node.update_state(inputs)
        
        assert hybrid_node.state == standard_node.state
        assert hybrid_messages == standard_messages
    
    def test_neural_mode_tensor_processing(self):
        """Neural mode should handle tensor activations seamlessly."""
        node = ReCoNNode("neural_hybrid", node_type="hybrid")
        node.set_execution_mode("neural")
        
        # Attach mock neural model
        node.neural_model = MockNeuralModel()
        
        # Process tensor activation
        tensor_input = torch.tensor([[0.5], [0.8], [0.2]])
        node.activation = tensor_input
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": tensor_input}
        messages = node.update_state(inputs)
        
        # Should preserve tensor structure
        assert isinstance(node.activation, torch.Tensor)
        assert node.activation.shape == tensor_input.shape
    
    def test_implicit_mode_subsymbolic_integration(self):
        """Implicit mode should blend symbolic and subsymbolic processing."""
        node = ReCoNNode("implicit_hybrid", node_type="hybrid")
        node.set_execution_mode("implicit")
        
        # Implicit mode allows probabilistic state transitions
        node.transition_threshold = 0.7  # Custom threshold
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.6}  # Below threshold
        messages = node.update_state(inputs)
        
        # Should still follow core ReCoN logic but with flexibility
        assert node.state in [ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING]
    
    def test_mode_switching_preserves_state(self):
        """Mode switches should preserve node state and critical data."""
        node = ReCoNNode("mode_switcher", node_type="hybrid")
        
        # Start in explicit mode and advance to ACTIVE
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        node.update_state(inputs)
        node.update_state(inputs)  # INACTIVE -> REQUESTED -> ACTIVE
        
        original_state = node.state
        original_activation = node.activation
        
        # Switch to neural mode
        node.set_execution_mode("neural")
        
        # State and data should be preserved
        assert node.state == original_state
        assert node.activation == original_activation
        
        # Should continue functioning in new mode
        messages = node.update_state(inputs)
        assert len(messages) > 0
    
    def test_enhanced_message_protocol(self):
        """Enhanced message protocol should auto-convert between discrete/continuous."""
        node = ReCoNNode("enhanced", node_type="hybrid")
        
        # Discrete message input
        discrete_input = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        messages = node.get_outgoing_messages(discrete_input)
        
        # Should produce standard discrete messages
        assert all(isinstance(msg, str) for msg in messages.values())
        
        # Continuous tensor input
        tensor_input = {"sub": torch.tensor([0.9, 0.1]), "por": 0.0, "ret": 0.0, "sur": 0.0}
        node.set_execution_mode("neural")
        messages = node.update_state(tensor_input)
        
        # Should handle tensor inputs gracefully
        assert node.state != ReCoNState.INACTIVE  # Should activate with high tensor value
    
    def test_neural_terminal_integration(self):
        """Neural terminals should integrate with hybrid architecture."""
        terminal = ReCoNNode("neural_terminal", node_type="terminal")
        terminal.neural_model = MockNeuralModel()
        
        # Custom measurement function using neural model
        def neural_measurement(environment):
            if hasattr(terminal, 'neural_model') and terminal.neural_model:
                input_tensor = torch.tensor([[0.8]])  # High confidence input
                output = terminal.neural_model(input_tensor)
                return output.item()
            return 0.5
        
        terminal.measurement_fn = neural_measurement
        
        # Terminal should use neural measurement
        measurement = terminal.measure()
        assert 0.5 < measurement < 1.0  # Neural model with sigmoid output
        
        # Should confirm if measurement above threshold
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        
        assert terminal.state == ReCoNState.CONFIRMED
    
    def test_hybrid_graph_execution(self):
        """Hybrid nodes should work seamlessly in ReCoN graphs."""
        graph = ReCoNGraph()
        
        # Mix of standard and hybrid nodes
        graph.add_node("root", "script")
        graph.add_node("hybrid_branch", "hybrid")
        graph.add_node("neural_terminal", "terminal")
        
        graph.add_link("root", "hybrid_branch", "sub")
        graph.add_link("hybrid_branch", "neural_terminal", "sub")
        
        # Set hybrid mode
        hybrid_node = graph.get_node("hybrid_branch")
        hybrid_node.set_execution_mode("neural")
        
        # Neural terminal
        neural_terminal = graph.get_node("neural_terminal")
        neural_terminal.neural_model = MockNeuralModel()
        
        graph.request_root("root")
        
        # Should execute normally despite mixed node types
        for _ in range(10):
            graph.propagate_step()
            if graph.get_node("root").state in [ReCoNState.CONFIRMED, ReCoNState.FAILED]:
                break
        
        # Graph should reach terminal state
        assert graph.get_node("root").state in [ReCoNState.CONFIRMED, ReCoNState.FAILED, ReCoNState.TRUE]
    
    def test_backward_compatibility(self):
        """Hybrid features should not break existing ReCoN functionality."""
        graph = ReCoNGraph()
        
        # Create standard ReCoN sequence under a common parent as in paper
        for node_id in ["A", "B", "C"]:
            graph.add_node(node_id, "script")

        # Parent requests all sequence elements; por controls ordering
        graph.add_node("Parent", "script")
        graph.add_link("Parent", "A", "sub")
        graph.add_link("Parent", "B", "sub")
        graph.add_link("Parent", "C", "sub")

        # Sequence order control
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Add terminal to C
        graph.add_node("T", "terminal")
        graph.add_link("C", "T", "sub")

        # Ensure intermediate sequence nodes have sub children per paper
        graph.add_node("TA", "terminal")
        graph.add_link("A", "TA", "sub")
        graph.add_node("TB", "terminal")
        graph.add_link("B", "TB", "sub")
        
        graph.request_root("Parent")
        
        # Should execute exactly as in paper
        execution_order = []
        for step in range(20):
            graph.propagate_step()
            
            for node_id in ["A", "B", "C"]:
                if (graph.get_node(node_id).state == ReCoNState.ACTIVE and 
                    node_id not in execution_order):
                    execution_order.append(node_id)
            
            # Auto-confirm terminal when C is waiting
            if graph.get_node("C").state == ReCoNState.WAITING:
                graph.get_node("T").state = ReCoNState.CONFIRMED
        
        # Should maintain strict sequential order from paper
        assert execution_order == ["A", "B", "C"]
    
    def test_performance_mode_switching(self):
        """Mode switching should be efficient for real-time applications."""
        node = ReCoNNode("performance_test", node_type="hybrid")
        
        # Rapid mode switching
        modes = ["explicit", "neural", "implicit", "explicit"]
        
        for mode in modes:
            node.set_execution_mode(mode)
            
            # Should maintain responsiveness
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            messages = node.update_state(inputs)
            
            assert len(messages) >= 0  # Should not crash or hang
            assert node.execution_mode == mode
    
    def test_arc_agi_integration_patterns(self):
        """Test patterns useful for ARC-AGI style tasks."""
        graph = ReCoNGraph()
        
        # Pattern recognition hierarchy
        graph.add_node("pattern_detector", "hybrid")
        graph.add_node("transformation_engine", "hybrid") 
        graph.add_node("output_generator", "terminal")
        
        graph.add_link("pattern_detector", "transformation_engine", "sub")
        graph.add_link("transformation_engine", "output_generator", "sub")
        
        # Configure for ARC-AGI style processing
        pattern_node = graph.get_node("pattern_detector")
        pattern_node.set_execution_mode("neural")
        
        transform_node = graph.get_node("transformation_engine")
        transform_node.set_execution_mode("implicit")
        
        # Should support grid-like tensor processing
        grid_input = torch.zeros(8, 8)  # 8x8 ARC grid
        grid_input[2:6, 2:6] = 1.0  # Pattern in center
        
        pattern_node.activation = grid_input
        
        graph.request_root("pattern_detector")
        
        # Should process hierarchically
        for _ in range(5):
            graph.propagate_step()
        
        assert pattern_node.state != ReCoNState.INACTIVE
        assert isinstance(pattern_node.activation, torch.Tensor)


class TestHybridMessageProtocol:
    """Test enhanced message protocol for hybrid nodes."""
    
    def test_message_type_conversion(self):
        """Messages should auto-convert between discrete and continuous forms."""
        node = ReCoNNode("converter", node_type="hybrid")
        
        # Discrete to continuous
        discrete_msg = "request"
        continuous_value = node.message_to_activation(discrete_msg)
        assert continuous_value == 1.0
        
        discrete_msg = "inhibit_request"
        continuous_value = node.message_to_activation(discrete_msg)
        assert continuous_value == -1.0
        
        # Continuous to discrete
        activation_msg = node.activation_to_message(0.8, "sub")
        assert activation_msg == "request"
        
        activation_msg = node.activation_to_message(-0.5, "por")
        assert activation_msg == "inhibit_request"
    
    def test_tensor_message_aggregation(self):
        """Multiple tensor messages should aggregate properly."""
        node = ReCoNNode("aggregator", node_type="hybrid")
        
        # Add multiple tensor messages
        msg1 = torch.tensor([0.3, 0.7, 0.1])
        msg2 = torch.tensor([0.2, 0.1, 0.8])
        
        # Should aggregate maintaining tensor structure
        combined = node.aggregate_tensor_messages([msg1, msg2])
        expected = torch.tensor([0.5, 0.8, 0.9])  # Element-wise max or sum
        
        assert combined.shape == expected.shape
        assert torch.allclose(combined, expected, atol=0.1)
    
    def test_mixed_protocol_handling(self):
        """Should handle graphs with mixed discrete/continuous messages."""
        graph = ReCoNGraph()
        
        # Discrete node -> Hybrid node -> Neural terminal
        graph.add_node("discrete", "script")
        graph.add_node("hybrid", "hybrid")
        graph.add_node("neural", "terminal")
        
        graph.add_link("discrete", "hybrid", "sub")
        graph.add_link("hybrid", "neural", "sub")
        
        hybrid_node = graph.get_node("hybrid")
        hybrid_node.set_execution_mode("neural")
        
        neural_terminal = graph.get_node("neural")
        neural_terminal.neural_model = MockNeuralModel()
        
        # Ensure discrete requests hybrid, and hybrid requests neural via sub
        graph.request_root("discrete")
        
        # Should propagate through mixed protocol chain
        for _ in range(30):
            graph.propagate_step()
            # If hybrid is waiting on terminal, auto-confirm terminal to simulate measurement
            if graph.get_node("hybrid").state == ReCoNState.WAITING:
                graph.get_node("neural").state = ReCoNState.CONFIRMED
        
        # All nodes should reach appropriate terminal states
        assert graph.get_node("discrete").state in [ReCoNState.CONFIRMED, ReCoNState.FAILED, ReCoNState.TRUE]
        assert graph.get_node("hybrid").state in [ReCoNState.CONFIRMED, ReCoNState.FAILED, ReCoNState.TRUE]
        # Neural terminal should confirm on request; it may reset to inactive later when request ceases
        assert graph.get_node("neural").state in [ReCoNState.CONFIRMED, ReCoNState.FAILED, ReCoNState.INACTIVE]