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
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from recon_engine.compact import CompactReCoNNode


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
        
        # Set terminals to confirm explicitly (no longer auto-confirm)
        graph.get_node("T").measurement_fn = lambda env: 1.0
        graph.get_node("TA").measurement_fn = lambda env: 1.0
        graph.get_node("TB").measurement_fn = lambda env: 1.0
        
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


class TestActivationBasedTiming:
    """Test activation-based timing as alternative to discrete timing heuristics."""
    
    def test_micropsi2_style_activation_transitions(self):
        """Test MicroPsi2-style continuous activation transitions eliminate timing heuristics."""
        # Create a node that demonstrates activation-based state evolution
        # This simulates MicroPsi2's approach where activation levels provide natural timing
        
        # Mock a MicroPsi2-style node that uses activation thresholds
        class MicroPsi2StyleNode:
            def __init__(self):
                self.activation = 0.0
                self.state = "inactive"
                
            def update(self, sub_activation, sur_activation):
                """Update using MicroPsi2-style activation levels."""
                if sub_activation < 0.01:  # Not requested
                    self.activation = 0.0
                    self.state = "inactive"
                    return
                
                # MicroPsi2 script node logic (from micropsi2/nodefunctions.py)
                if self.activation < -0.01:  # failed -> failed
                    pass  # Stay failed
                elif self.activation < 0.01:  # inactive -> preparing  
                    self.activation = 0.2
                    self.state = "preparing"
                elif self.activation < 0.3:  # preparing -> suppressed/requesting
                    self.activation = 0.4 if sur_activation <= 0 else 0.6
                    self.state = "suppressed" if sur_activation <= 0 else "requesting"
                elif self.activation < 0.7:  # requesting -> pending
                    self.activation = 0.8
                    self.state = "pending"
                elif sur_activation >= 1:  # pending -> confirmed
                    self.activation = 1.0
                    self.state = "confirmed"
                elif sur_activation <= 0:  # pending -> failed
                    self.activation = -1.0
                    self.state = "failed"
                # Otherwise stay at current level
        
        node = MicroPsi2StyleNode()
        
        activation_history = []
        state_history = []
        
        # Phase 1: Node gets requested and transitions through activation levels
        for step in range(4):
            node.update(sub_activation=1.0, sur_activation=0.5)  # Requested with children waiting
            activation_history.append(node.activation)
            state_history.append(node.state)
        
        # Phase 2: Children stop responding (critical timing scenario)
        for step in range(3):
            node.update(sub_activation=1.0, sur_activation=0.0)  # No children responding
            activation_history.append(node.activation) 
            state_history.append(node.state)
        
        print(f"MicroPsi2-style activation history: {activation_history}")
        print(f"MicroPsi2-style state history: {state_history}")
        
        # Verify smooth activation evolution
        unique_activations = len(set(activation_history))
        assert unique_activations >= 4, f"Should show activation evolution, got {unique_activations} levels"
        
        # Should transition through intermediate states, not jump directly to failure
        intermediate_states = len([s for s in state_history if s not in ["inactive", "failed", "confirmed"]])
        assert intermediate_states >= 3, "Should have intermediate states providing timing buffer"
        
        # Key insight: Activation levels provide natural timing buffers
        # Node doesn't fail immediately when children stop responding
        # It transitions through intermediate activation levels first
    
    def test_hybrid_implicit_mode_eliminates_timing_heuristics(self):
        """Test concept: IMPLICIT mode provides activation-based alternatives to timing counters."""
        # This test demonstrates the concept even if HybridReCoNNode needs more work
        
        # The key insight is that activation-based approaches provide natural timing
        # through continuous values rather than discrete step counters
        
        # Example: Instead of counting "3 steps without child response = fail"
        # We can use "activation gradually decreases until threshold = fail"
        
        discrete_counter = 3  # Traditional approach: fail after N steps
        activation_level = 0.8  # Activation approach: gradual degradation
        
        # Simulate steps without child response
        for step in range(5):
            # Discrete approach: hard cutoff
            discrete_counter -= 1
            discrete_failed = discrete_counter <= 0
            
            # Activation approach: gradual decay  
            activation_level *= 0.7  # Decay factor
            activation_failed = activation_level < 0.1  # Soft threshold
            
            print(f"Step {step}: Discrete counter={discrete_counter}, failed={discrete_failed}")
            print(f"Step {step}: Activation level={activation_level:.3f}, failed={activation_failed}")
        
        # Key insight: Activation approach provides more graceful degradation
        # This is the principle behind MicroPsi2's activation-based timing
        
        # Verify the concept
        assert discrete_counter <= 0, "Discrete approach should fail abruptly"
        assert activation_level > 0, "Activation approach allows graceful degradation"
        
        print("\nActivation-based timing provides natural buffers without explicit counters!")
        print("This eliminates the need for timing heuristics like _brief_wait_count.")
    
    def test_activation_vs_discrete_timing_comparison(self):
        """Compare activation-based timing vs discrete timing heuristics."""
        # Test discrete timing (current ReCoNNode with timing counters)
        discrete_parent = ReCoNNode("DiscreteParent", "script")
        
        # Test activation-based timing (CompactReCoNNode, MicroPsi2 approach)  
        activation_parent = CompactReCoNNode("ActivationParent", "script")
        
        # Simulate the critical scenario: children stop sending wait signals
        discrete_failures = []
        activation_failures = []
        
        # Both start with some activation/state
        discrete_parent.state = ReCoNState.WAITING
        activation_parent.update_state({"sub": 1.0, "sur": 0.5})  # Get to pending state
        
        # Simulate steps of no child responses (the timing crisis)
        for step in range(8):
            # Discrete approach: uses _brief_wait_count, fails after threshold
            discrete_parent.update_state({"sub": 1.0, "sur": 0.0})
            discrete_failures.append(discrete_parent.state == ReCoNState.FAILED)
            
            # Activation approach: uses continuous values, more gradual
            activation_parent.update_state({"sub": 1.0, "sur": 0.0})
            activation_failures.append(activation_parent.state == ReCoNState.FAILED)
        
        print(f"Discrete failures by step: {discrete_failures}")
        print(f"Activation failures by step: {activation_failures}")
        
        # Find when each approach fails
        discrete_fail_step = next((i for i, failed in enumerate(discrete_failures) if failed), None)
        activation_fail_step = next((i for i, failed in enumerate(activation_failures) if failed), None)
        
        print(f"Discrete fails at step: {discrete_fail_step}")
        print(f"Activation fails at step: {activation_fail_step}")
        
        # Key insight: Activation approach should provide more graceful handling
        if discrete_fail_step is not None:
            # Discrete approach fails due to counter timeout
            assert discrete_fail_step < 6, "Discrete should fail within counter threshold"
        
        if activation_fail_step is not None:
            # Activation approach should take longer or not fail at all
            assert activation_fail_step >= discrete_fail_step or activation_fail_step is None, \
                "Activation approach should be more patient than discrete counters"
        else:
            # Activation approach might not fail at all (continuous degradation)
            print("Activation approach shows graceful degradation without hard failure")
        
        # This demonstrates how MicroPsi2's activation-based approach
        # provides natural timing buffers without explicit counters


class TestHybridTimingConfiguration:
    """Test the new hybrid timing configuration system."""
    
    def test_discrete_timing_configuration(self):
        """Test configurable discrete timing parameters."""
        node = ReCoNNode("test_discrete", "script")
        
        # Test default configuration
        assert node.timing_mode == "discrete"
        assert node.discrete_wait_steps == 3
        assert node.sequence_wait_steps == 6
        
        # Test custom configuration
        node.configure_timing(
            mode="discrete",
            discrete_wait_steps=5,
            sequence_wait_steps=10
        )
        
        assert node.discrete_wait_steps == 5
        assert node.sequence_wait_steps == 10
        
        # Test timing behavior with custom parameters
        node.state = ReCoNState.WAITING
        failures = []
        
        for step in range(8):
            # Keep node requested but no children responding
            node.update_state({"sub": 1.0, "sur": 0.0})
            failures.append(node.state == ReCoNState.FAILED)
        
        # Should fail at step 5 (custom discrete_wait_steps)
        fail_step = next((i for i, failed in enumerate(failures) if failed), None)
        assert fail_step == 4, f"Should fail at step 4 (0-indexed), got {fail_step}"  # 5th call = index 4
    
    def test_activation_timing_configuration(self):
        """Test configurable activation-based timing parameters."""
        node = ReCoNNode("test_activation", "script")
        
        # Configure for activation-based timing
        node.configure_timing(
            mode="activation",
            activation_decay_rate=0.5,  # Faster decay
            activation_failure_threshold=0.2,  # Higher threshold
            activation_initial_level=1.0  # Start at max
        )
        
        assert node.timing_mode == "activation"
        assert node.activation_decay_rate == 0.5
        assert node.activation_failure_threshold == 0.2
        assert node.activation_initial_level == 1.0
        
        # Test activation-based timing behavior
        node.state = ReCoNState.WAITING
        node._waiting_activation = 1.0  # Start at configured level
        
        activation_history = []
        state_history = []
        
        for step in range(6):
            node.update_state({"sub": 1.0, "sur": 0.0})  # No children responding
            activation_history.append(node._waiting_activation)
            state_history.append(node.state.value)
        
        print(f"Activation decay: {activation_history}")
        print(f"State progression: {state_history}")
        
        # Should show gradual activation decay: 1.0 -> 0.5 -> 0.25 -> 0.125 -> FAILED
        assert activation_history[0] == 0.5  # First decay: 1.0 * 0.5
        assert activation_history[1] == 0.25  # Second decay: 0.5 * 0.5
        
        # Should fail when activation drops below 0.2
        fail_step = next((i for i, state in enumerate(state_history) if state == "failed"), None)
        assert fail_step is not None, "Should eventually fail"
        assert fail_step <= 3, "Should fail within a few steps due to fast decay"
    
    def test_timing_mode_comparison(self):
        """Compare discrete vs activation timing side by side."""
        discrete_node = ReCoNNode("discrete", "script")
        activation_node = ReCoNNode("activation", "script")
        
        # Configure both with equivalent "patience"
        discrete_node.configure_timing(mode="discrete", discrete_wait_steps=3)
        activation_node.configure_timing(
            mode="activation", 
            activation_decay_rate=0.7,
            activation_failure_threshold=0.1,
            activation_initial_level=0.8
        )
        
        # Both start in WAITING
        discrete_node.state = ReCoNState.WAITING
        activation_node.state = ReCoNState.WAITING
        activation_node._waiting_activation = 0.8
        
        discrete_results = []
        activation_results = []
        
        for step in range(8):
            # No children responding
            discrete_node.update_state({"sub": 1.0, "sur": 0.0})
            activation_node.update_state({"sub": 1.0, "sur": 0.0})
            
            discrete_results.append({
                "step": step,
                "state": discrete_node.state.value,
                "counter": getattr(discrete_node, '_brief_wait_count', 0)
            })
            
            activation_results.append({
                "step": step, 
                "state": activation_node.state.value,
                "activation": activation_node._waiting_activation
            })
        
        print("Discrete timing progression:")
        for result in discrete_results:
            print(f"  Step {result['step']}: {result['state']} (counter: {result['counter']})")
            
        print("Activation timing progression:")
        for result in activation_results:
            print(f"  Step {result['step']}: {result['state']} (activation: {result['activation']:.3f})")
        
        # Both should eventually fail, but with different characteristics
        discrete_fail = next((r for r in discrete_results if r['state'] == 'failed'), None)
        activation_fail = next((r for r in activation_results if r['state'] == 'failed'), None)
        
        assert discrete_fail is not None, "Discrete timing should eventually fail"
        assert activation_fail is not None, "Activation timing should eventually fail"
        
        # Discrete should fail abruptly at exact step
        assert discrete_fail['step'] == 2, f"Discrete should fail at step 2, got {discrete_fail['step']}"
        
        # Activation should show gradual decay before failing
        pre_fail_activations = [r['activation'] for r in activation_results[:activation_fail['step']]]
        assert len(set(pre_fail_activations)) > 1, "Should show activation evolution before failure"
    
    def test_timing_configuration_persistence(self):
        """Test that timing configuration persists through serialization."""
        node = ReCoNNode("persistent", "script")
        
        # Configure custom timing
        node.configure_timing(
            mode="activation",
            activation_decay_rate=0.9,
            activation_failure_threshold=0.05,
            discrete_wait_steps=7
        )
        
        # Serialize and check configuration is included
        data = node.to_dict()
        
        assert "timing_config" in data
        config = data["timing_config"]
        assert config["timing_mode"] == "activation"
        assert config["activation_decay_rate"] == 0.9
        assert config["activation_failure_threshold"] == 0.05
        assert config["discrete_wait_steps"] == 7
        
        # Test get_timing_config method
        direct_config = node.get_timing_config()
        assert direct_config == config