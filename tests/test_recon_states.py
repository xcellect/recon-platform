"""
Test ReCoN State Machine Behavior

Tests the 8 states and transitions as specified in the ReCoN paper:
- inactive, requested, active, suppressed, waiting, true, confirmed, failed
"""

import pytest
import torch
from recon_engine import ReCoNNode, ReCoNState


class TestReCoNStates:
    """Test individual node state behavior and transitions."""
    
    def test_initial_state(self):
        """New nodes should start in inactive state."""
        node = ReCoNNode("test_node")
        assert node.state == ReCoNState.INACTIVE
        assert node.activation == 0.0
    
    def test_inactive_to_requested_transition(self):
        """When sub > 0, inactive nodes should transition to requested."""
        node = ReCoNNode("test_node")
        
        # Simulate receiving request via sub link
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        node.update_state(inputs)
        
        assert node.state == ReCoNState.REQUESTED
    
    def test_requested_state_behavior(self):
        """Requested nodes should send inhibit signals and transition to active."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.REQUESTED
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        signals = node.update_state(inputs)
        
        # Should send inhibit_request via por and inhibit_confirm via ret
        assert signals["por"] == "inhibit_request"
        assert signals["ret"] == "inhibit_confirm" 
        assert signals["sur"] == "wait"
        assert node.state == ReCoNState.ACTIVE
    
    def test_active_state_behavior(self):
        """Active nodes should request children and send wait to parent."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.ACTIVE
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        signals = node.update_state(inputs)
        
        # Should request children and continue inhibiting siblings
        assert signals["sub"] == "request"
        assert signals["por"] == "inhibit_request"
        assert signals["ret"] == "inhibit_confirm"
        assert signals["sur"] == "wait"
        assert node.state == ReCoNState.WAITING
    
    def test_suppressed_state_behavior(self):
        """Suppressed nodes should wait for por inhibition to cease."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.SUPPRESSED
        
        # While inhibited by predecessor
        inputs = {"sub": 1.0, "por": -1.0, "ret": 0.0, "sur": 0.0}
        signals = node.update_state(inputs)
        
        assert node.state == ReCoNState.SUPPRESSED
        assert signals["por"] == "inhibit_request"
        assert signals["ret"] == "inhibit_confirm"
        
        # When inhibition stops
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        signals = node.update_state(inputs)
        
        assert node.state == ReCoNState.ACTIVE
    
    def test_waiting_to_true_transition(self):
        """Waiting nodes should transition to true when a child confirms."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.WAITING
        
        # Child confirms via sur
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 1.0}
        signals = node.update_state(inputs)
        
        assert node.state == ReCoNState.TRUE
    
    def test_waiting_to_failed_transition(self):
        """Waiting nodes should fail when no children are waiting."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.WAITING
        
        # No more children waiting (sur = 0)
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        signals = node.update_state(inputs)
        
        assert node.state == ReCoNState.FAILED
    
    def test_true_state_behavior(self):
        """True nodes should stop inhibiting but not yet confirm."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.TRUE
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 1.0}
        signals = node.update_state(inputs)
        
        # Should not send por inhibition anymore
        assert signals.get("por") != "inhibit_request"
        assert signals["ret"] == "inhibit_confirm"
        assert node.state == ReCoNState.TRUE
    
    def test_true_to_confirmed_transition(self):
        """True nodes should confirm when no ret inhibition."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.TRUE
        
        # No ret inhibition (last in sequence)
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 1.0}
        signals = node.update_state(inputs)
        
        assert node.state == ReCoNState.CONFIRMED
        assert signals["sur"] == "confirm"
    
    def test_confirmed_state_persistence(self):
        """Confirmed nodes should remain confirmed until request ends."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.CONFIRMED
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 1.0}
        signals = node.update_state(inputs)
        
        assert node.state == ReCoNState.CONFIRMED
        assert signals["sur"] == "confirm"
    
    def test_failed_state_persistence(self):
        """Failed nodes should remain failed until request ends."""
        node = ReCoNNode("test_node")
        node.state = ReCoNState.FAILED
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        signals = node.update_state(inputs)
        
        assert node.state == ReCoNState.FAILED
    
    def test_request_termination_reset(self):
        """All nodes should reset to inactive when sub <= 0."""
        for state in [ReCoNState.CONFIRMED, ReCoNState.FAILED, ReCoNState.WAITING]:
            node = ReCoNNode("test_node")
            node.state = state
            
            # Request terminated
            inputs = {"sub": 0.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            signals = node.update_state(inputs)
            
            assert node.state == ReCoNState.INACTIVE
            assert node.activation == 0.0
    
    def test_terminal_node_behavior(self):
        """Terminal nodes should have different behavior than script nodes."""
        terminal = ReCoNNode("terminal", node_type="terminal")
        
        # Terminal nodes should only have states: inactive, active, confirmed
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        signals = terminal.update_state(inputs)
        
        assert terminal.state in [ReCoNState.INACTIVE, ReCoNState.ACTIVE, ReCoNState.CONFIRMED]
        
        # Terminal nodes can only be targeted by sub, source of sur
        assert "sub" not in signals  # Cannot request children
        assert "por" not in signals  # Not part of sequences
        assert "ret" not in signals  # Not part of sequences


class TestReCoNStateIntegration:
    """Test state behavior in integrated scenarios."""
    
    def test_activation_tensor_support(self):
        """Nodes should support tensor activations for subsymbolic processing."""
        node = ReCoNNode("test_node")
        
        # Set tensor activation
        tensor_activation = torch.tensor([0.8, 0.2, 0.9])
        node.activation = tensor_activation
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        node.update_state(inputs)
        
        # Activation should be preserved during state transitions
        assert torch.allclose(node.activation, tensor_activation)
    
    def test_concurrent_state_updates(self):
        """Multiple nodes should be able to update simultaneously."""
        nodes = [ReCoNNode(f"node_{i}") for i in range(3)]
        
        # All receive request simultaneously
        for node in nodes:
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            node.update_state(inputs)
        
        # All should transition to requested
        assert all(node.state == ReCoNState.REQUESTED for node in nodes)