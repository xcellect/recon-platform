"""
Terminal Threshold Behavior Tests

Tests terminal node measurement and threshold behavior to ensure compliance
with paper specifications and neutral default behavior.

Verifies that terminals:
1. Use neutral default measurement (0.5, below threshold)
2. Require explicit measurement_fn for confirmation
3. Follow proper threshold logic from paper
4. Don't imply semantics not stated in paper
"""

import pytest
from hypothesis import given, strategies as st, settings
import torch
from recon_engine import ReCoNNode, ReCoNState, ReCoNGraph


class TestTerminalThresholdBehavior:
    """Test terminal node threshold and measurement behavior."""
    
    def test_neutral_default_measurement(self):
        """Test that terminals have neutral default behavior (don't auto-confirm)."""
        terminal = ReCoNNode("neutral_terminal", "terminal")
        
        # Default measurement should be neutral (below threshold)
        default_measurement = terminal.measure()
        assert default_measurement == 0.5, f"Default measurement should be 0.5, got {default_measurement}"
        
        # Default threshold should be 0.8 (from paper)
        assert terminal.transition_threshold == 0.8, f"Default threshold should be 0.8, got {terminal.transition_threshold}"
        
        # With default measurement (0.5 < 0.8), terminal should fail
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        
        assert terminal.state == ReCoNState.FAILED, \
            f"Terminal with default measurement should fail, got {terminal.state}"
    
    def test_explicit_measurement_required_for_confirmation(self):
        """Test that terminals require explicit measurement_fn for confirmation."""
        terminal = ReCoNNode("explicit_terminal", "terminal")
        
        # Without explicit measurement_fn, should fail
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.FAILED, "Should fail without explicit measurement"
        
        # With explicit measurement_fn above threshold, should confirm
        terminal.reset()
        terminal.measurement_fn = lambda env: 0.9  # Above threshold (0.8)
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.CONFIRMED, "Should confirm with explicit high measurement"
        
        # With explicit measurement_fn below threshold, should fail
        terminal.reset()
        terminal.measurement_fn = lambda env: 0.3  # Below threshold (0.8)
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.FAILED, "Should fail with explicit low measurement"
    
    @given(
        measurement_value=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        threshold=st.floats(min_value=0.1, max_value=1.5, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_threshold_behavior_property(self, measurement_value, threshold):
        """Property test: Terminal threshold behavior should be consistent."""
        terminal = ReCoNNode("threshold_test", "terminal")
        terminal.transition_threshold = threshold
        terminal.measurement_fn = lambda env: measurement_value
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        
        # Behavior should be deterministic based on measurement vs threshold
        if measurement_value > threshold:
            assert terminal.state == ReCoNState.CONFIRMED, \
                f"measurement {measurement_value} > threshold {threshold} should confirm"
            assert terminal.activation == 1.0, "Confirmed terminal should have activation 1.0"
        else:
            assert terminal.state == ReCoNState.FAILED, \
                f"measurement {measurement_value} <= threshold {threshold} should fail"
            assert terminal.activation == 0.0, "Failed terminal should have activation 0.0"
    
    def test_terminal_measurement_types(self):
        """Test different types of measurement functions."""
        terminal = ReCoNNode("measurement_test", "terminal")
        
        # Test different measurement function types
        measurement_functions = [
            (lambda env: 0.9, ReCoNState.CONFIRMED, "High constant"),
            (lambda env: 0.3, ReCoNState.FAILED, "Low constant"),
            (lambda env: env if env is not None else 0.7, ReCoNState.FAILED, "Environment-dependent (None)"),
            (lambda env: 1.0, ReCoNState.CONFIRMED, "Maximum value"),
            (lambda env: 0.0, ReCoNState.FAILED, "Minimum value")
        ]
        
        for measurement_fn, expected_state, description in measurement_functions:
            terminal.reset()
            terminal.measurement_fn = measurement_fn
            
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            terminal.update_state(inputs)
            
            print(f"{description}: measurement={measurement_fn(None):.1f} → {terminal.state.value}")
            assert terminal.state == expected_state, \
                f"{description}: expected {expected_state.value}, got {terminal.state.value}"
    
    def test_terminal_with_neural_model(self):
        """Test terminal with neural model measurement."""
        import torch.nn as nn
        
        terminal = ReCoNNode("neural_terminal", "terminal")
        
        # Create simple neural model
        model = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        
        # Initialize to produce high output
        with torch.no_grad():
            model[0].weight.fill_(5.0)  # High weight
            model[0].bias.fill_(2.0)    # High bias
        
        terminal.neural_model = model
        
        # Should use neural model for measurement
        measurement = terminal.measure()
        print(f"Neural model measurement: {measurement}")
        assert measurement > 0.8, "Neural model should produce high measurement"
        
        # Should confirm when requested
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.CONFIRMED, "Neural terminal should confirm with high output"
    
    def test_configurable_threshold(self):
        """Test that threshold is configurable and affects behavior."""
        terminal = ReCoNNode("config_terminal", "terminal")
        
        # Fixed measurement, variable threshold
        terminal.measurement_fn = lambda env: 0.6
        
        # Test with different thresholds
        threshold_tests = [
            (0.5, ReCoNState.CONFIRMED, "Low threshold"),
            (0.6, ReCoNState.FAILED, "Equal threshold (should fail)"),
            (0.7, ReCoNState.FAILED, "High threshold"),
            (0.1, ReCoNState.CONFIRMED, "Very low threshold")
        ]
        
        for threshold, expected_state, description in threshold_tests:
            terminal.reset()
            terminal.transition_threshold = threshold
            
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            terminal.update_state(inputs)
            
            print(f"{description}: threshold={threshold}, measurement=0.6 → {terminal.state.value}")
            assert terminal.state == expected_state, \
                f"{description}: expected {expected_state.value}, got {terminal.state.value}"


class TestTerminalInGraphContext:
    """Test terminal behavior in graph execution context."""
    
    def test_terminal_requires_explicit_setup_in_graph(self):
        """Test that terminals in graphs require explicit setup for confirmation."""
        graph = ReCoNGraph()
        
        # Create simple hierarchy
        graph.add_node("Parent", "script")
        graph.add_node("Child", "script") 
        graph.add_node("Terminal", "terminal")
        
        graph.add_link("Parent", "Child", "sub")
        graph.add_link("Child", "Terminal", "sub")
        
        # Execute without setting up terminal measurement
        graph.request_root("Parent")
        
        for step in range(10):
            graph.propagate_step()
            if graph.is_completed():
                break
        
        # Terminal should fail with default neutral measurement
        terminal = graph.get_node("Terminal")
        assert terminal.state == ReCoNState.FAILED, \
            f"Terminal should fail with default measurement, got {terminal.state}"
        
        # Parent should eventually fail due to child failure
        parent = graph.get_node("Parent")
        assert parent.state == ReCoNState.FAILED, \
            f"Parent should fail when terminal fails, got {parent.state}"
    
    def test_terminal_explicit_confirmation_in_graph(self):
        """Test that terminals can be explicitly configured to confirm."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Child", "script")
        graph.add_node("Terminal", "terminal")
        
        graph.add_link("Parent", "Child", "sub")
        graph.add_link("Child", "Terminal", "sub")
        
        # Set terminal to confirm explicitly
        terminal = graph.get_node("Terminal")
        terminal.measurement_fn = lambda env: 0.9  # Above threshold
        
        graph.request_root("Parent")
        
        for step in range(10):
            graph.propagate_step()
            if graph.is_completed():
                break
        
        # Terminal should confirm with explicit high measurement
        assert terminal.state == ReCoNState.CONFIRMED, \
            f"Terminal should confirm with explicit measurement, got {terminal.state}"
        
        # Parent should eventually confirm due to child success
        parent = graph.get_node("Parent")
        assert parent.state in [ReCoNState.CONFIRMED, ReCoNState.TRUE], \
            f"Parent should succeed when terminal confirms, got {parent.state}"
    
    def test_terminal_threshold_edge_cases(self):
        """Test edge cases around threshold boundary."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Terminal", "terminal")
        graph.add_link("Parent", "Terminal", "sub")
        
        # Test exact threshold boundary
        terminal = graph.get_node("Terminal")
        terminal.transition_threshold = 0.8
        
        edge_cases = [
            (0.8, ReCoNState.FAILED, "Exactly at threshold should fail"),
            (0.800001, ReCoNState.CONFIRMED, "Just above threshold should confirm"),
            (0.799999, ReCoNState.FAILED, "Just below threshold should fail")
        ]
        
        for measurement, expected_state, description in edge_cases:
            graph.reset()
            terminal.measurement_fn = lambda env: measurement
            
            graph.request_root("Parent")
            
            for _ in range(5):
                graph.propagate_step()
                if graph.is_completed():
                    break
            
            print(f"{description}: measurement={measurement} → {terminal.state.value}")
            assert terminal.state == expected_state, \
                f"{description}: expected {expected_state.value}, got {terminal.state.value}"


class TestTerminalSemanticNeutrality:
    """Test that terminals don't imply semantics not stated in paper."""
    
    def test_no_implicit_confirmation_semantics(self):
        """Test that terminals don't imply automatic success semantics."""
        # The paper doesn't specify what terminals should do by default
        # Our implementation should be neutral and require explicit configuration
        
        terminal = ReCoNNode("semantic_test", "terminal")
        
        # Default behavior should not imply success
        default_measurement = terminal.measure()
        assert default_measurement < terminal.transition_threshold, \
            "Default measurement should not imply automatic success"
        
        # Should require explicit decision about measurement
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        
        # Default should be failure (neutral, requires explicit setup)
        assert terminal.state == ReCoNState.FAILED, \
            "Default terminal behavior should be neutral (fail), requiring explicit setup"
        
        print("✅ Terminal requires explicit measurement configuration")
        print("   This avoids implying semantics not specified in the paper")
    
    def test_measurement_function_documentation(self):
        """Document proper terminal measurement function usage."""
        terminal = ReCoNNode("doc_terminal", "terminal")
        
        # Examples of proper measurement function usage
        measurement_examples = {
            "sensor_reading": lambda env: env.get("sensor_value", 0.0) if env else 0.5,
            "neural_classification": lambda env: 0.95 if env and env.get("class") == "target" else 0.1,
            "threshold_detector": lambda env: 1.0 if env and env.get("value", 0) > 10 else 0.0,
            "probabilistic": lambda env: 0.85,  # High confidence
            "always_fail": lambda env: 0.0,     # Always below threshold
            "always_confirm": lambda env: 1.0   # Always above threshold
        }
        
        for name, measurement_fn in measurement_examples.items():
            terminal.reset()
            terminal.measurement_fn = measurement_fn
            
            measurement = terminal.measure()
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            terminal.update_state(inputs)
            
            expected_state = ReCoNState.CONFIRMED if measurement > 0.8 else ReCoNState.FAILED
            
            print(f"{name}: measurement={measurement:.1f} → {terminal.state.value}")
            assert terminal.state == expected_state, \
                f"{name}: measurement {measurement} should lead to {expected_state.value}"
        
        print("\n✅ Measurement functions provide explicit semantic control")
        print("   Users must explicitly define what terminals should measure")
    
    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        measurement=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_threshold_logic_property(self, threshold, measurement):
        """Property test: Threshold logic should be consistent across all values."""
        terminal = ReCoNNode("property_terminal", "terminal")
        terminal.transition_threshold = threshold
        terminal.measurement_fn = lambda env: measurement
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        
        # Threshold logic should be consistent
        if measurement > threshold:
            assert terminal.state == ReCoNState.CONFIRMED, \
                f"measurement {measurement} > threshold {threshold} should confirm"
        else:
            assert terminal.state == ReCoNState.FAILED, \
                f"measurement {measurement} <= threshold {threshold} should fail"
    
    def test_terminal_paper_compliance(self):
        """Test that terminal behavior follows paper specifications."""
        # From paper: "A terminal node performs a measurement and has a state s ∈ {inactive, active, confirmed}"
        # Note: Paper shows "active" state for terminals, but our implementation uses "requested"
        
        terminal = ReCoNNode("paper_terminal", "terminal")
        
        # Test state transitions from paper
        # INACTIVE → measurement when requested
        assert terminal.state == ReCoNState.INACTIVE, "Should start inactive"
        
        # Request terminal
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        
        # With neutral measurement (0.5 < 0.8), should fail
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.FAILED, "Should fail with neutral measurement"
        
        # With high measurement, should confirm
        terminal.reset()
        terminal.measurement_fn = lambda env: 0.95
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.CONFIRMED, "Should confirm with high measurement"
        
        print("✅ Terminal behavior follows paper specification")
        print("   Measurement determines confirm/fail outcome")


class TestTerminalMessageBehavior:
    """Test terminal message behavior with threshold logic."""
    
    def test_terminal_message_depends_on_measurement(self):
        """Test that terminal messages depend on measurement outcome."""
        terminal = ReCoNNode("message_terminal", "terminal")
        
        # Test confirming terminal
        terminal.measurement_fn = lambda env: 0.9  # Above threshold
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        
        messages = terminal.get_outgoing_messages(inputs)
        assert messages.get("sur") == "confirm", \
            f"Confirmed terminal should send 'confirm', got '{messages.get('sur')}'"
        
        # Test failing terminal
        terminal.reset()
        terminal.measurement_fn = lambda env: 0.3  # Below threshold
        terminal.update_state(inputs)
        
        messages = terminal.get_outgoing_messages(inputs)
        assert "sur" not in messages or messages["sur"] is None, \
            f"Failed terminal should send no sur message, got '{messages.get('sur')}'"
    
    def test_terminal_only_sends_sur_messages(self):
        """Test that terminals only send sur messages per paper constraints."""
        terminal = ReCoNNode("sur_only_terminal", "terminal")
        terminal.measurement_fn = lambda env: 0.9  # High measurement
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        
        messages = terminal.get_outgoing_messages(inputs)
        
        # Should only send sur messages
        forbidden_messages = ["por", "ret", "sub"]
        for msg_type in forbidden_messages:
            assert msg_type not in messages or messages[msg_type] is None, \
                f"Terminal should not send {msg_type} messages, got '{messages.get(msg_type)}'"
        
        # Should send sur message when confirmed
        assert "sur" in messages, "Confirmed terminal should send sur message"
        assert messages["sur"] == "confirm", f"Should send 'confirm', got '{messages['sur']}'"
    
    @given(
        measurement_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=50)
    def test_terminal_measurement_consistency_property(self, measurement_values):
        """Property test: Terminal should behave consistently with same measurement."""
        terminal = ReCoNNode("consistency_terminal", "terminal")
        
        for measurement in measurement_values:
            terminal.reset()
            terminal.measurement_fn = lambda env: measurement
            
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            terminal.update_state(inputs)
            
            # Same measurement should always produce same state
            expected_state = ReCoNState.CONFIRMED if measurement > 0.8 else ReCoNState.FAILED
            assert terminal.state == expected_state, \
                f"Measurement {measurement} should consistently produce {expected_state.value}"
            
            # Messages should be consistent with state
            messages = terminal.get_outgoing_messages(inputs)
            if terminal.state == ReCoNState.CONFIRMED:
                assert messages.get("sur") == "confirm", "Confirmed should send confirm"
            elif terminal.state == ReCoNState.FAILED:
                assert "sur" not in messages or messages["sur"] is None, "Failed should send no sur"


class TestTerminalBreakingChangeImpact:
    """Test impact of changing default measurement from 1.0 to 0.5."""
    
    def test_breaking_change_documentation(self):
        """Document the impact of changing default terminal behavior."""
        # OLD behavior: terminals auto-confirmed (measurement = 1.0)
        # NEW behavior: terminals are neutral (measurement = 0.5, below threshold)
        
        terminal = ReCoNNode("breaking_test", "terminal")
        
        # New default behavior
        default_measurement = terminal.measure()
        assert default_measurement == 0.5, "New default should be 0.5"
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.FAILED, "New default should fail"
        
        # How to restore old behavior if needed
        terminal.reset()
        terminal.measurement_fn = lambda env: 1.0  # Explicit auto-confirm
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.CONFIRMED, "Explicit high measurement should confirm"
        
        print("📋 Breaking Change Impact:")
        print("  OLD: Terminals auto-confirmed (implied success semantics)")
        print("  NEW: Terminals are neutral, require explicit measurement_fn")
        print("  FIX: Add measurement_fn = lambda env: 1.0 for auto-confirm behavior")
        print("  BENEFIT: No implied semantics, explicit configuration required")
    
    def test_existing_code_migration_pattern(self):
        """Show migration pattern for existing code that relied on auto-confirm."""
        # Pattern for updating existing code
        
        # OLD pattern (relied on auto-confirm):
        # terminal = ReCoNNode("old", "terminal")
        # # Terminal would auto-confirm
        
        # NEW pattern (explicit measurement):
        terminal = ReCoNNode("new", "terminal")
        
        # Option 1: Always confirm
        terminal.measurement_fn = lambda env: 1.0
        
        # Option 2: Environment-based
        terminal.measurement_fn = lambda env: 0.9 if env and env.get("success") else 0.1
        
        # Option 3: Probabilistic
        import random
        terminal.measurement_fn = lambda env: 0.95 if random.random() > 0.1 else 0.2
        
        # Option 4: Neural model (already supported)
        # terminal.neural_model = some_trained_model
        
        print("✅ Migration patterns documented for explicit terminal configuration")
        print("   Existing code can be updated with appropriate measurement_fn")
        
        # Test that explicit configuration works
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        terminal.update_state(inputs)
        assert terminal.state == ReCoNState.CONFIRMED, "Explicit configuration should work"
