"""
Property-Based Tests for Table 1 Compliance

Verifies that ReCoN node message passing strictly follows Table 1 from
"Request Confirmation Networks for Neuro-Symbolic Script Execution"
across all possible input combinations using property-based testing.

Table 1: Message passing in ReCoNs
| Unit state    | POR               | RET               | SUB     | SUR       |
|---------------|-------------------|-------------------|---------|-----------|
| inactive (∅)  | -                 | -                 | -       | -         |
| requested (R) | inhibit request   | inhibit confirm   | -       | wait      |
| active (A)    | inhibit request   | inhibit confirm   | request | wait      |
| suppressed (S)| inhibit request   | inhibit confirm   | -       | -         |
| waiting (W)   | inhibit request   | inhibit confirm   | request | wait      |
| true (T)      | -                 | inhibit confirm   | -       | -         |
| confirmed (C) | -                 | inhibit confirm   | -       | confirm   |
| failed (F)    | inhibit request   | inhibit confirm   | -       | -         |
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from recon_engine import ReCoNNode, ReCoNState


class TestTable1PropertyCompliance:
    """Property-based tests verifying strict Table 1 compliance."""
    
    # Define the expected message mapping from Table 1
    EXPECTED_MESSAGES = {
        ReCoNState.INACTIVE: {
            "por": None, "ret": None, "sub": None, "sur": None
        },
        ReCoNState.REQUESTED: {
            "por": "inhibit_request", "ret": "inhibit_confirm", "sub": None, "sur": "wait"
        },
        ReCoNState.ACTIVE: {
            "por": "inhibit_request", "ret": "inhibit_confirm", "sub": "request", "sur": "wait"
        },
        ReCoNState.SUPPRESSED: {
            "por": "inhibit_request", "ret": "inhibit_confirm", "sub": None, "sur": None
        },
        ReCoNState.WAITING: {
            "por": "inhibit_request", "ret": "inhibit_confirm", "sub": "request", "sur": "wait"
        },
        ReCoNState.TRUE: {
            "por": None, "ret": "inhibit_confirm", "sub": None, "sur": None
        },
        ReCoNState.CONFIRMED: {
            "por": None, "ret": "inhibit_confirm", "sub": None, "sur": "confirm"
        },
        ReCoNState.FAILED: {
            "por": "inhibit_request", "ret": "inhibit_confirm", "sub": None, "sur": None
        }
    }
    
    @given(
        state=st.sampled_from(list(ReCoNState)),
        sub_activation=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        por_activation=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        ret_activation=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        sur_activation=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=500)  # Test many combinations
    def test_table1_message_mapping_property(self, state, sub_activation, por_activation, ret_activation, sur_activation):
        """Property test: For any state and input combination, messages must match Table 1."""
        # Create node and set to specific state
        node = ReCoNNode("property_test", "script")
        node.state = state
        
        # Prepare inputs
        inputs = {
            "sub": sub_activation,
            "por": por_activation, 
            "ret": ret_activation,
            "sur": sur_activation
        }
        
        # Get outgoing messages
        messages = node.get_outgoing_messages(inputs)
        
        # Get expected messages for this state
        expected = self.EXPECTED_MESSAGES[state]
        
        # Verify each message type matches Table 1
        for link_type in ["por", "ret", "sub", "sur"]:
            expected_msg = expected[link_type]
            actual_msg = messages.get(link_type)
            
            if expected_msg is None:
                # Table 1 shows "-" (no message)
                assert actual_msg is None or actual_msg == "", \
                    f"State {state.value}: {link_type} should send no message, got '{actual_msg}'"
            else:
                # Table 1 shows specific message
                assert actual_msg == expected_msg, \
                    f"State {state.value}: {link_type} should send '{expected_msg}', got '{actual_msg}'"
    
    @given(
        sub_activation=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        por_activation=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        ret_activation=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        sur_activation=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=200)
    def test_terminal_node_message_compliance(self, sub_activation, por_activation, ret_activation, sur_activation):
        """Property test: Terminal nodes should only send sur messages per Table 1."""
        terminal = ReCoNNode("terminal_test", "terminal")
        
        inputs = {
            "sub": sub_activation,
            "por": por_activation,
            "ret": ret_activation, 
            "sur": sur_activation
        }
        
        # Update terminal state
        terminal.update_state(inputs)
        
        # Get messages
        messages = terminal.get_outgoing_messages(inputs)
        
        # Terminal nodes should only send sur messages (per paper constraints)
        for link_type in ["por", "ret", "sub"]:
            assert link_type not in messages or messages[link_type] is None, \
                f"Terminal node should not send {link_type} messages, got '{messages.get(link_type)}'"
        
        # sur message should be appropriate for terminal state
        if terminal.state == ReCoNState.CONFIRMED:
            assert messages.get("sur") == "confirm", \
                f"CONFIRMED terminal should send 'confirm', got '{messages.get('sur')}'"
        elif terminal.state == ReCoNState.FAILED:
            assert "sur" not in messages or messages["sur"] is None, \
                f"FAILED terminal should send no sur message, got '{messages.get('sur')}'"
    
    @given(
        initial_state=st.sampled_from([ReCoNState.INACTIVE, ReCoNState.REQUESTED, ReCoNState.ACTIVE]),
        sub_request=st.booleans(),
        por_inhibit=st.booleans(),
        ret_inhibit=st.booleans()
    )
    @settings(max_examples=100)
    def test_state_transition_consistency(self, initial_state, sub_request, por_inhibit, ret_inhibit):
        """Property test: State transitions should be deterministic and follow paper rules."""
        node = ReCoNNode("transition_test", "script")
        node.state = initial_state
        
        # Convert booleans to activation values
        inputs = {
            "sub": 1.0 if sub_request else 0.0,
            "por": -1.0 if por_inhibit else 0.0,
            "ret": -1.0 if ret_inhibit else 0.0,
            "sur": 0.0  # No children responding
        }
        
        old_state = node.state
        messages_before = node.get_outgoing_messages(inputs)
        
        # Update state
        node.update_state(inputs)
        
        new_state = node.state
        messages_after = node.get_outgoing_messages(inputs)
        
        # Verify messages match the new state per Table 1
        expected_messages = self.EXPECTED_MESSAGES[new_state]
        
        for link_type in ["por", "ret", "sub", "sur"]:
            expected = expected_messages[link_type]
            actual = messages_after.get(link_type)
            
            if expected is None:
                assert actual is None or actual == "", \
                    f"State {new_state.value}: {link_type} should send nothing, got '{actual}'"
            else:
                assert actual == expected, \
                    f"State {new_state.value}: {link_type} should send '{expected}', got '{actual}'"
    
    @given(
        node_type=st.sampled_from(["script", "terminal"]),
        timing_mode=st.sampled_from(["discrete", "activation"]),
        activation_value=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        discrete_wait_steps=st.integers(min_value=1, max_value=10),
        sequence_wait_steps=st.integers(min_value=1, max_value=15),
        activation_decay_rate=st.floats(min_value=0.1, max_value=0.99),
        activation_failure_threshold=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(max_examples=100)
    def test_message_mapping_independence_from_config(self, node_type, timing_mode, activation_value,
                                                     discrete_wait_steps, sequence_wait_steps,
                                                     activation_decay_rate, activation_failure_threshold):
        """Property test: Message mapping should be independent of timing configuration."""
        node = ReCoNNode("config_test", node_type)
        
        # Configure timing with provided parameters
        if timing_mode == "discrete":
            node.configure_timing(
                mode="discrete",
                discrete_wait_steps=discrete_wait_steps,
                sequence_wait_steps=sequence_wait_steps
            )
        else:
            node.configure_timing(
                mode="activation",
                activation_decay_rate=activation_decay_rate,
                activation_failure_threshold=activation_failure_threshold
            )
        
        # Set node to a specific state
        test_states = [ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE]
        for state in test_states:
            node.state = state
            node.activation = activation_value
            
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.5}
            messages = node.get_outgoing_messages(inputs)
            
            # Messages should match Table 1 regardless of timing configuration
            expected = self.EXPECTED_MESSAGES[state]
            
            for link_type in ["por", "ret", "sub", "sur"]:
                if node_type == "terminal" and link_type in ["por", "ret", "sub"]:
                    # Terminals don't send these messages
                    continue
                    
                expected_msg = expected[link_type]
                actual_msg = messages.get(link_type)
                
                if expected_msg is None:
                    assert actual_msg is None or actual_msg == "", \
                        f"{node_type} {state.value} with {timing_mode}: {link_type} should send nothing"
                else:
                    assert actual_msg == expected_msg, \
                        f"{node_type} {state.value} with {timing_mode}: {link_type} should send '{expected_msg}'"
    
    def test_table1_reference_implementation(self):
        """Reference test: Manually verify Table 1 for each state with controlled inputs."""
        # This serves as a reference for the property tests
        
        test_cases = [
            # (state, expected_messages)
            (ReCoNState.INACTIVE, {"por": None, "ret": None, "sub": None, "sur": None}),
            (ReCoNState.REQUESTED, {"por": "inhibit_request", "ret": "inhibit_confirm", "sub": None, "sur": "wait"}),
            (ReCoNState.ACTIVE, {"por": "inhibit_request", "ret": "inhibit_confirm", "sub": "request", "sur": "wait"}),
            (ReCoNState.SUPPRESSED, {"por": "inhibit_request", "ret": "inhibit_confirm", "sub": None, "sur": None}),
            (ReCoNState.WAITING, {"por": "inhibit_request", "ret": "inhibit_confirm", "sub": "request", "sur": "wait"}),
            (ReCoNState.TRUE, {"por": None, "ret": "inhibit_confirm", "sub": None, "sur": None}),
            (ReCoNState.CONFIRMED, {"por": None, "ret": "inhibit_confirm", "sub": None, "sur": "confirm"}),
            (ReCoNState.FAILED, {"por": "inhibit_request", "ret": "inhibit_confirm", "sub": None, "sur": None})
        ]
        
        for state, expected_messages in test_cases:
            node = ReCoNNode(f"ref_test_{state.value}", "script")
            node.state = state
            
            # Use neutral inputs that don't trigger state changes
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.5}
            messages = node.get_outgoing_messages(inputs)
            
            print(f"\nState {state.value}:")
            print(f"  Expected: {expected_messages}")
            print(f"  Actual:   {messages}")
            
            for link_type, expected_msg in expected_messages.items():
                actual_msg = messages.get(link_type)
                
                if expected_msg is None:
                    assert actual_msg is None or actual_msg == "", \
                        f"State {state.value}: {link_type} should send nothing, got '{actual_msg}'"
                else:
                    assert actual_msg == expected_msg, \
                        f"State {state.value}: {link_type} should send '{expected_msg}', got '{actual_msg}'"


class TestTable1EdgeCases:
    """Test edge cases and boundary conditions for Table 1 compliance."""
    
    @given(
        sub_val=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        por_val=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        ret_val=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        sur_val=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
    )
    @settings(max_examples=200)
    def test_extreme_activation_values(self, sub_val, por_val, ret_val, sur_val):
        """Property test: Message mapping should work with extreme activation values."""
        node = ReCoNNode("extreme_test", "script")
        
        # Test with each possible state
        for state in ReCoNState:
            node.state = state
            
            inputs = {
                "sub": sub_val,
                "por": por_val,
                "ret": ret_val,
                "sur": sur_val
            }
            
            # Should not crash and should produce valid messages
            try:
                messages = node.get_outgoing_messages(inputs)
                
                # Messages should be valid types
                for link_type, message in messages.items():
                    assert link_type in ["por", "ret", "sub", "sur"], \
                        f"Invalid link type: {link_type}"
                    
                    if message is not None:
                        valid_messages = ["request", "inhibit_request", "inhibit_confirm", "wait", "confirm", "fail"]
                        assert message in valid_messages or message == "", \
                            f"Invalid message: '{message}' for {link_type}"
                            
            except Exception as e:
                pytest.fail(f"get_outgoing_messages crashed with state {state.value} and inputs {inputs}: {e}")
    
    @given(
        initial_state=st.sampled_from(list(ReCoNState)),
        steps=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_message_consistency_over_time(self, initial_state, steps):
        """Property test: Message mapping should be consistent across multiple steps."""
        node = ReCoNNode("consistency_test", "script")
        node.state = initial_state
        
        # Fixed inputs to avoid state changes
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.5}
        
        # Get initial messages
        initial_messages = node.get_outgoing_messages(inputs)
        
        # Messages should be consistent if state doesn't change
        for step in range(steps):
            current_state = node.state
            messages = node.get_outgoing_messages(inputs)
            
            # If state hasn't changed, messages should be identical
            if current_state == initial_state:
                assert messages == initial_messages, \
                    f"Messages changed without state change at step {step}: {initial_messages} -> {messages}"
    
    @given(
        node_type=st.sampled_from(["script", "terminal"]),
        state=st.sampled_from(list(ReCoNState))
    )
    @settings(max_examples=50)
    def test_node_type_message_constraints(self, node_type, state):
        """Property test: Node type should constrain which messages can be sent."""
        node = ReCoNNode("type_test", node_type)
        node.state = state
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.5}
        messages = node.get_outgoing_messages(inputs)
        
        if node_type == "terminal":
            # Terminal nodes should only send sur messages (per paper)
            for link_type in ["por", "ret", "sub"]:
                assert link_type not in messages or messages[link_type] is None or messages[link_type] == "", \
                    f"Terminal node should not send {link_type} messages, got '{messages.get(link_type)}'"
        
        # Script nodes can send all message types per their state
        # (already covered by main property test)
    
    @given(
        timing_mode=st.sampled_from(["discrete", "activation"]),
        discrete_steps=st.integers(min_value=1, max_value=20),
        decay_rate=st.floats(min_value=0.1, max_value=0.99),
        threshold=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(max_examples=50)
    def test_timing_config_does_not_affect_messages(self, timing_mode, discrete_steps, decay_rate, threshold):
        """Property test: Timing configuration should not affect message mapping."""
        node = ReCoNNode("timing_config_test", "script")
        
        # Configure timing randomly
        if timing_mode == "discrete":
            node.configure_timing(mode="discrete", discrete_wait_steps=discrete_steps)
        else:
            node.configure_timing(
                mode="activation",
                activation_decay_rate=decay_rate,
                activation_failure_threshold=threshold
            )
        
        # Test message mapping for each state
        for state in ReCoNState:
            node.state = state
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.5}
            
            messages = node.get_outgoing_messages(inputs)
            expected = TestTable1PropertyCompliance.EXPECTED_MESSAGES[state]
            
            # Messages should match Table 1 regardless of timing config
            for link_type in ["por", "ret", "sub", "sur"]:
                expected_msg = expected[link_type]
                actual_msg = messages.get(link_type)
                
                if expected_msg is None:
                    assert actual_msg is None or actual_msg == "", \
                        f"Timing config {timing_mode} should not affect message mapping for {state.value}"
                else:
                    assert actual_msg == expected_msg, \
                        f"Timing config {timing_mode} should not affect message mapping for {state.value}"


class TestTable1Invariants:
    """Test invariants that should hold across all states and inputs."""
    
    @given(
        state=st.sampled_from(list(ReCoNState)),
        inputs=st.fixed_dictionaries({
            "sub": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            "por": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            "ret": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            "sur": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)
        })
    )
    @settings(max_examples=300)
    def test_message_type_invariants(self, state, inputs):
        """Property test: Certain message patterns should hold across all states."""
        node = ReCoNNode("invariant_test", "script")
        node.state = state
        
        messages = node.get_outgoing_messages(inputs)
        
        # Invariant 1: Only specific message types should be sent
        valid_messages = {"request", "inhibit_request", "inhibit_confirm", "wait", "confirm", "fail", None, ""}
        for link_type, message in messages.items():
            assert message in valid_messages, \
                f"Invalid message type: '{message}' for {link_type}"
        
        # Invariant 2: inhibit_confirm is sent on ret for most active states (per Table 1)
        active_states = [ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.SUPPRESSED, 
                        ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED, ReCoNState.FAILED]
        if state in active_states:
            assert messages.get("ret") == "inhibit_confirm", \
                f"State {state.value} should send inhibit_confirm on ret, got '{messages.get('ret')}'"
        
        # Invariant 3: INACTIVE and TRUE states send minimal messages
        minimal_states = [ReCoNState.INACTIVE, ReCoNState.TRUE]
        if state in minimal_states:
            active_messages = [msg for msg in messages.values() if msg and msg != "inhibit_confirm"]
            if state == ReCoNState.INACTIVE:
                assert len(active_messages) == 0, \
                    f"INACTIVE should send no active messages, got {active_messages}"
            elif state == ReCoNState.TRUE:
                assert len(active_messages) == 0, \
                    f"TRUE should only send inhibit_confirm, got {active_messages}"
        
        # Invariant 4: Only CONFIRMED state sends confirm
        if messages.get("sur") == "confirm":
            assert state == ReCoNState.CONFIRMED, \
                f"Only CONFIRMED state should send confirm, but {state.value} sent it"
        
        # Invariant 5: Only ACTIVE and WAITING send sub requests
        if messages.get("sub") == "request":
            assert state in [ReCoNState.ACTIVE, ReCoNState.WAITING], \
                f"Only ACTIVE/WAITING should send sub requests, but {state.value} sent it"
    
    @given(
        num_nodes=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=20)
    def test_network_level_table1_compliance(self, num_nodes, seed):
        """Property test: Table 1 compliance should hold in network context."""
        from recon_engine import ReCoNGraph
        import random
        
        random.seed(seed)
        graph = ReCoNGraph()
        
        # Create random network structure
        node_ids = [f"N{i}" for i in range(num_nodes)]
        for node_id in node_ids:
            graph.add_node(node_id, "script")
        
        # Add some random links
        for i in range(min(3, num_nodes - 1)):
            source = random.choice(node_ids[:-1])
            target = random.choice(node_ids[1:])
            if source != target and not graph.has_link(source, target, "sub"):
                try:
                    graph.add_link(source, target, "sub")
                except ValueError:
                    pass  # Skip if link constraints violated
        
        # Set nodes to random states
        for node_id in node_ids:
            node = graph.get_node(node_id)
            node.state = random.choice(list(ReCoNState))
        
        # Verify all nodes follow Table 1
        for node_id in node_ids:
            node = graph.get_node(node_id)
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.5}
            messages = node.get_outgoing_messages(inputs)
            
            expected = TestTable1PropertyCompliance.EXPECTED_MESSAGES[node.state]
            
            for link_type in ["por", "ret", "sub", "sur"]:
                expected_msg = expected[link_type]
                actual_msg = messages.get(link_type)
                
                if expected_msg is None:
                    assert actual_msg is None or actual_msg == "", \
                        f"Network node {node_id} state {node.state.value}: {link_type} should send nothing"
                else:
                    assert actual_msg == expected_msg, \
                        f"Network node {node_id} state {node.state.value}: {link_type} should send '{expected_msg}'"
