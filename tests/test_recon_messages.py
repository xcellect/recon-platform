"""
Test ReCoN Message Passing Behavior

Tests message semantics based on Table 1 from the ReCoN paper:
- inhibit_request, inhibit_confirm, wait, confirm, fail messages
- Proper routing via por, ret, sub, sur links
"""

import pytest
from recon_engine import ReCoNNode, ReCoNState, ReCoNGraph, MessageType


class TestReCoNMessages:
    """Test message passing according to Table 1 specification."""
    
    def test_message_types_defined(self):
        """All message types from paper should be defined."""
        expected_messages = {
            "inhibit_request", "inhibit_confirm", "wait", "confirm", "fail"
        }
        
        # Check MessageType enum has all required values
        message_names = {msg.name.lower() for msg in MessageType}
        assert expected_messages.issubset(message_names)
    
    @pytest.mark.parametrize("state,expected_por,expected_ret,expected_sub,expected_sur", [
        # From Table 1: Message passing in ReCoNs
        (ReCoNState.INACTIVE, None, None, None, None),
        (ReCoNState.REQUESTED, "inhibit_request", "inhibit_confirm", None, "wait"),
        (ReCoNState.ACTIVE, "inhibit_request", "inhibit_confirm", "request", "wait"),
        (ReCoNState.SUPPRESSED, "inhibit_request", "inhibit_confirm", None, None),
        (ReCoNState.WAITING, "inhibit_request", "inhibit_confirm", "request", "wait"),
        (ReCoNState.TRUE, None, "inhibit_confirm", None, None),
        (ReCoNState.CONFIRMED, None, "inhibit_confirm", None, "confirm"),
        (ReCoNState.FAILED, "inhibit_request", "inhibit_confirm", None, None),
    ])
    def test_state_message_mapping(self, state, expected_por, expected_ret, expected_sub, expected_sur):
        """Each state should send correct messages via each link type."""
        node = ReCoNNode("test_node")
        node.state = state
        
        inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
        signals = node.get_outgoing_messages(inputs)
        
        assert signals.get("por") == expected_por
        assert signals.get("ret") == expected_ret  
        assert signals.get("sub") == expected_sub
        assert signals.get("sur") == expected_sur
    
    def test_inhibit_request_propagation(self):
        """inhibit_request should prevent successors from activating."""
        graph = ReCoNGraph()
        
        # Create proper hierarchy with sequence: Parent -> [A, B, C] where A->B->C
        parent = graph.add_node("Parent", "script")
        node_a = graph.add_node("A", "script")
        node_b = graph.add_node("B", "script") 
        node_c = graph.add_node("C", "script")
        
        # All are children of parent (receive requests)
        graph.add_link("Parent", "A", "sub")
        graph.add_link("Parent", "B", "sub")
        graph.add_link("Parent", "C", "sub")
        
        # They form a sequence
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Request root Parent
        graph.request_root("Parent")
        
        # Parent activates and sends requests
        graph.propagate_step()
        assert graph.get_node("Parent").state == ReCoNState.ACTIVE
        
        # Children receive requests
        graph.propagate_step()
        assert graph.get_node("A").state == ReCoNState.REQUESTED
        assert graph.get_node("B").state == ReCoNState.REQUESTED
        assert graph.get_node("C").state == ReCoNState.REQUESTED
        
        # A should activate and inhibit B and C
        graph.propagate_step()
        assert graph.get_node("A").state == ReCoNState.ACTIVE
        assert graph.get_node("B").state == ReCoNState.SUPPRESSED
        assert graph.get_node("C").state == ReCoNState.SUPPRESSED
    
    def test_inhibit_confirm_propagation(self):
        """inhibit_confirm should prevent predecessors from confirming prematurely."""
        graph = ReCoNGraph()
        
        # Create ret chain: A <- B <- C  
        node_a = graph.add_node("A", "script")
        node_b = graph.add_node("B", "script")
        node_c = graph.add_node("C", "script")
        
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Manually set B to true state
        graph.get_node("B").state = ReCoNState.TRUE
        
        # B should send inhibit_confirm to A via ret
        messages = graph.get_node("B").get_outgoing_messages({})
        assert messages["ret"] == "inhibit_confirm"
        
        # A should not be able to confirm while receiving inhibit_confirm
        graph.get_node("A").state = ReCoNState.TRUE
        inputs = {"ret": -1.0}  # inhibit_confirm signal
        assert not graph.get_node("A").can_confirm(inputs)
    
    def test_wait_message_keeps_parent_pending(self):
        """wait messages should prevent parent from failing."""
        graph = ReCoNGraph()
        
        # Create hierarchy: parent -> child
        parent = graph.add_node("parent", "script")
        child = graph.add_node("child", "script")
        
        graph.add_link("parent", "child", "sub")
        
        # Request parent to start execution
        graph.request_root("parent")
        
        # Parent goes to ACTIVE
        graph.propagate_step()
        assert graph.get_node("parent").state == ReCoNState.ACTIVE
        
        # Child receives request and goes to REQUESTED
        graph.propagate_step()
        assert graph.get_node("parent").state == ReCoNState.WAITING
        assert graph.get_node("child").state == ReCoNState.REQUESTED
        
        # Child goes to ACTIVE and sends wait
        graph.propagate_step()
        assert graph.get_node("child").state == ReCoNState.ACTIVE
        messages = graph.get_node("child").get_outgoing_messages({})
        assert messages["sur"] == "wait"
        
        # Parent should remain in waiting state while child is active
        assert graph.get_node("parent").state == ReCoNState.WAITING
    
    def test_confirm_message_activates_parent(self):
        """confirm messages should cause parent to transition to true/confirmed."""
        graph = ReCoNGraph()
        
        parent = graph.add_node("parent", "script")
        child = graph.add_node("child", "script")
        terminal = graph.add_node("terminal", "terminal")
        
        graph.add_link("parent", "child", "sub")
        graph.add_link("child", "terminal", "sub")
        
        # Set terminal to confirm explicitly (neutral default now requires explicit setup)
        terminal.measurement_fn = lambda env: 1.0
        
        # Request parent to start execution
        graph.request_root("parent")
        
        # Run until child is waiting for terminal
        for _ in range(5):
            graph.propagate_step()
            if graph.get_node("child").state == ReCoNState.WAITING:
                break
        
        # Terminal confirms
        graph.get_node("terminal").state = ReCoNState.CONFIRMED
        messages = graph.get_node("terminal").get_outgoing_messages({})
        assert messages["sur"] == "confirm"
        
        # Propagate confirmation
        graph.propagate_step()
        
        # Child should transition to TRUE
        assert graph.get_node("child").state == ReCoNState.TRUE
        
        # Parent should also transition toward confirmation (may take a few steps)
        for _ in range(3):
            graph.propagate_step()
            if graph.get_node("parent").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]:
                break
        assert graph.get_node("parent").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
    
    def test_fail_propagation(self):
        """Failed children should cause parent to fail if no others are active."""
        graph = ReCoNGraph()
        
        parent = graph.add_node("parent", "script")
        child1 = graph.add_node("child1", "script")
        child2 = graph.add_node("child2", "script")
        
        graph.add_link("parent", "child1", "sub")
        graph.add_link("parent", "child2", "sub")
        
        # Request parent
        graph.request_root("parent")
        
        # Run until parent is waiting for children
        for _ in range(3):
            graph.propagate_step()
            if graph.get_node("parent").state == ReCoNState.WAITING:
                break
        
        # Both children should be at least requested
        assert graph.get_node("child1").state != ReCoNState.INACTIVE
        assert graph.get_node("child2").state != ReCoNState.INACTIVE
        
        # Force both children to fail (no terminals)
        graph.get_node("child1").state = ReCoNState.FAILED
        graph.get_node("child2").state = ReCoNState.FAILED
        
        # Parent should detect no children waiting and fail
        inputs = {"sur": 0.0}  # No wait or confirm signals
        graph.get_node("parent").update_state(inputs)
        
        assert graph.get_node("parent").state == ReCoNState.FAILED


class TestMessagePropagationIntegration:
    """Test message propagation in complex scenarios."""
    
    def test_sequence_message_flow(self):
        """Test complete message flow in a por/ret sequence."""
        graph = ReCoNGraph()
        
        # Create proper hierarchy: Parent -> [A, B, C] where A->B->C sequence
        parent = graph.add_node("Parent", "script")
        for node_id in ["A", "B", "C"]:
            graph.add_node(node_id, "script")
            graph.add_link("Parent", node_id, "sub")  # All are children of parent
        
        # Create sequence constraints
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Add terminal to C
        graph.add_node("T", "terminal")
        graph.add_link("C", "T", "sub")
        
        graph.request_root("Parent")
        
        # Parent activates
        graph.propagate_step()
        assert graph.get_node("Parent").state == ReCoNState.ACTIVE
        
        # All children get requested
        graph.propagate_step()
        assert graph.get_node("A").state == ReCoNState.REQUESTED
        assert graph.get_node("B").state == ReCoNState.REQUESTED
        assert graph.get_node("C").state == ReCoNState.REQUESTED
        
        # A activates, B and C are suppressed by sequence
        graph.propagate_step()
        assert graph.get_node("A").state == ReCoNState.ACTIVE
        assert graph.get_node("B").state == ReCoNState.SUPPRESSED
        assert graph.get_node("C").state == ReCoNState.SUPPRESSED
        
        # Per paper, script nodes must have a sub child. Provide A's child now
        graph.add_node("TA", "terminal")
        graph.add_link("A", "TA", "sub")
        # Next step: A requests its child and goes to WAITING
        graph.propagate_step()
        assert graph.get_node("A").state in [ReCoNState.WAITING, ReCoNState.ACTIVE]

        # Confirm A's terminal to allow A -> TRUE/CONFIRMED
        if graph.get_node("A").state == ReCoNState.WAITING:
            terminal = graph.get_node("TA")
            terminal.state = ReCoNState.CONFIRMED
            terminal.activation = 1.0  # Set activation for proper propagation
        # Allow for message generation and delivery across steps
        graph.propagate_step()
        graph.propagate_step()
        assert graph.get_node("A").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
        
        # Now B can activate
        graph.propagate_step()
        assert graph.get_node("B").state == ReCoNState.ACTIVE
    
    def test_hierarchy_message_flow(self):
        """Test message flow in sub/sur hierarchy."""
        graph = ReCoNGraph()
        
        # Create hierarchy: Root -> [Child1, Child2]
        graph.add_node("Root", "script")
        graph.add_node("Child1", "script") 
        graph.add_node("Child2", "script")
        
        graph.add_link("Root", "Child1", "sub")
        graph.add_link("Root", "Child2", "sub")
        
        # Add terminals with explicit confirmation setup
        graph.add_node("T1", "terminal")
        graph.add_node("T2", "terminal")
        graph.add_link("Child1", "T1", "sub")
        graph.add_link("Child2", "T2", "sub")
        
        # Set terminals to confirm explicitly (neutral default now requires explicit setup)
        graph.get_node("T1").measurement_fn = lambda env: 1.0
        graph.get_node("T2").measurement_fn = lambda env: 1.0
        
        graph.request_root("Root")
        
        # Root becomes ACTIVE first
        graph.propagate_step()
        assert graph.get_node("Root").state == ReCoNState.ACTIVE
        
        # Root goes to WAITING, children get requested
        graph.propagate_step()
        assert graph.get_node("Root").state == ReCoNState.WAITING
        assert graph.get_node("Child1").state == ReCoNState.REQUESTED
        assert graph.get_node("Child2").state == ReCoNState.REQUESTED
        
        # Children become ACTIVE
        graph.propagate_step()
        assert graph.get_node("Child1").state == ReCoNState.ACTIVE
        assert graph.get_node("Child2").state == ReCoNState.ACTIVE
        
        # Children go to WAITING to request their terminals
        graph.propagate_step()
        assert graph.get_node("Child1").state == ReCoNState.WAITING
        assert graph.get_node("Child2").state == ReCoNState.WAITING
        
        # Terminals auto-confirm
        graph.propagate_step()
        assert graph.get_node("T1").state == ReCoNState.CONFIRMED
        assert graph.get_node("T2").state == ReCoNState.CONFIRMED
        
        # Children receive confirmations and complete
        graph.propagate_step()
        assert graph.get_node("Child1").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
        assert graph.get_node("Child2").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
        
        # Root receives confirmation from children
        graph.propagate_step()
        assert graph.get_node("Root").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
    
    def test_message_timing_consistency(self):
        """Messages should be processed consistently across propagation steps.""" 
        graph = ReCoNGraph()
        
        # Create simple parent-child
        graph.add_node("P", "script")
        graph.add_node("C", "script")
        graph.add_link("P", "C", "sub")
        
        graph.request_root("P")
        
        # Track message consistency over multiple steps
        for step in range(5):
            old_states = {nid: graph.get_node(nid).state for nid in ["P", "C"]}
            graph.propagate_step()
            new_states = {nid: graph.get_node(nid).state for nid in ["P", "C"]}
            
            # States should change predictably
            for node_id in ["P", "C"]:
                old_state = old_states[node_id]
                new_state = new_states[node_id]
                
                # No invalid transitions
                invalid_transitions = [
                    (ReCoNState.CONFIRMED, ReCoNState.FAILED),
                    (ReCoNState.FAILED, ReCoNState.CONFIRMED),
                    (ReCoNState.INACTIVE, ReCoNState.CONFIRMED)
                ]
                assert (old_state, new_state) not in invalid_transitions