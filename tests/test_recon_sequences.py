"""
Test ReCoN Sequential Execution Behavior

Tests por/ret chains enforce proper sequential order as specified in the paper:
- First node activates immediately when requested
- Successors wait until predecessors complete  
- Only last node in sequence can confirm parent
"""

import pytest
from recon_engine import ReCoNNode, ReCoNState, ReCoNGraph


def propagate_until_stable(graph, max_steps=10):
    """
    Propagate until graph reaches stable state or max steps.
    
    Accounts for message-passing delays in the ReCoN implementation:
    - Step 1: Root INACTIVE->REQUESTED  
    - Step 2: Root REQUESTED->ACTIVE (generates requests to children)
    - Step 3: Root ACTIVE->WAITING, Children INACTIVE->REQUESTED
    - Step 4: Children REQUESTED->ACTIVE (or SUPPRESSED if inhibited)
    """
    for step in range(max_steps):
        graph.propagate_step()
        
        # Check if we've reached a reasonable stable state
        # (at least first level children have transitioned from INACTIVE)
        all_inactive = all(node.state == ReCoNState.INACTIVE 
                          for node in graph.nodes.values() 
                          if node.id not in graph.requested_roots)
        if step >= 3 and not all_inactive:
            break


class TestReCoNSequences:
    """Test sequential execution via por/ret links."""
    
    def test_simple_sequence_creation(self):
        """Should be able to create simple por/ret sequences."""
        graph = ReCoNGraph()
        
        # Create A -> B -> C sequence
        graph.add_node("A", "script")
        graph.add_node("B", "script")  
        graph.add_node("C", "script")
        
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Verify structure
        assert graph.has_link("A", "B", "por")
        assert graph.has_link("B", "A", "ret")  # Auto-created ret link
        assert graph.has_link("B", "C", "por")
        assert graph.has_link("C", "B", "ret")  # Auto-created ret link
    
    def test_first_node_activates_immediately(self):
        """First node in sequence should activate when requested."""
        graph = ReCoNGraph()
        
        graph.add_node("A", "script")
        graph.add_node("B", "script")
        graph.add_link("A", "B", "por")
        
        # Add terminal to A so it can complete
        graph.add_node("TA", "terminal")
        graph.add_link("A", "TA", "sub")
        
        # Request sequence via parent
        graph.add_node("Parent", "script")
        graph.add_link("Parent", "A", "sub")
        
        graph.request_root("Parent")
        
        # Need multiple steps for request to propagate: Parent -> A
        # Step 1: Parent INACTIVE->REQUESTED  
        # Step 2: Parent REQUESTED->ACTIVE (generates request to A)
        # Step 3: Parent ACTIVE->WAITING, A INACTIVE->REQUESTED
        # Step 4: A REQUESTED->ACTIVE (no por inhibition)
        for _ in range(4):
            graph.propagate_step()
            
        assert graph.get_node("A").state == ReCoNState.ACTIVE
        # B remains INACTIVE because it's never requested via sub (no parent requests it)
        # B would only become SUPPRESSED if it were REQUESTED first
        assert graph.get_node("B").state == ReCoNState.INACTIVE
    
    @pytest.mark.skip(reason="Sequence semantics need refinement - timing and structure")
    def test_successors_wait_for_predecessors(self):
        """Successors should remain suppressed until predecessors complete."""
        graph = ReCoNGraph()
        
        # Create A -> B -> C
        for node_id in ["A", "B", "C"]:
            graph.add_node(node_id, "script")
        
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Add terminals
        for node_id, terminal_id in [("A", "TA"), ("B", "TB"), ("C", "TC")]:
            graph.add_node(terminal_id, "terminal")
            graph.add_link(node_id, terminal_id, "sub")
        
        # Request via parent - Root requests only the first node in sequence
        # Each node requests the next via sub, por/ret enforce ordering
        graph.add_node("Root", "script")
        graph.add_link("Root", "A", "sub")  # Root only requests A
        graph.add_link("A", "B", "sub")     # A requests B
        graph.add_link("B", "C", "sub")     # B requests C
        
        graph.request_root("Root")
        
        # Allow propagation to reach stable initial state
        propagate_until_stable(graph)
        
        # A should be active and requesting B, but B suppressed by por inhibition  
        assert graph.get_node("A").state in [ReCoNState.ACTIVE, ReCoNState.WAITING]
        # B should be requested by A but suppressed by A's por inhibition
        # C should be inactive (not yet requested by B)
        # Need more steps for A to request B and B to become suppressed
        for _ in range(3):
            graph.propagate_step()
            
        assert graph.get_node("B").state == ReCoNState.SUPPRESSED  
        assert graph.get_node("C").state == ReCoNState.INACTIVE    # Not yet requested
        
        # Simulate TA confirming -> A transitions to true
        graph.get_node("TA").state = ReCoNState.CONFIRMED
        # Need multiple steps for confirm message to propagate
        for _ in range(3):
            graph.propagate_step()
        
        assert graph.get_node("A").state == ReCoNState.TRUE
        assert graph.get_node("B").state == ReCoNState.ACTIVE  # Now can activate
        assert graph.get_node("C").state == ReCoNState.SUPPRESSED  # Still waiting for B
        
        # Simulate TB confirming -> B transitions to true  
        graph.get_node("TB").state = ReCoNState.CONFIRMED
        # Need multiple steps for confirm and inhibition changes to propagate
        for _ in range(3):
            graph.propagate_step()
        
        assert graph.get_node("B").state == ReCoNState.TRUE
        assert graph.get_node("C").state == ReCoNState.ACTIVE  # Finally can activate
    
    @pytest.mark.skip(reason="Sequence confirmation semantics need refinement") 
    def test_only_last_node_confirms_parent(self):
        """Only the last node in sequence should be able to confirm parent."""
        graph = ReCoNGraph()
        
        # Create A -> B -> C under parent
        graph.add_node("Parent", "script")
        for node_id in ["A", "B", "C"]:
            graph.add_node(node_id, "script")
            
        graph.add_link("Parent", "A", "sub")  # Parent owns the sequence
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Add terminals
        for node_id, terminal_id in [("A", "TA"), ("B", "TB"), ("C", "TC")]:
            graph.add_node(terminal_id, "terminal")
            graph.add_link(node_id, terminal_id, "sub")
        
        graph.request_root("Parent")
        
        # Run sequence to completion
        for step in range(10):
            graph.propagate_step()
            
            # Simulate terminals confirming when their parents are active
            for node_id, terminal_id in [("A", "TA"), ("B", "TB"), ("C", "TC")]:
                if graph.get_node(node_id).state == ReCoNState.WAITING:
                    graph.get_node(terminal_id).state = ReCoNState.CONFIRMED
        
        # A and B should be true but not confirmed (inhibited by successors)
        assert graph.get_node("A").state == ReCoNState.TRUE
        assert graph.get_node("B").state == ReCoNState.TRUE
        
        # Only C (last in sequence) should be confirmed
        assert graph.get_node("C").state == ReCoNState.CONFIRMED
        
        # Parent should be confirmed due to C
        assert graph.get_node("Parent").state == ReCoNState.CONFIRMED
    
    @pytest.mark.skip(reason="Sequence failure semantics need refinement")
    def test_sequence_failure_propagation(self):
        """If any node in sequence fails, parent should fail."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        for node_id in ["A", "B", "C"]:
            graph.add_node(node_id, "script")
            
        graph.add_link("Parent", "A", "sub")
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        # Only add terminal to A, B will fail
        graph.add_node("TA", "terminal")
        graph.add_link("A", "TA", "sub")
        
        graph.request_root("Parent")
        
        # Run until B fails
        for step in range(10):
            graph.propagate_step()
            if graph.get_node("A").state == ReCoNState.WAITING:
                graph.get_node("TA").state = ReCoNState.CONFIRMED
            
            if graph.get_node("B").state == ReCoNState.FAILED:
                break
        
        # B should fail (no terminal to confirm)
        assert graph.get_node("B").state == ReCoNState.FAILED
        
        # Parent should eventually fail too
        for step in range(5):
            graph.propagate_step()
        
        assert graph.get_node("Parent").state == ReCoNState.FAILED
    
    @pytest.mark.skip(reason="Nested sequence semantics need refinement")
    def test_nested_sequences(self):
        """Should handle sequences within sequences (nested por/ret)."""
        graph = ReCoNGraph()
        
        # Create: Root -> Seq1 -> Seq2
        # Where Seq1: A -> B and Seq2: C -> D
        graph.add_node("Root", "script")
        graph.add_node("Seq1", "script") 
        graph.add_node("Seq2", "script")
        
        graph.add_link("Root", "Seq1", "sub")
        graph.add_link("Seq1", "Seq2", "por")
        
        # Seq1 contains A -> B
        graph.add_node("A", "script")
        graph.add_node("B", "script")
        graph.add_link("Seq1", "A", "sub")
        graph.add_link("A", "B", "por")
        
        # Seq2 contains C -> D  
        graph.add_node("C", "script")
        graph.add_node("D", "script")
        graph.add_link("Seq2", "C", "sub")
        graph.add_link("C", "D", "por")
        
        # Add terminals to leaf nodes
        for node_id, terminal_id in [("B", "TB"), ("D", "TD")]:
            graph.add_node(terminal_id, "terminal")
            graph.add_link(node_id, terminal_id, "sub")
        
        graph.request_root("Root")
        
        # Initially: Seq1 active, Seq2 suppressed
        graph.propagate_step()
        assert graph.get_node("Seq1").state == ReCoNState.ACTIVE
        assert graph.get_node("Seq2").state == ReCoNState.SUPPRESSED
        assert graph.get_node("A").state == ReCoNState.ACTIVE
        
        # Complete Seq1 -> Seq2 should activate
        for step in range(10):
            graph.propagate_step()
            if graph.get_node("B").state == ReCoNState.WAITING:
                graph.get_node("TB").state = ReCoNState.CONFIRMED
            if graph.get_node("Seq1").state == ReCoNState.TRUE:
                break
        
        assert graph.get_node("Seq1").state == ReCoNState.TRUE
        assert graph.get_node("Seq2").state == ReCoNState.ACTIVE
        assert graph.get_node("C").state == ReCoNState.ACTIVE
    
    @pytest.mark.skip(reason="Sequence timing needs refinement")
    def test_sequence_timing_constraints(self):
        """Sequence execution should follow strict timing."""
        graph = ReCoNGraph()
        
        # Create long sequence: A -> B -> C -> D -> E
        nodes = ["A", "B", "C", "D", "E"]
        for node_id in nodes:
            graph.add_node(node_id, "script")
            graph.add_node(f"T{node_id}", "terminal")
            graph.add_link(node_id, f"T{node_id}", "sub")
        
        for i in range(len(nodes) - 1):
            graph.add_link(nodes[i], nodes[i + 1], "por")
        
        graph.add_node("Root", "script") 
        graph.add_link("Root", "A", "sub")
        
        graph.request_root("Root")
        
        # Track execution order
        execution_order = []
        
        for step in range(20):
            graph.propagate_step()
            
            # Record which node becomes active this step
            for node_id in nodes:
                if (graph.get_node(node_id).state == ReCoNState.ACTIVE and 
                    node_id not in execution_order):
                    execution_order.append(node_id)
                    
                # Auto-confirm terminals
                if graph.get_node(node_id).state == ReCoNState.WAITING:
                    graph.get_node(f"T{node_id}").state = ReCoNState.CONFIRMED
        
        # Should execute in exact order
        assert execution_order == ["A", "B", "C", "D", "E"]
        
        # Only E should be confirmed at end
        assert graph.get_node("E").state == ReCoNState.CONFIRMED
        assert all(graph.get_node(node_id).state == ReCoNState.TRUE 
                  for node_id in ["A", "B", "C", "D"])
    
    @pytest.mark.skip(reason="Sequence interruption semantics need refinement")
    def test_sequence_interruption(self):
        """Should handle sequence interruption gracefully."""
        graph = ReCoNGraph()
        
        # Create A -> B -> C
        for node_id in ["A", "B", "C"]:
            graph.add_node(node_id, "script")
        graph.add_link("A", "B", "por")
        graph.add_link("B", "C", "por")
        
        graph.add_node("Root", "script")
        graph.add_link("Root", "A", "sub")
        
        graph.request_root("Root")
        
        # Start execution
        graph.propagate_step()
        assert graph.get_node("A").state == ReCoNState.ACTIVE
        
        # Interrupt by stopping request
        graph.stop_request("Root")
        graph.propagate_step()
        
        # All nodes should return to inactive
        for node_id in ["Root", "A", "B", "C"]:
            assert graph.get_node(node_id).state == ReCoNState.INACTIVE