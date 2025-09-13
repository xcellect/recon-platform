"""
Test ReCoN Hierarchical Script Behavior

Tests sub/sur hierarchical validation as specified in the paper:
- Parent requests all children via sub
- Children send wait signals while processing
- Parent confirms when any child confirms (alternatives)
- Parent fails when all children fail
"""

import pytest
from recon_engine import ReCoNNode, ReCoNState, ReCoNGraph


class TestReCoNHierarchies:
    """Test hierarchical execution via sub/sur links."""
    
    def test_simple_hierarchy_creation(self):
        """Should be able to create parent-child hierarchies."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Child1", "script")
        graph.add_node("Child2", "script")
        
        graph.add_link("Parent", "Child1", "sub")
        graph.add_link("Parent", "Child2", "sub")
        
        # Verify structure - sur links auto-created
        assert graph.has_link("Parent", "Child1", "sub")
        assert graph.has_link("Child1", "Parent", "sur")
        assert graph.has_link("Parent", "Child2", "sub")
        assert graph.has_link("Child2", "Parent", "sur")
    
    def test_parent_requests_all_children(self):
        """When parent activates, it should request all children simultaneously."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Child1", "script")
        graph.add_node("Child2", "script")
        graph.add_node("Child3", "script")
        
        for child_id in ["Child1", "Child2", "Child3"]:
            graph.add_link("Parent", child_id, "sub")
            # Add terminals so children can complete
            graph.add_node(f"T{child_id}", "terminal")
            graph.add_link(child_id, f"T{child_id}", "sub")
        
        graph.request_root("Parent")
        # Need multiple steps for messages to propagate through hierarchy
        for _ in range(4):
            graph.propagate_step()
        
        # Parent should be waiting, children should be waiting for their terminals
        assert graph.get_node("Parent").state == ReCoNState.WAITING
        # Children have terminals so they go ACTIVE -> WAITING after requesting terminals
        assert graph.get_node("Child1").state == ReCoNState.WAITING
        assert graph.get_node("Child2").state == ReCoNState.WAITING
        assert graph.get_node("Child3").state == ReCoNState.WAITING
    
    def test_children_send_wait_signals(self):
        """Active children should send wait signals to prevent parent failure."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Child", "script")
        terminal = graph.add_node("Terminal", "terminal")
        # Set terminal to not auto-confirm so Child stays in WAITING
        terminal.measurement_fn = lambda env: 0.0
        
        graph.add_link("Parent", "Child", "sub")
        graph.add_link("Child", "Terminal", "sub")
        
        graph.request_root("Parent")
        # Need multiple steps for child to be activated
        for _ in range(4):
            graph.propagate_step()
        
        # Child should send wait signal when active
        child = graph.get_node("Child")
        messages = child.get_outgoing_messages({})
        assert messages["sur"] == "wait"
        
        # Parent should remain waiting (not fail)
        parent = graph.get_node("Parent")
        assert parent.state == ReCoNState.WAITING
        
        # Simulate no wait signals
        inputs = {"sur": 0.0}
        parent.update_state(inputs)
        assert parent.state == ReCoNState.FAILED
    
    def test_any_child_confirms_parent(self):
        """Parent should confirm when ANY child confirms (OR semantics)."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Child1", "script") 
        graph.add_node("Child2", "script")
        graph.add_node("Child3", "script")
        
        for child_id in ["Child1", "Child2", "Child3"]:
            graph.add_link("Parent", child_id, "sub")
            graph.add_node(f"T{child_id}", "terminal")
            graph.add_link(child_id, f"T{child_id}", "sub")
            
            # Set custom measurement functions
            terminal = graph.get_node(f"T{child_id}")
            if child_id == "Child2":
                # Only Child2 should succeed
                terminal.measurement_fn = lambda env: 1.0  # Above threshold
            else:
                # Child1 and Child3 should fail
                terminal.measurement_fn = lambda env: 0.5  # Below threshold
        
        graph.request_root("Parent")
        # Initial propagation to get things started
        for _ in range(4):
            graph.propagate_step()
        
        # Terminals will now auto-measure with their custom functions
        # TChild2 will succeed (measurement = 1.0 > 0.8)
        # TChild1 and TChild3 will fail (measurement = 0.5 < 0.8)
        
        # Run until Child2 confirms
        for step in range(8):
            graph.propagate_step()
            if graph.get_node("Child2").state == ReCoNState.CONFIRMED:
                break
        
        # Run one more step for parent to receive confirmation
        graph.propagate_step()
        
        # Parent should now be confirmed (even though Child1, Child3 haven't)
        assert graph.get_node("Child2").state == ReCoNState.CONFIRMED
        assert graph.get_node("Parent").state == ReCoNState.TRUE  # Or CONFIRMED
        
        # Other children should still be running or can fail
        assert graph.get_node("Child1").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.FAILED]
        assert graph.get_node("Child3").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.FAILED]
    
    def test_all_children_fail_parent_fails(self):
        """Parent should fail only when ALL children fail (AND semantics for failure)."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Child1", "script")
        graph.add_node("Child2", "script")
        
        graph.add_link("Parent", "Child1", "sub")
        graph.add_link("Parent", "Child2", "sub")
        
        # No terminals - children will fail
        
        graph.request_root("Parent")
        
        # Run until children fail
        for step in range(10):
            graph.propagate_step()
            if (graph.get_node("Child1").state == ReCoNState.FAILED and
                graph.get_node("Child2").state == ReCoNState.FAILED):
                break
        
        # Both children should have failed
        assert graph.get_node("Child1").state == ReCoNState.FAILED
        assert graph.get_node("Child2").state == ReCoNState.FAILED
        
        # Parent should now fail (no more wait signals) - may take a few steps
        for _ in range(3):
            graph.propagate_step()
            if graph.get_node("Parent").state == ReCoNState.FAILED:
                break
        assert graph.get_node("Parent").state == ReCoNState.FAILED
    
    def test_nested_hierarchies(self):
        """Should handle hierarchies within hierarchies."""
        graph = ReCoNGraph()
        
        # Create: Root -> [Branch1, Branch2]
        # Branch1 -> [Leaf1, Leaf2] 
        # Branch2 -> [Leaf3, Leaf4]
        
        graph.add_node("Root", "script")
        graph.add_node("Branch1", "script")
        graph.add_node("Branch2", "script")
        
        graph.add_link("Root", "Branch1", "sub")
        graph.add_link("Root", "Branch2", "sub")
        
        # Branch1 children
        graph.add_node("Leaf1", "script")
        graph.add_node("Leaf2", "script")
        graph.add_link("Branch1", "Leaf1", "sub")
        graph.add_link("Branch1", "Leaf2", "sub")
        
        # Branch2 children  
        graph.add_node("Leaf3", "script")
        graph.add_node("Leaf4", "script")
        graph.add_link("Branch2", "Leaf3", "sub")
        graph.add_link("Branch2", "Leaf4", "sub")
        
        # Add terminals to leaves (set to not auto-confirm)
        for leaf in ["Leaf1", "Leaf2", "Leaf3", "Leaf4"]:
            terminal = graph.add_node(f"T{leaf}", "terminal")
            # Set measurement function to return low value (won't auto-confirm)
            terminal.measurement_fn = lambda env: 0.0
            graph.add_link(leaf, f"T{leaf}", "sub")
        
        graph.request_root("Root")
        
        # Should propagate down the hierarchy over multiple steps
        # Step 1: Root becomes ACTIVE
        graph.propagate_step()
        assert graph.get_node("Root").state == ReCoNState.ACTIVE
        
        # Step 2: Root becomes WAITING and sends requests to branches
        graph.propagate_step()
        assert graph.get_node("Root").state == ReCoNState.WAITING
        assert graph.get_node("Branch1").state == ReCoNState.REQUESTED
        assert graph.get_node("Branch2").state == ReCoNState.REQUESTED
        
        # Step 3: Branch1, Branch2 become ACTIVE  
        graph.propagate_step()
        assert graph.get_node("Branch1").state == ReCoNState.ACTIVE
        assert graph.get_node("Branch2").state == ReCoNState.ACTIVE
        
        # Step 4: Branch1, Branch2 become WAITING
        graph.propagate_step()
        assert graph.get_node("Branch1").state == ReCoNState.WAITING
        assert graph.get_node("Branch2").state == ReCoNState.WAITING
        
        # Skip detailed leaf progression since they fail without terminals
        # Just verify the hierarchy structure works
        
        # Simulate Leaf1 succeeding
        graph.get_node("TLeaf1").state = ReCoNState.CONFIRMED
        
        for step in range(10):
            graph.propagate_step()
            if graph.get_node("Root").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]:
                break
        
        # Branch1 should confirm, causing Root to confirm
        assert graph.get_node("Leaf1").state == ReCoNState.CONFIRMED
        assert graph.get_node("Branch1").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
        assert graph.get_node("Root").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
    
    def test_mixed_sequences_and_hierarchies(self):
        """Should handle combinations of por/ret sequences and sub/sur hierarchies."""
        graph = ReCoNGraph()
        
        # Create: Root -> Alt1 -> Seq1 -> Seq2
        #               -> Alt2 -> Seq3
        
        graph.add_node("Root", "script")
        graph.add_node("Alt1", "script")
        graph.add_node("Alt2", "script") 
        
        # Root has alternatives
        graph.add_link("Root", "Alt1", "sub")
        graph.add_link("Root", "Alt2", "sub")
        
        # Alt1 has sequence
        graph.add_node("Seq1", "script")
        graph.add_node("Seq2", "script")
        graph.add_link("Alt1", "Seq1", "sub")
        graph.add_link("Seq1", "Seq2", "por")
        
        # Alt2 has single sequence
        graph.add_node("Seq3", "script")
        graph.add_link("Alt2", "Seq3", "sub")
        
        # Add terminals
        for seq in ["Seq2", "Seq3"]:  # Only terminals at sequence ends
            graph.add_node(f"T{seq}", "terminal")
            graph.add_link(seq, f"T{seq}", "sub")
        
        graph.request_root("Root")
        
        # Run the mixed hierarchy/sequence structure
        for step in range(10):
            graph.propagate_step()
            
            # Auto-confirm terminals when nodes are waiting
            for seq in ["Seq2", "Seq3"]:
                if graph.get_node(seq).state == ReCoNState.WAITING:
                    graph.get_node(f"T{seq}").state = ReCoNState.CONFIRMED
        
        # Verify the mixed structure works:
        # 1. Root should have activated both alternatives
        assert graph.get_node("Root").state in [ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED]
        
        # 2. Both alternatives should be processing or completed
        assert graph.get_node("Alt1").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED]
        assert graph.get_node("Alt2").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED]
        
        # 3. Sequence nodes should progress according to por/ret constraints
        # All should eventually complete successfully
        assert graph.get_node("Seq1").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED]
        assert graph.get_node("Seq3").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED]
        
        # Simulate Seq3 (Alt2) succeeding first
        graph.get_node("TSeq3").state = ReCoNState.CONFIRMED
        
        for step in range(5):
            graph.propagate_step()
            if graph.get_node("Root").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]:
                break
        
        # Alt2 -> Seq3 path should have won
        assert graph.get_node("Seq3").state == ReCoNState.CONFIRMED
        assert graph.get_node("Alt2").state == ReCoNState.CONFIRMED
        assert graph.get_node("Root").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
    
    def test_hierarchy_failure_modes(self):
        """Test various failure scenarios in hierarchies."""
        graph = ReCoNGraph()
        
        # Test 1: Timeout failure
        graph.add_node("TimeoutParent", "script")
        graph.add_node("SlowChild", "script")
        graph.add_link("TimeoutParent", "SlowChild", "sub")
        # No terminal for SlowChild - will eventually fail
        
        graph.request_root("TimeoutParent")
        
        # Run until failure
        for step in range(10):
            graph.propagate_step()
            if graph.get_node("TimeoutParent").state == ReCoNState.FAILED:
                break
        
        assert graph.get_node("SlowChild").state == ReCoNState.FAILED
        assert graph.get_node("TimeoutParent").state == ReCoNState.FAILED
        
        # Test 2: Partial success shouldn't cause parent failure
        graph = ReCoNGraph()
        graph.add_node("Parent", "script")
        graph.add_node("GoodChild", "script") 
        graph.add_node("BadChild", "script")
        
        graph.add_link("Parent", "GoodChild", "sub")
        graph.add_link("Parent", "BadChild", "sub")
        
        # Only GoodChild has terminal
        graph.add_node("TGood", "terminal")
        graph.add_link("GoodChild", "TGood", "sub")
        
        graph.request_root("Parent")
        
        # GoodChild should succeed despite BadChild failing
        for step in range(10):
            graph.propagate_step()
            if graph.get_node("GoodChild").state == ReCoNState.WAITING:
                graph.get_node("TGood").state = ReCoNState.CONFIRMED
            if graph.get_node("Parent").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]:
                break
        
        assert graph.get_node("GoodChild").state == ReCoNState.CONFIRMED
        assert graph.get_node("Parent").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]
        # BadChild may fail, but parent still succeeds
    
    def test_dynamic_hierarchy_modification(self):
        """Should handle runtime changes to hierarchy structure."""
        graph = ReCoNGraph()
        
        graph.add_node("Parent", "script")
        graph.add_node("Child1", "script") 
        graph.add_link("Parent", "Child1", "sub")
        
        graph.add_node("TChild1", "terminal")
        graph.add_link("Child1", "TChild1", "sub")
        
        graph.request_root("Parent")
        graph.propagate_step()
        
        # Add second child during execution
        graph.add_node("Child2", "script")
        graph.add_node("TChild2", "terminal")
        graph.add_link("Parent", "Child2", "sub")
        graph.add_link("Child2", "TChild2", "sub")
        
        # For dynamic modification to work, need to re-trigger parent
        # This is because parent has already sent its initial requests
        graph.stop_request("Parent")
        graph.request_root("Parent")
        
        # Now both children should be requested
        for _ in range(3):
            graph.propagate_step()
        
        assert graph.get_node("Child1").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE]
        assert graph.get_node("Child2").state in [ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.REQUESTED]
        
        # Either child confirming should confirm parent
        graph.get_node("TChild2").state = ReCoNState.CONFIRMED
        
        for step in range(5):
            graph.propagate_step()
            if graph.get_node("Parent").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]:
                break
        
        assert graph.get_node("Parent").state in [ReCoNState.TRUE, ReCoNState.CONFIRMED]