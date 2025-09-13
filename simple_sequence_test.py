#!/usr/bin/env python3
"""
Simple sequence test to debug the structure
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def test_simple_sequence():
    """Test simple sequence structure"""
    
    print("=== Testing Simple Sequence ===")
    
    graph = ReCoNGraph()
    
    # According to the paper, a sequence should be:
    # Parent -> [A -> B] where A and B are in sequence
    # Each element in sequence (A, B) needs to validate something
    
    # Create: Parent -> A -> TA (terminal for A)
    #                -> B -> TB (terminal for B)  
    # Where A por B (A must complete before B can start)
    
    graph.add_node("Parent", "script")
    graph.add_node("A", "script")  
    graph.add_node("B", "script")
    graph.add_node("TA", "terminal")  # A's terminal
    graph.add_node("TB", "terminal")  # B's terminal
    
    # Parent has two children in sequence
    graph.add_link("Parent", "A", "sub")  # Parent requests A
    graph.add_link("Parent", "B", "sub")  # Parent requests B 
    graph.add_link("A", "B", "por")       # A inhibits B until A completes
    
    # Each script node needs something to validate
    graph.add_link("A", "TA", "sub")      # A requests TA
    graph.add_link("B", "TB", "sub")      # B requests TB
    
    print(f"Links: {[(l.source, l.target, l.type) for l in graph.links]}")
    
    graph.request_root("Parent")
    
    print("\\nExecution steps:")
    for step in range(10):
        states = {nid: graph.get_node(nid).state.value for nid in ["Parent", "A", "B", "TA", "TB"]}
        print(f"Step {step}: {states}")
        
        graph.propagate_step()
        
        if graph.is_completed():
            print("Execution completed!")
            break
    
    final_states = {nid: graph.get_node(nid).state.value for nid in ["Parent", "A", "B", "TA", "TB"]}
    print(f"\\nFinal: {final_states}")
    
    # Expected: A should complete first, then B
    # Only the last in sequence (B) should be able to confirm parent
    print(f"\\nA state: {graph.get_node('A').state}")
    print(f"B state: {graph.get_node('B').state}") 
    print(f"Parent state: {graph.get_node('Parent').state}")

if __name__ == "__main__":
    test_simple_sequence()