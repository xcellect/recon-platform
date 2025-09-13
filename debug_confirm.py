#!/usr/bin/env python3
"""
Debug confirmation logic
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def test_confirmed_messages():
    """Test what CONFIRMED nodes send"""
    
    node = ReCoNNode("B", "script")
    node.state = ReCoNState.CONFIRMED
    
    # Test without ret inhibition
    inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 1.0}
    messages = node.get_outgoing_messages(inputs)
    
    print(f"CONFIRMED node (no ret inhibition): {messages}")
    
    # Test with ret inhibition  
    inputs_inhibited = {"sub": 1.0, "por": 0.0, "ret": -1.0, "sur": 1.0}
    messages_inhibited = node.get_outgoing_messages(inputs_inhibited)
    
    print(f"CONFIRMED node (ret inhibited): {messages_inhibited}")

def debug_full_sequence():
    """Debug full sequence to see Parent transition"""
    
    graph = ReCoNGraph()
    
    graph.add_node("Parent", "script")
    graph.add_node("A", "script")  
    graph.add_node("B", "script")
    graph.add_node("TA", "terminal")
    graph.add_node("TB", "terminal")
    
    graph.add_link("Parent", "A", "sub")
    graph.add_link("Parent", "B", "sub")  
    graph.add_link("A", "B", "por")
    graph.add_link("A", "TA", "sub")
    graph.add_link("B", "TB", "sub")
    
    graph.request_root("Parent")
    
    # Run until completion
    for step in range(12):
        states = {nid: graph.get_node(nid).state.value for nid in ["Parent", "A", "B"]}
        print(f"Step {step}: {states}")
        
        # Check B's messages when it's confirmed
        b_node = graph.get_node("B")
        if b_node.state == ReCoNState.CONFIRMED:
            messages = b_node.get_outgoing_messages({})
            print(f"  B (CONFIRMED) sends: {messages}")
            
        # Check Parent's sur activation
        parent = graph.get_node("Parent")
        parent_sur = parent.get_link_activation("sur")
        print(f"  Parent sur activation: {parent_sur}")
        
        graph.propagate_step()
        
        if graph.is_completed():
            print(f"  Execution completed!")
            break

if __name__ == "__main__":
    print("=== Testing CONFIRMED Messages ===")
    test_confirmed_messages()
    
    print(f"\\n=== Testing Full Sequence ===")
    debug_full_sequence()