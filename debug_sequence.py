#!/usr/bin/env python3
"""
Debug sequence execution to understand message flow
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def debug_sequence():
    """Debug the sequence structure and message flow"""
    
    graph = ReCoNGraph()
    
    graph.add_node("Parent", "script")
    graph.add_node("A", "script")  
    graph.add_node("B", "script")
    graph.add_node("TA", "terminal")
    graph.add_node("TB", "terminal")
    
    graph.add_link("Parent", "A", "sub")
    graph.add_link("Parent", "B", "sub")  
    graph.add_link("A", "B", "por")  # A por B, creates B ret A automatically
    graph.add_link("A", "TA", "sub")
    graph.add_link("B", "TB", "sub")
    
    print("Links created:")
    for link in graph.links:
        print(f"  {link.source} --{link.type}--> {link.target}")
    
    graph.request_root("Parent")
    
    # Run until B is in TRUE state
    for step in range(10):
        print(f"\\n=== Step {step} ===")
        
        # Show states
        for nid in ["Parent", "A", "B", "TA", "TB"]:
            node = graph.get_node(nid)
            print(f"{nid}: {node.state.value}")
        
        # Show B's incoming messages specifically
        b_node = graph.get_node("B")
        print(f"\\nB's incoming messages:")
        for link_type, messages in b_node.incoming_messages.items():
            if messages:
                for msg in messages:
                    print(f"  {link_type}: {msg.type.value} from {msg.source}")
        
        # Calculate what B should send
        if b_node.state == ReCoNState.TRUE:
            print(f"\\nB is TRUE - checking ret inhibition:")
            ret_activation = b_node.get_link_activation("ret")
            print(f"  B's ret activation: {ret_activation}")
            
            inputs = {}
            messages = b_node.get_outgoing_messages(inputs)
            print(f"  B's outgoing messages: {messages}")
            
            # The key question: is B receiving ret inhibition?
            print(f"  Is B ret inhibited? {ret_activation < 0}")
        
        graph.propagate_step()
        
        if graph.is_completed() or step > 8:
            break
    
    print(f"\\n=== Final Analysis ===")
    print(f"B final state: {graph.get_node('B').state}")
    print(f"Parent final state: {graph.get_node('Parent').state}")

if __name__ == "__main__":
    debug_sequence()