#!/usr/bin/env python3

from recon_engine import ReCoNGraph, ReCoNState

def debug_sequence():
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

    # Request via parent - parent requests ALL nodes in sequence
    graph.add_node("Root", "script")
    graph.add_link("Root", "A", "sub")
    graph.add_link("Root", "B", "sub")  # B is also requested by Root
    graph.add_link("Root", "C", "sub")  # C is also requested by Root

    graph.request_root("Root")
    
    print("=== Initial state after requesting Root ===")
    for step in range(10):
        print(f"\nStep {step + 1}:")
        graph.propagate_step()
        
        for node_id in ["Root", "A", "B", "C", "TA", "TB", "TC"]:
            node = graph.get_node(node_id)
            print(f"  {node_id}: {node.state}")
            
        if step == 4:  # After initial stabilization
            print("\n=== Simulating TA confirming ===")
            graph.get_node("TA").state = ReCoNState.CONFIRMED

if __name__ == "__main__":
    debug_sequence()