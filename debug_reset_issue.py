#!/usr/bin/env python3
"""Debug script to test reset behavior."""

import sys
sys.path.insert(0, '/workspace/recon-platform')

from recon_engine import ReCoNGraph

# Create the same demo network
demo_graph = ReCoNGraph()
demo_graph.add_node("Root", "script")
demo_graph.add_node("A", "script")
demo_graph.add_node("B", "script")
demo_graph.add_node("TA", "terminal")
demo_graph.add_node("TB", "terminal")

demo_graph.add_link("Root", "A", "sub")    
demo_graph.add_link("Root", "B", "sub")    
demo_graph.add_link("A", "B", "por")       
demo_graph.add_link("A", "TA", "sub")      
demo_graph.add_link("B", "TB", "sub")      

for run in range(3):
    print(f"\n\n{'='*50}")
    print(f"RUN {run + 1}")
    print('='*50)
    
    print("\nBefore reset:")
    for node_id, node in demo_graph.nodes.items():
        print(f"  {node_id}: {node.state.value}")
        if hasattr(node, '_request_absent_count'):
            print(f"    _request_absent_count: {node._request_absent_count}")
    
    # Reset the network
    demo_graph.reset()
    
    print("\nAfter reset:")
    for node_id, node in demo_graph.nodes.items():
        print(f"  {node_id}: {node.state.value}")
        if hasattr(node, '_request_absent_count'):
            print(f"    _request_absent_count: {node._request_absent_count}")
    
    # Execute
    result = demo_graph.execute_script_with_history("Root", max_steps=100)
    
    print(f"\nExecution result: {result['result']}")
    print(f"Total steps: {result['total_steps']}")
    
    # Check final states
    final_step = result['steps'][-1]
    print("\nFinal states:")
    for node_id, state in final_step['states'].items():
        print(f"  {node_id}: {state}")
