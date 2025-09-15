#!/usr/bin/env python3
"""Debug script to trace the execution flow."""

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

demo_graph.add_link("Root", "A", "sub")    # Root requests A
demo_graph.add_link("Root", "B", "sub")    # Root requests B  
demo_graph.add_link("A", "B", "por")       # A inhibits B until A completes
demo_graph.add_link("A", "TA", "sub")      # A validates via TA
demo_graph.add_link("B", "TB", "sub")      # B validates via TB

print("Initial state:")
for node_id, node in demo_graph.nodes.items():
    print(f"  {node_id}: {node.state.value}")

print("\nExecuting script...")
demo_graph.request_root("Root")

for step in range(10):
    print(f"\n=== Step {step + 1} ===")
    
    # Show messages
    print("Messages in queue:")
    for msg in demo_graph.message_queue:
        print(f"  {msg.type.value}: {msg.source} -> {msg.target}")
    
    demo_graph.propagate_step()
    
    # Show states after step
    print("States after step:")
    for node_id, node in demo_graph.nodes.items():
        print(f"  {node_id}: {node.state.value}")
    
    if demo_graph.is_completed():
        print(f"\nCompleted! Root result: {demo_graph.get_results()['Root']}")
        break

print("\n\n=== Checking node B's por inhibition ===")
# Check if B is receiving proper messages
b_node = demo_graph.nodes["B"]
print(f"B state: {b_node.state.value}")
print(f"B has_por_predecessors: {b_node._has_por_predecessors}")

# Check the por link from A
por_links = demo_graph.get_links(target="B", link_type="por")
print(f"POR links to B: {len(por_links)}")
for link in por_links:
    print(f"  {link.source} -> {link.target}")
