#!/usr/bin/env python3

from recon_engine import ReCoNGraph, ReCoNState

def debug_propagation():
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
    
    print("Initial states:")
    for node_id in ["Parent", "A", "B", "TA"]:
        node = graph.get_node(node_id)
        print(f"  {node_id}: {node.state}")
    
    print("\nRequesting root Parent...")
    graph.request_root("Parent")
    
    print(f"Requested roots: {graph.requested_roots}")
    
    print("\nAfter request_root, before propagate_step:")
    for node_id in ["Parent", "A", "B", "TA"]:
        node = graph.get_node(node_id)
        print(f"  {node_id}: {node.state}")
        print(f"    Incoming messages: {[(lt, len(msgs)) for lt, msgs in node.incoming_messages.items() if msgs]}")
    
    print("\nPropagating step 1...")
    graph.propagate_step()
    
    print("After step 1:")
    for node_id in ["Parent", "A", "B", "TA"]:
        node = graph.get_node(node_id)
        print(f"  {node_id}: {node.state}")
        print(f"    Incoming messages: {[(lt, len(msgs)) for lt, msgs in node.incoming_messages.items() if msgs]}")
    
    print("\nChecking what messages Parent should send:")
    parent = graph.get_node("Parent")
    parent_messages = parent.get_outgoing_messages({})
    print(f"Parent outgoing messages: {parent_messages}")
    
    print(f"\nLinks from Parent:")
    parent_links = graph.get_links(source="Parent")
    for link in parent_links:
        print(f"  {link.source} -> {link.target} via {link.type}")
    
    print("\nPropagating step 2...")
    graph.propagate_step()
    
    print("After step 2:")
    for node_id in ["Parent", "A", "B", "TA"]:
        node = graph.get_node(node_id)
        print(f"  {node_id}: {node.state}")
        print(f"    Incoming messages: {[(lt, len(msgs)) for lt, msgs in node.incoming_messages.items() if msgs]}")
        
    print(f"\nMessage queue size: {len(graph.message_queue)}")
    for i, msg in enumerate(graph.message_queue):
        print(f"  Message {i}: {msg.source} -> {msg.target} via {msg.link_type} ({msg.type})")
        
    print("\nPropagating step 3...")
    graph.propagate_step()
    
    print("After step 3:")
    for node_id in ["Parent", "A", "B", "TA"]:
        node = graph.get_node(node_id)
        print(f"  {node_id}: {node.state}")
        print(f"    Incoming messages: {[(lt, len(msgs)) for lt, msgs in node.incoming_messages.items() if msgs]}")
        
    print("\nPropagating step 4...")
    graph.propagate_step()
    
    print("After step 4:")
    for node_id in ["Parent", "A", "B", "TA"]:
        node = graph.get_node(node_id)
        print(f"  {node_id}: {node.state}")
        print(f"    Incoming messages: {[(lt, len(msgs)) for lt, msgs in node.incoming_messages.items() if msgs]}")
        
        # Check por activation for B specifically
        if node_id == "B":
            por_activation = node.get_link_activation("por")
            print(f"    B por activation: {por_activation}")
            for msg in node.incoming_messages["por"]:
                print(f"      Message: {msg.type} with activation {msg.activation}")
        
    print("\nChecking A's outgoing messages:")
    a_node = graph.get_node("A")
    a_messages = a_node.get_outgoing_messages({})
    print(f"A outgoing messages: {a_messages}")
    
    print(f"\nLinks from A:")
    a_links = graph.get_links(source="A")
    for link in a_links:
        print(f"  {link.source} -> {link.target} via {link.type}")

if __name__ == "__main__":
    debug_propagation()