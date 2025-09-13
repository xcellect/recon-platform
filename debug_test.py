#!/usr/bin/env python3
"""
Debug test to understand current ReCoN behavior
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def test_simple_case():
    """Test a very simple case to debug"""
    
    print("=== Testing Simple ReCoN Node ===")
    
    # Test individual node
    node = ReCoNNode("test")
    print(f"Initial state: {node.state}")
    
    # Test requesting
    inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
    signals = node.update_state(inputs)
    print(f"After request - State: {node.state}, Signals: {signals}")
    
    print("\n=== Testing Simple Graph ===")
    
    # Test simple graph
    graph = ReCoNGraph()
    
    # Create A -> T
    graph.add_node("A", "script")
    graph.add_node("T", "terminal")
    graph.add_link("A", "T", "sub")
    
    print(f"Graph created: {graph}")
    print(f"Links: {[(l.source, l.target, l.type) for l in graph.links]}")
    
    # Request A
    graph.request_root("A")
    print(f"Requested A. Root requests: {graph.requested_roots}")
    
    # Check initial states
    print(f"Initial - A: {graph.get_node('A').state}, T: {graph.get_node('T').state}")
    
    # Step 1
    graph.propagate_step()
    print(f"Step 1 - A: {graph.get_node('A').state}, T: {graph.get_node('T').state}")
    
    # Step 2
    graph.propagate_step()
    print(f"Step 2 - A: {graph.get_node('A').state}, T: {graph.get_node('T').state}")
    
    # Step 3
    graph.propagate_step()
    print(f"Step 3 - A: {graph.get_node('A').state}, T: {graph.get_node('T').state}")

if __name__ == "__main__":
    test_simple_case()