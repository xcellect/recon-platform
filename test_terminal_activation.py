#!/usr/bin/env python3
"""
Test script to debug why terminals with activation=1 aren't confirming.
"""

import sys
import os
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph

# Test network data from user
network_data = {
    "nodes": [
        {"id": "Root", "type": "script", "state": "inactive", "activation": 0, "gates": {"sub": 0, "sur": 0, "por": 0, "ret": 0, "gen": 0}, "timing_config": {"timing_mode": "discrete", "discrete_wait_steps": 3, "sequence_wait_steps": 6, "activation_decay_rate": 0.8, "activation_failure_threshold": 0.1, "activation_initial_level": 0.8, "current_waiting_activation": 0}},
        {"id": "A", "type": "script", "state": "inactive", "activation": 0, "gates": {"sub": 0, "sur": 0, "por": 0, "ret": 0, "gen": 0}, "timing_config": {"timing_mode": "discrete", "discrete_wait_steps": 3, "sequence_wait_steps": 6, "activation_decay_rate": 0.8, "activation_failure_threshold": 0.1, "activation_initial_level": 0.8, "current_waiting_activation": 0}},
        {"id": "B", "type": "script", "state": "inactive", "activation": 0, "gates": {"sub": 0, "sur": 0, "por": 0, "ret": 0, "gen": 0}, "timing_config": {"timing_mode": "discrete", "discrete_wait_steps": 3, "sequence_wait_steps": 6, "activation_decay_rate": 0.8, "activation_failure_threshold": 0.1, "activation_initial_level": 0.8, "current_waiting_activation": 0}},
        {"id": "TA", "type": "terminal", "state": "inactive", "activation": 1, "gates": {"sub": 0, "sur": 0, "por": 0, "ret": 0, "gen": 0}, "timing_config": {"timing_mode": "discrete", "discrete_wait_steps": 3, "sequence_wait_steps": 6, "activation_decay_rate": 0.8, "activation_failure_threshold": 0.1, "activation_initial_level": 0.8, "current_waiting_activation": 0}},
        {"id": "TB", "type": "terminal", "state": "inactive", "activation": 1, "gates": {"sub": 0, "sur": 0, "por": 0, "ret": 0, "gen": 0}, "timing_config": {"timing_mode": "discrete", "discrete_wait_steps": 3, "sequence_wait_steps": 6, "activation_decay_rate": 0.8, "activation_failure_threshold": 0.1, "activation_initial_level": 0.8, "current_waiting_activation": 0}}
    ],
    "links": [
        {"source": "Root", "target": "A", "type": "sub", "weight": 1},
        {"source": "A", "target": "Root", "type": "sur", "weight": 1},
        {"source": "Root", "target": "B", "type": "sub", "weight": 1},
        {"source": "B", "target": "Root", "type": "sur", "weight": 1},
        {"source": "A", "target": "B", "type": "por", "weight": 1},
        {"source": "B", "target": "A", "type": "ret", "weight": 1},
        {"source": "A", "target": "TA", "type": "sub", "weight": 1},
        {"source": "TA", "target": "A", "type": "sur", "weight": 1},
        {"source": "B", "target": "TB", "type": "sub", "weight": 1},
        {"source": "TB", "target": "B", "type": "sur", "weight": 1}
    ],
    "requested_roots": [],
    "step_count": 0
}

def test_terminal_behavior():
    print("=== Testing Terminal Behavior ===")
    
    # Create graph from data
    graph = ReCoNGraph.from_dict(network_data)
    
    print("Created graph with nodes:")
    for node_id, node in graph.nodes.items():
        print(f"  {node_id}: {node.type}, activation={node.activation}, state={node.state.value}")
    
    print("\nLinks:")
    for link in graph.links:
        print(f"  {link.source} -> {link.target} ({link.type}, weight={link.weight})")
    
    # Test terminal measurement behavior
    print("\n=== Testing Terminal Measurement ===")
    ta_node = graph.get_node("TA")
    tb_node = graph.get_node("TB")
    
    print(f"TA measurement: {ta_node.measure()}")
    print(f"TB measurement: {tb_node.measure()}")
    print(f"TA transition_threshold: {ta_node.transition_threshold}")
    print(f"TB transition_threshold: {tb_node.transition_threshold}")
    
    # Execute and trace
    print("\n=== Executing Network ===")
    graph.reset()
    result = graph.execute_script_with_history("Root", max_steps=20)
    
    print(f"Final result: {result['result']}")
    print(f"Total steps: {result['total_steps']}")
    
    print("\nFinal states:")
    for node_id, node in graph.nodes.items():
        print(f"  {node_id}: {node.state.value}")
    
    # Show execution trace
    print("\nExecution trace:")
    for i, step in enumerate(result['steps'][:10]):  # Show first 10 steps
        print(f"Step {step['step']}:")
        for node_id, state in step['states'].items():
            if state != 'inactive':
                print(f"  {node_id}: {state}")
        if step['messages']:
            print(f"  Messages: {len(step['messages'])}")

if __name__ == "__main__":
    test_terminal_behavior()
