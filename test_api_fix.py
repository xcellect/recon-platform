#!/usr/bin/env python3
"""
Test the new API endpoint with terminal configurations.
"""

import requests
import json

# Test network data with high activation terminals
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

# Configure terminals to confirm based on activation
terminal_configs = [
    {"node_id": "TA", "measurement_type": "confirm", "measurement_value": 1.0},
    {"node_id": "TB", "measurement_type": "confirm", "measurement_value": 1.0}
]

payload = {
    "network_data": network_data,
    "root_node": "Root",
    "max_steps": 100,
    "terminal_configs": terminal_configs
}

print("Testing new API endpoint...")
response = requests.post("http://localhost:8001/execute-network", json=payload)

if response.status_code == 200:
    result = response.json()
    print(f"Success! Result: {result['result']}")
    print("Final states:")
    final_step = result['steps'][-1] if result['steps'] else {}
    for node_id, state in final_step.get('states', {}).items():
        print(f"  {node_id}: {state}")
else:
    print(f"Error {response.status_code}: {response.text}")
