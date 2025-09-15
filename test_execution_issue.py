#!/usr/bin/env python3
"""Test script to reproduce the issue where the root node fails on the third execution."""

import sys
import requests
import json
import time

API_URL = "http://localhost:8000"

def test_execute_with_history():
    """Test executing the demo network multiple times."""
    
    print("Testing execute-with-history endpoint...")
    
    for i in range(5):
        print(f"\n=== Execution {i+1} ===")
        
        # Execute with history
        response = requests.post(
            f"{API_URL}/networks/demo/execute-with-history",
            json={
                "root_node": "Root",
                "max_steps": 100
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Result: {result['result']}")
            print(f"Final state: {result['final_state']}")
            print(f"Total steps: {result['total_steps']}")
            
            # Check final states of all nodes
            if result['steps']:
                final_step = result['steps'][-1]
                print(f"\nFinal node states:")
                for node_id, state in final_step['states'].items():
                    print(f"  {node_id}: {state}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
        
        time.sleep(0.5)  # Small delay between executions

if __name__ == "__main__":
    test_execute_with_history()
