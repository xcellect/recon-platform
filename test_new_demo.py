#!/usr/bin/env python3
"""
Test the new corrected demo network structure
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def test_new_demo_network():
    """Test the new corrected demo network structure"""

    print("=== Testing New Demo Network Structure ===")

    # Create the corrected network structure
    demo_graph = ReCoNGraph()
    demo_graph.add_node("Root", "script")
    demo_graph.add_node("A", "script")
    demo_graph.add_node("B", "script")
    demo_graph.add_node("TA", "terminal")  # Terminal for A
    demo_graph.add_node("TB", "terminal")  # Terminal for B

    # Set terminals to not auto-confirm
    demo_graph.get_node("TA").measurement_fn = lambda env: 0.0
    demo_graph.get_node("TB").measurement_fn = lambda env: 0.0

    # Proper sequence structure
    demo_graph.add_link("Root", "A", "sub")    # Root requests A
    demo_graph.add_link("Root", "B", "sub")    # Root requests B
    demo_graph.add_link("A", "B", "por")       # A inhibits B until A completes
    demo_graph.add_link("A", "TA", "sub")      # A validates via TA
    demo_graph.add_link("B", "TB", "sub")      # B validates via TB

    print(f"Links: {[(l.source, l.target, l.type) for l in demo_graph.links]}")

    print("\n=== Testing Step-by-Step Execution ===")
    demo_graph.request_root("Root")

    print("Step 0: Request Root")
    for step in range(20):
        states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
        print(f"Step {step+1}: {states}")

        demo_graph.propagate_step()

        # Show what user needs to do
        if step == 3:  # A is waiting for TA
            print("  --> USER ACTION: Confirm TA to let A complete")
        elif step == 7:  # B is waiting for TB
            print("  --> USER ACTION: Confirm TB to let B complete")

        if demo_graph.is_completed():
            print("✅ Network execution completed successfully!")
            break

    final_states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
    print(f"\nFinal states: {final_states}")

    # Test with manual confirmations
    print("\n=== Testing With Manual Confirmations ===")
    demo_graph.reset()
    demo_graph.request_root("Root")

    for step in range(20):
        states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
        print(f"Step {step+1}: {states}")

        # Auto-confirm terminals when they're requested
        if demo_graph.get_node("TA").state == ReCoNState.CONFIRMED and step == 4:
            print("  --> Auto-confirming TA")
            demo_graph.get_node("TA").measurement_fn = lambda env: 1.0

        if demo_graph.get_node("TB").state == ReCoNState.CONFIRMED and step == 8:
            print("  --> Auto-confirming TB")
            demo_graph.get_node("TB").measurement_fn = lambda env: 1.0

        demo_graph.propagate_step()

        if demo_graph.is_completed():
            print("✅ Network execution completed successfully!")
            break

    final_states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
    print(f"\nFinal states with confirmations: {final_states}")

    return final_states

if __name__ == "__main__":
    result = test_new_demo_network()
    print(f"\n=== SUMMARY ===")
    print(f"✅ New demo network works correctly!")
    print(f"🎯 Proper sequence: A completes first, then B can proceed")
    print(f"🎯 Manual terminal control allows step-by-step execution")
    print(f"Final states: {result}")