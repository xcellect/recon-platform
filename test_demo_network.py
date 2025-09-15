#!/usr/bin/env python3
"""
Test the demo network structure to ensure it can complete execution
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def test_demo_network():
    """Test the exact demo network structure from the API"""

    print("=== Testing Demo Network Structure ===")

    # Create the same network as in api/app.py
    demo_graph = ReCoNGraph()
    demo_graph.add_node("Root", "script")
    demo_graph.add_node("A", "script")
    demo_graph.add_node("B", "script")
    demo_graph.add_node("T", "terminal")

    # Set terminal to not auto-confirm (same as API)
    terminal_node = demo_graph.get_node("T")
    terminal_node.measurement_fn = lambda env: 0.0  # Below threshold, won't confirm

    demo_graph.add_link("Root", "A", "sub")
    demo_graph.add_link("A", "B", "por")
    demo_graph.add_link("B", "T", "sub")

    print(f"Links: {[(l.source, l.target, l.type) for l in demo_graph.links]}")

    # Analyze the structure
    print("\nNetwork Analysis:")
    print(f"Root has children: {demo_graph.get_node('Root')._has_children}")
    print(f"A has por successors: {demo_graph.get_node('A')._has_por_successors}")
    print(f"B has children: {demo_graph.get_node('B')._has_children}")

    # Check what happens without requesting
    print("\n=== Testing Execution ===")
    demo_graph.request_root("Root")

    print("Execution steps:")
    for step in range(15):
        states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "T"]}
        print(f"Step {step}: {states}")

        # Check if B can proceed
        b_node = demo_graph.get_node("B")
        print(f"  B incoming messages: {[(msg_type, len(msgs)) for msg_type, msgs in b_node.incoming_messages.items() if msgs]}")

        demo_graph.propagate_step()

        # After step 10, manually confirm the terminal to see if B can complete
        if step == 10:
            print("\n  --> Manually confirming terminal T")
            terminal_node.measurement_fn = lambda env: 1.0  # Above threshold, will confirm

        if demo_graph.is_completed():
            print("Execution completed!")
            break

    final_states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "T"]}
    print(f"\nFinal: {final_states}")

    # Check for any issues
    print(f"\nIssue Analysis:")
    print(f"Is B stuck in suppressed? {demo_graph.get_node('B').state == ReCoNState.SUPPRESSED}")
    print(f"Does A have por link to B? {demo_graph.has_link('A', 'B', 'por')}")
    print(f"A state: {demo_graph.get_node('A').state}")

    # The issue might be that A never completes, so B stays suppressed
    # A needs something to validate to go from ACTIVE -> TRUE

    return final_states

def test_corrected_network():
    """Test a corrected version with proper structure"""

    print("\n\n=== Testing Corrected Network ===")

    graph = ReCoNGraph()
    graph.add_node("Root", "script")
    graph.add_node("A", "script")
    graph.add_node("B", "script")
    graph.add_node("TA", "terminal")  # Terminal for A
    graph.add_node("TB", "terminal")  # Terminal for B

    # A and B each need their own terminals to validate
    graph.add_link("Root", "A", "sub")
    graph.add_link("Root", "B", "sub")  # Root requests both A and B
    graph.add_link("A", "B", "por")     # A inhibits B until A completes
    graph.add_link("A", "TA", "sub")    # A validates via TA
    graph.add_link("B", "TB", "sub")    # B validates via TB

    print(f"Links: {[(l.source, l.target, l.type) for l in graph.links]}")

    graph.request_root("Root")

    print("\nExecution steps:")
    for step in range(20):
        states = {nid: graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
        print(f"Step {step}: {states}")

        graph.propagate_step()

        # Confirm TA after a few steps to let A complete
        if step == 5:
            print("  --> Confirming TA (A's terminal)")
            graph.get_node("TA").measurement_fn = lambda env: 1.0

        # Confirm TB after A completes to let B complete
        if step == 10:
            print("  --> Confirming TB (B's terminal)")
            graph.get_node("TB").measurement_fn = lambda env: 1.0

        if graph.is_completed():
            print("Execution completed!")
            break

    final_states = {nid: graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
    print(f"\nFinal: {final_states}")

    return final_states

if __name__ == "__main__":
    demo_result = test_demo_network()
    corrected_result = test_corrected_network()

    print(f"\n=== SUMMARY ===")
    print(f"Demo network final states: {demo_result}")
    print(f"Corrected network final states: {corrected_result}")