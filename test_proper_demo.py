#!/usr/bin/env python3
"""
Test the demo network with proper timing for terminal confirmations
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def test_demo_with_proper_timing():
    """Test demo network with proper terminal confirmation timing"""

    print("=== Testing Demo Network With Proper Manual Confirmations ===")

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

    demo_graph.request_root("Root")

    print("Execution with manual confirmations at proper timing:")
    for step in range(25):
        states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
        print(f"Step {step}: {states}")

        # Confirm TA when it's requested (but still failing)
        if step == 5 and demo_graph.get_node("TA").state == ReCoNState.CONFIRMED:
            print("  --> 🎯 Confirming TA (A's terminal) - A should now complete")
            demo_graph.get_node("TA").measurement_fn = lambda env: 1.0

        # Confirm TB when it's requested and A has completed
        if step == 10 and demo_graph.get_node("TB").state == ReCoNState.CONFIRMED:
            print("  --> 🎯 Confirming TB (B's terminal) - B should now complete")
            demo_graph.get_node("TB").measurement_fn = lambda env: 1.0

        demo_graph.propagate_step()

        if demo_graph.is_completed():
            print("✅ Network execution completed!")
            break

    final_states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
    print(f"\nFinal states: {final_states}")

    # Check the sequence worked correctly
    success = (
        final_states["TA"] == "confirmed" and
        final_states["TB"] == "confirmed" and
        final_states["Root"] in ["true", "confirmed"] and
        final_states["A"] in ["true", "confirmed"] and
        final_states["B"] in ["true", "confirmed"]
    )

    return success, final_states

def test_simpler_demo():
    """Test an even simpler demo network for the UI"""

    print("\n=== Testing Simpler Demo Network ===")

    # Create simpler network: Root -> A -> TA, A por B -> TB
    demo_graph = ReCoNGraph()
    demo_graph.add_node("Root", "script")
    demo_graph.add_node("A", "script")
    demo_graph.add_node("B", "script")
    demo_graph.add_node("TA", "terminal")
    demo_graph.add_node("TB", "terminal")

    # Start with terminals that will be manually confirmed
    demo_graph.get_node("TA").measurement_fn = lambda env: 0.0
    demo_graph.get_node("TB").measurement_fn = lambda env: 0.0

    # Links: Root only requests A initially, A requests B via por
    demo_graph.add_link("Root", "A", "sub")    # Root requests A
    demo_graph.add_link("A", "TA", "sub")      # A validates via TA
    demo_graph.add_link("A", "B", "por")       # A->B sequence
    demo_graph.add_link("B", "TB", "sub")      # B validates via TB

    print(f"Links: {[(l.source, l.target, l.type) for l in demo_graph.links]}")

    demo_graph.request_root("Root")

    print("\nSimpler execution:")
    for step in range(20):
        states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
        print(f"Step {step}: {states}")

        # Confirm terminals when they are in confirmed state (auto-confirmed by default)
        if step == 4 and demo_graph.get_node("TA").state == ReCoNState.CONFIRMED:
            print("  --> 🎯 Manually confirming TA")
            demo_graph.get_node("TA").measurement_fn = lambda env: 1.0

        if step == 8 and demo_graph.get_node("TB").state == ReCoNState.CONFIRMED:
            print("  --> 🎯 Manually confirming TB")
            demo_graph.get_node("TB").measurement_fn = lambda env: 1.0

        demo_graph.propagate_step()

        if demo_graph.is_completed():
            print("✅ Simpler network completed!")
            break

    final_states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
    print(f"\nSimpler final states: {final_states}")

    return final_states

if __name__ == "__main__":
    print("Testing demo network structures...\n")

    success1, result1 = test_demo_with_proper_timing()
    result2 = test_simpler_demo()

    print(f"\n=== RESULTS ===")
    print(f"Complex demo success: {success1}")
    print(f"Complex demo final: {result1}")
    print(f"Simple demo final: {result2}")

    if success1:
        print("✅ Demo network structure is correct and can complete!")
    else:
        print("❌ Demo network still has issues")