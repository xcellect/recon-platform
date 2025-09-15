#!/usr/bin/env python3
"""
Test a properly working demo network
"""

import sys
sys.path.append('/workspace/recon-platform')

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState

def test_working_demo():
    """Test a demo network that actually works correctly"""

    print("=== Testing Working Demo Network ===")

    # Create network
    demo_graph = ReCoNGraph()
    demo_graph.add_node("Root", "script")
    demo_graph.add_node("A", "script")
    demo_graph.add_node("B", "script")
    demo_graph.add_node("TA", "terminal")
    demo_graph.add_node("TB", "terminal")

    # Links for sequence: Root -> A (with TA), then A -> B (with TB)
    demo_graph.add_link("Root", "A", "sub")
    demo_graph.add_link("Root", "B", "sub")
    demo_graph.add_link("A", "B", "por")  # A inhibits B until A completes
    demo_graph.add_link("A", "TA", "sub")
    demo_graph.add_link("B", "TB", "sub")

    # Start execution
    demo_graph.request_root("Root")

    print("Manual execution simulation:")
    for step in range(20):
        states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
        print(f"Step {step}: {states}")

        # Check terminal states and manually intervene
        ta_node = demo_graph.get_node("TA")
        tb_node = demo_graph.get_node("TB")

        # When TA gets requested and confirms, let it succeed
        if step == 5 and ta_node.state == ReCoNState.CONFIRMED:
            print("  --> ✅ TA confirmed automatically - A should complete")

        # When TB gets requested and confirms, let it succeed
        if step >= 8 and tb_node.state == ReCoNState.CONFIRMED:
            print("  --> ✅ TB confirmed automatically - B should complete")

        demo_graph.propagate_step()

        if demo_graph.is_completed():
            print("✅ Network completed!")
            break

    final_states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
    print(f"\nFinal states: {final_states}")

    # Check if sequence worked
    success = (
        final_states["A"] in ["true", "confirmed"] and
        final_states["B"] in ["true", "confirmed"] and
        final_states["Root"] in ["true", "confirmed"]
    )

    return success, final_states

def test_manual_intervention_demo():
    """Test demo where we manually fail then succeed terminals"""

    print("\n=== Testing Manual Intervention Demo ===")

    demo_graph = ReCoNGraph()
    demo_graph.add_node("Root", "script")
    demo_graph.add_node("A", "script")
    demo_graph.add_node("B", "script")
    demo_graph.add_node("TA", "terminal")
    demo_graph.add_node("TB", "terminal")

    # Set terminals to initially fail
    demo_graph.get_node("TA").measurement_fn = lambda env: 0.0  # Fail
    demo_graph.get_node("TB").measurement_fn = lambda env: 0.0  # Fail

    demo_graph.add_link("Root", "A", "sub")
    demo_graph.add_link("Root", "B", "sub")
    demo_graph.add_link("A", "B", "por")
    demo_graph.add_link("A", "TA", "sub")
    demo_graph.add_link("B", "TB", "sub")

    demo_graph.request_root("Root")

    ta_confirmed = False
    tb_confirmed = False

    print("Manual intervention execution:")
    for step in range(25):
        states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
        print(f"Step {step}: {states}")

        ta_node = demo_graph.get_node("TA")
        tb_node = demo_graph.get_node("TB")

        # When TA is requested and fails, manually make it succeed
        if not ta_confirmed and ta_node.state == ReCoNState.FAILED:
            print("  --> 🎯 MANUAL: Making TA succeed")
            ta_node.measurement_fn = lambda env: 1.0  # Now succeed
            ta_node.state = ReCoNState.INACTIVE  # Reset to try again
            ta_confirmed = True

        # When TB is requested and fails, manually make it succeed
        if not tb_confirmed and tb_node.state == ReCoNState.FAILED and ta_confirmed:
            print("  --> 🎯 MANUAL: Making TB succeed")
            tb_node.measurement_fn = lambda env: 1.0  # Now succeed
            tb_node.state = ReCoNState.INACTIVE  # Reset to try again
            tb_confirmed = True

        demo_graph.propagate_step()

        if demo_graph.is_completed():
            print("✅ Manual intervention demo completed!")
            break

    final_states = {nid: demo_graph.get_node(nid).state.value for nid in ["Root", "A", "B", "TA", "TB"]}
    print(f"\nManual intervention final: {final_states}")

    return final_states

if __name__ == "__main__":
    print("Testing working demo networks...\n")

    success1, result1 = test_working_demo()
    result2 = test_manual_intervention_demo()

    print(f"\n=== RESULTS ===")
    print(f"Auto-confirm demo success: {success1}")
    print(f"Auto-confirm final: {result1}")
    print(f"Manual intervention final: {result2}")

    if success1:
        print("✅ Found a working demo structure!")
    else:
        print("❌ Still looking for working structure...")