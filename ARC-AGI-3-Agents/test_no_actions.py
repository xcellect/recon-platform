#!/usr/bin/env python3
"""
Test for "No Actions" issue in BlindSquirrel ReCoN implementation
"""

import sys
import os
import random
import numpy as np
from typing import Any, List

# Add paths for both implementations
sys.path.insert(0, '/workspace/recon-platform')
sys.path.insert(0, '/workspace/ARC-AGI-3-Agents-blindsquirrel')

from agents.structs import GameAction, GameState

# Mock frame data for testing
class MockFrame:
    def __init__(self, game_id: str, score: int, frame_data: List[List[int]],
                 state: str = "NOT_FINISHED"):
        self.game_id = game_id
        self.score = score
        self.frame = [frame_data] if frame_data else []
        self.state = GameState.NOT_FINISHED if state == "NOT_FINISHED" else GameState.WIN
        self.available_actions = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6
        ]

def create_test_frame(game_id="test", score=0, size=8):
    """Create a test frame with random data."""
    frame_data = np.random.randint(0, 16, (size, size)).tolist()
    return MockFrame(game_id, score, frame_data)

def test_zero_back_propagation():
    """Test the zero_back propagation that causes 'No Actions'."""
    print("=== Testing zero_back Propagation ===")

    from agents.blind_squirrel import StateGraph as OriginalGraph
    from recon_agents.blindsquirrel.state_graph import BlindSquirrelStateGraph as ReCoNGraph
    import agents.blind_squirrel as orig_module

    # Disable training
    orig_epsilon = orig_module.AGENT_E
    orig_module.AGENT_E = 2.0

    # Create graphs
    orig_graph = OriginalGraph()
    recon_graph = ReCoNGraph()
    recon_graph.EPSILON = 2.0

    # Create a chain of states: state1 -> state2 -> state3
    frame1 = create_test_frame("test", 0)
    frame2 = create_test_frame("test", 0)  # Same level
    frame3 = create_test_frame("test", 0)  # Same level

    # Make sure they're different states
    frame2.frame = [[[1] * 8 for _ in range(8)]]  # Different frame
    frame3.frame = [[[2] * 8 for _ in range(8)]]  # Different frame

    # Create states
    orig_state1 = orig_graph.get_state(frame1)
    orig_state2 = orig_graph.get_state(frame2)
    orig_state3 = orig_graph.get_state(frame3)

    recon_state1 = recon_graph.get_state(frame1)
    recon_state2 = recon_graph.get_state(frame2)
    recon_state3 = recon_graph.get_state(frame3)

    # Add initial state
    orig_graph.add_init_state(orig_state1)
    recon_graph.add_init_state(recon_state1)

    print(f"Created 3 different states")
    print(f"Original states equal: {orig_state1 == orig_state2}, {orig_state2 == orig_state3}")
    print(f"ReCoN states equal: {recon_state1 == recon_state2}, {recon_state2 == recon_state3}")

    # Create chain: state1 -action0-> state2 -action1-> state3
    action1, action2 = 0, 1

    # First transition (good)
    orig_graph.update(orig_state1, action1, orig_state2)
    recon_graph.update(recon_state1, action1, recon_state2)

    # Second transition (good)
    orig_graph.update(orig_state2, action2, orig_state3)
    recon_graph.update(recon_state2, action2, recon_state3)

    print(f"\nAfter good transitions:")
    print(f"Original state1 action_rweights[{action1}]: {orig_state1.action_rweights[action1]}")
    print(f"ReCoN state1 action_rweights[{action1}]: {recon_state1.action_rweights[action1]}")
    print(f"Original state2 action_rweights[{action2}]: {orig_state2.action_rweights[action2]}")
    print(f"ReCoN state2 action_rweights[{action2}]: {recon_state2.action_rweights[action2]}")

    # Now make state3 have a bad action that returns to itself
    action3 = 2
    print(f"\nMaking bad action {action3} on state3 (returns to itself):")

    orig_graph.update(orig_state3, action3, orig_state3)
    recon_graph.update(recon_state3, action3, recon_state3)

    print(f"Original state3 action_rweights[{action3}]: {orig_state3.action_rweights[action3]}")
    print(f"ReCoN state3 action_rweights[{action3}]: {recon_state3.action_rweights[action3]}")

    # Check if zero_back propagated
    print(f"\nAfter bad action and zero_back:")
    print(f"Original state1 action_rweights[{action1}]: {orig_state1.action_rweights[action1]}")
    print(f"ReCoN state1 action_rweights[{action1}]: {recon_state1.action_rweights[action1]}")
    print(f"Original state2 action_rweights[{action2}]: {orig_state2.action_rweights[action2]}")
    print(f"ReCoN state2 action_rweights[{action2}]: {recon_state2.action_rweights[action2]}")

    # Check if all actions are disabled in any state
    def count_disabled_actions(state):
        return sum(1 for v in state.action_rweights.values() if v == 0)

    def count_enabled_actions(state):
        return sum(1 for v in state.action_rweights.values() if v is None or v == 1)

    print(f"\nAction status summary:")
    for i, (orig_state, recon_state, name) in enumerate([
        (orig_state1, recon_state1, "state1"),
        (orig_state2, recon_state2, "state2"),
        (orig_state3, recon_state3, "state3")
    ]):
        orig_disabled = count_disabled_actions(orig_state)
        orig_enabled = count_enabled_actions(orig_state)
        recon_disabled = count_disabled_actions(recon_state)
        recon_enabled = count_enabled_actions(recon_state)

        print(f"{name}: Original disabled={orig_disabled}, enabled={orig_enabled}")
        print(f"{name}: ReCoN disabled={recon_disabled}, enabled={recon_enabled}")

        if orig_enabled == 0:
            print(f"  WARNING: Original {name} has NO ACTIONS available!")
        if recon_enabled == 0:
            print(f"  WARNING: ReCoN {name} has NO ACTIONS available!")

    # Restore epsilon
    orig_module.AGENT_E = orig_epsilon

def test_action_selection_with_disabled_actions():
    """Test what happens when trying to select actions with many disabled."""
    print("\n=== Testing Action Selection with Disabled Actions ===")

    from recon_agents.blindsquirrel.state_graph import BlindSquirrelState as ReCoNState

    # Create a state
    frame = create_test_frame("test", 0)
    state = ReCoNState(frame)

    # Manually disable most actions (simulating heavy zero_back)
    total_actions = len(state.action_rweights)
    print(f"Total actions: {total_actions}")

    # Disable all but 2 actions
    for i in range(total_actions - 2):
        state.action_rweights[i] = 0

    print(f"Disabled {total_actions - 2} actions, left 2 enabled")

    # Count available actions
    available_actions = [i for i, weight in state.action_rweights.items() if weight != 0]
    print(f"Available actions: {available_actions}")

    # Now disable the remaining actions
    for i in available_actions:
        state.action_rweights[i] = 0

    # Check if any actions remain
    final_available = [i for i, weight in state.action_rweights.items() if weight != 0]
    print(f"Final available actions: {final_available}")

    if not final_available:
        print("NO ACTIONS AVAILABLE - this would trigger 'Warning: No Actions'")

def main():
    """Run focused tests on the No Actions issue."""
    print("BlindSquirrel No Actions Issue Test")
    print("=" * 50)

    random.seed(42)
    np.random.seed(42)

    try:
        test_zero_back_propagation()
        test_action_selection_with_disabled_actions()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()