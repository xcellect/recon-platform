#!/usr/bin/env python3
"""
BlindSquirrel 1-to-1 Mapping Test Script

Tests the ReCoN implementation against the original to identify discrepancies.
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

def test_state_creation():
    """Test state creation and equality."""
    print("=== Testing State Creation ===")

    # Import both implementations
    from agents.blind_squirrel import State as OriginalState
    from recon_agents.blindsquirrel.state_graph import BlindSquirrelState as ReCoNState

    # Create test frame
    frame = create_test_frame()

    # Create states
    orig_state = OriginalState(frame)
    recon_state = ReCoNState(frame)

    print(f"Original state - game_id: {orig_state.game_id}, score: {orig_state.score}")
    print(f"ReCoN state - game_id: {recon_state.game_id}, score: {recon_state.score}")

    print(f"Original frame type: {type(orig_state.frame)}")
    print(f"ReCoN frame type: {type(recon_state.frame)}")

    print(f"Original num_actions: {orig_state.num_actions}")
    print(f"ReCoN num_actions: {recon_state.num_actions}")

    print(f"Original object_data count: {len(orig_state.object_data)}")
    print(f"ReCoN object_data count: {len(recon_state.object_data)}")

    # Test equality
    orig_state2 = OriginalState(frame)
    recon_state2 = ReCoNState(frame)

    print(f"Original state equality: {orig_state == orig_state2}")
    print(f"ReCoN state equality: {recon_state == recon_state2}")

    # Test action weights initialization
    print(f"Original action_rweights: {list(orig_state.action_rweights.items())[:10]}")
    print(f"ReCoN action_rweights: {list(recon_state.action_rweights.items())[:10]}")

def test_action_availability():
    """Test action availability logic."""
    print("\n=== Testing Action Availability ===")

    from agents.blind_squirrel import State as OriginalState
    from recon_agents.blindsquirrel.state_graph import BlindSquirrelState as ReCoNState

    # Test with limited actions
    frame = create_test_frame()
    frame.available_actions = [GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION6]

    orig_state = OriginalState(frame)
    recon_state = ReCoNState(frame)

    print("Limited actions test (only ACTION1, ACTION3, ACTION6):")
    print(f"Original disabled actions: {[k for k, v in orig_state.action_rweights.items() if v == 0]}")
    print(f"ReCoN disabled actions: {[k for k, v in recon_state.action_rweights.items() if v == 0]}")

def test_state_graph():
    """Test state graph behavior."""
    print("\n=== Testing State Graph ===")

    from agents.blind_squirrel import StateGraph as OriginalGraph
    from recon_agents.blindsquirrel.state_graph import BlindSquirrelStateGraph as ReCoNGraph
    import agents.blind_squirrel as orig_module

    # Create graphs
    orig_graph = OriginalGraph()
    recon_graph = ReCoNGraph()

    # Disable training to avoid model errors in test
    orig_epsilon = orig_module.AGENT_E
    orig_module.AGENT_E = 2.0  # Disable training (EPSILON >= 1)
    recon_graph.EPSILON = 2.0  # Disable training

    # Create initial states
    frame1 = create_test_frame("test", 0)
    orig_state1 = orig_graph.get_state(frame1)
    recon_state1 = recon_graph.get_state(frame1)

    orig_graph.add_init_state(orig_state1)
    recon_graph.add_init_state(recon_state1)

    print(f"Original graph states: {len(orig_graph.states)}")
    print(f"ReCoN graph states: {len(recon_graph.states)}")

    # Create second state with same data (should be same object)
    frame2 = create_test_frame("test", 0)
    frame2.frame = frame1.frame  # Same frame data

    orig_state2 = orig_graph.get_state(frame2)
    recon_state2 = recon_graph.get_state(frame2)

    print(f"Same frame - Original same object: {orig_state1 is orig_state2}")
    print(f"Same frame - ReCoN same object: {recon_state1 is recon_state2}")

    # Test transition
    frame3 = create_test_frame("test", 1)  # Higher score
    orig_state3 = orig_graph.get_state(frame3)
    recon_state3 = recon_graph.get_state(frame3)

    # Simulate action and update
    action = 0  # ACTION1

    print(f"\nBefore update - Original action_rweights[{action}]: {orig_state1.action_rweights[action]}")
    print(f"Before update - ReCoN action_rweights[{action}]: {recon_state1.action_rweights[action]}")

    orig_graph.update(orig_state1, action, orig_state3)
    recon_graph.update(recon_state1, action, recon_state3)

    print(f"After update - Original action_rweights[{action}]: {orig_state1.action_rweights[action]}")
    print(f"After update - ReCoN action_rweights[{action}]: {recon_state1.action_rweights[action]}")

    # Restore original epsilon
    orig_module.AGENT_E = orig_epsilon

def test_bad_action_scenario():
    """Test the specific scenario causing 'Bad Action' warnings."""
    print("\n=== Testing Bad Action Scenario ===")

    from agents.blind_squirrel import StateGraph as OriginalGraph
    from recon_agents.blindsquirrel.state_graph import BlindSquirrelStateGraph as ReCoNGraph
    import agents.blind_squirrel as orig_module

    # Create graphs
    orig_graph = OriginalGraph()
    recon_graph = ReCoNGraph()

    # Disable training to avoid model errors
    orig_epsilon = orig_module.AGENT_E
    orig_module.AGENT_E = 2.0
    recon_graph.EPSILON = 2.0

    # Create initial state
    frame1 = create_test_frame("test", 0)
    orig_state1 = orig_graph.get_state(frame1)
    recon_state1 = recon_graph.get_state(frame1)

    orig_graph.add_init_state(orig_state1)
    recon_graph.add_init_state(recon_state1)

    # Simulate a "bad action" (same state returned)
    action = 1  # ACTION2

    print(f"Testing bad action scenario (action {action} returns same state):")

    # Same state = bad action
    orig_graph.update(orig_state1, action, orig_state1)
    recon_graph.update(recon_state1, action, recon_state1)

    print(f"Original action_rweights[{action}] after bad action: {orig_state1.action_rweights[action]}")
    print(f"ReCoN action_rweights[{action}] after bad action: {recon_state1.action_rweights[action]}")

    # Check if all actions are disabled
    orig_all_disabled = all(v == 0 for v in orig_state1.action_rweights.values())
    recon_all_disabled = all(v == 0 for v in recon_state1.action_rweights.values())

    print(f"Original all actions disabled: {orig_all_disabled}")
    print(f"ReCoN all actions disabled: {recon_all_disabled}")

    if orig_all_disabled != recon_all_disabled:
        print("DISCREPANCY: Different behavior in action disabling!")

    # Restore original epsilon
    orig_module.AGENT_E = orig_epsilon

def test_agents():
    """Test agent behavior."""
    print("\n=== Testing Agent Behavior ===")

    # Import both agents
    from agents.blind_squirrel import BlindSquirrel as OriginalAgent
    from recon_agents.blindsquirrel.agent import BlindSquirrelAgent as ReCoNAgent

    # Create agents
    orig_agent = OriginalAgent()
    recon_agent = ReCoNAgent()

    # Test NOT_PLAYED frame
    not_played_frame = MockFrame("test", 0, [], "NOT_PLAYED")
    not_played_frame.state = GameState.NOT_PLAYED
    not_played_frame.frame = []

    print("Testing NOT_PLAYED frame:")
    orig_agent.process_latest_frame(not_played_frame)
    recon_agent.process_latest_frame(not_played_frame)

    print(f"Original prev_state: {orig_agent.prev_state}")
    print(f"ReCoN prev_state: {recon_agent.prev_state}")

    # Test first real frame
    first_frame = create_test_frame("test", 0)

    print("\nTesting first game frame:")
    orig_agent.process_latest_frame(first_frame)
    recon_agent.process_latest_frame(first_frame)

    print(f"Original current_state exists: {orig_agent.current_state is not None}")
    print(f"ReCoN current_state exists: {recon_agent.current_state is not None}")

    print(f"Original game_id: {getattr(orig_agent, 'game_id', 'None')}")
    print(f"ReCoN game_id: {getattr(recon_agent, 'game_id', 'None')}")

    # Test action selection
    try:
        orig_action = orig_agent.choose_action([], first_frame)
        print(f"Original action: {orig_action}")
    except Exception as e:
        print(f"Original action error: {e}")

    try:
        recon_action = recon_agent._choose_action(first_frame)
        print(f"ReCoN action: {recon_action}")
    except Exception as e:
        print(f"ReCoN action error: {e}")

def main():
    """Run all tests."""
    print("BlindSquirrel 1-to-1 Mapping Test")
    print("=" * 50)

    random.seed(42)
    np.random.seed(42)

    try:
        test_state_creation()
        test_action_availability()
        test_state_graph()
        test_bad_action_scenario()
        test_agents()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()