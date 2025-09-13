#!/usr/bin/env python3
"""
Test for game_id handling in BlindSquirrel ReCoN implementation
"""

import sys
import os

# Add paths for both implementations
sys.path.insert(0, '/workspace/recon-platform')
sys.path.insert(0, '/workspace/ARC-AGI-3-Agents-blindsquirrel')

from agents.structs import GameAction, GameState

# Mock frame data for testing
class MockFrame:
    def __init__(self, game_id: str, score: int, frame_data=None, state: str = "NOT_FINISHED"):
        self.game_id = game_id
        self.score = score
        self.frame = [frame_data] if frame_data else []
        self.state = GameState.NOT_FINISHED if state == "NOT_FINISHED" else GameState.WIN
        if state == "NOT_PLAYED":
            self.state = GameState.NOT_PLAYED
            self.frame = []
        self.available_actions = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6
        ]

def test_agent_game_id_tracking():
    """Test how agent tracks game_id."""
    print("=== Testing Agent game_id Tracking ===")

    from recon_agents.blindsquirrel.agent import BlindSquirrelAgent

    # Create agent
    agent = BlindSquirrelAgent()

    print(f"Initial agent.game_id: {repr(agent.game_id)}")

    # Test NOT_PLAYED frame
    not_played_frame = MockFrame("test_game", 0, None, "NOT_PLAYED")
    print(f"NOT_PLAYED frame game_id: {repr(not_played_frame.game_id)}")

    agent.process_latest_frame(not_played_frame)
    print(f"After NOT_PLAYED - agent.game_id: {repr(agent.game_id)}")

    # Test first game frame
    first_frame = MockFrame("test_game", 0, [[0] * 8 for _ in range(8)])
    print(f"First frame game_id: {repr(first_frame.game_id)}")

    agent.process_latest_frame(first_frame)
    print(f"After first frame - agent.game_id: {repr(agent.game_id)}")

    # Test action selection to see what game_id is used in warnings
    print(f"\nTesting action selection with game_id: {repr(agent.game_id)}")

    # Manually disable all actions to trigger warning
    if agent.current_state:
        for key in agent.current_state.action_rweights:
            agent.current_state.action_rweights[key] = 0

        # Try to select action - should trigger "No Actions" warning
        try:
            action = agent._get_rweights_action()
            print(f"Selected action: {action}")
        except Exception as e:
            print(f"Error selecting action: {e}")

def test_original_agent_game_id():
    """Test how original agent tracks game_id."""
    print("\n=== Testing Original Agent game_id Tracking ===")

    try:
        from agents.agent import Agent
        from agents.blind_squirrel import BlindSquirrel

        # Create mock agent with required args
        class MockBlindSquirrel(BlindSquirrel):
            def __init__(self):
                # Don't call super().__init__ to avoid argument requirements
                self.prev_state = None
                self.current_state = None
                self.game_id = None  # This is the key
                self.graph = None

        agent = MockBlindSquirrel()
        print(f"Original agent.game_id initially: {repr(getattr(agent, 'game_id', 'NOT_SET'))}")

        # Test NOT_PLAYED
        not_played_frame = MockFrame("test_game", 0, None, "NOT_PLAYED")
        agent.process_latest_frame(not_played_frame)
        print(f"After NOT_PLAYED - original agent.game_id: {repr(getattr(agent, 'game_id', 'NOT_SET'))}")

        # Test first frame
        first_frame = MockFrame("test_game", 0, [[0] * 8 for _ in range(8)])
        agent.process_latest_frame(first_frame)
        print(f"After first frame - original agent.game_id: {repr(getattr(agent, 'game_id', 'NOT_SET'))}")

    except Exception as e:
        print(f"Could not test original agent: {e}")

def main():
    """Run game_id tracking tests."""
    print("BlindSquirrel game_id Tracking Test")
    print("=" * 50)

    try:
        test_agent_game_id_tracking()
        test_original_agent_game_id()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()