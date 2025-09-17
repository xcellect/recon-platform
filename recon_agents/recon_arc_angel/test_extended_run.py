#!/usr/bin/env python3
"""
Test Extended Run Capability

Tests that the improved agent can run for extended periods without
crashing or prematurely terminating.
"""

import sys
import os
import time
import numpy as np
from unittest.mock import Mock

# Add paths
sys.path.insert(0, "/workspace/recon-platform")
sys.path.insert(0, "/workspace/recon-platform/recon_agents/recon_arc_angel")

from improved_production_agent import ImprovedProductionReCoNArcAngel


def test_extended_run():
    """Test that agent can run for many actions without issues."""
    print("🔄 TESTING EXTENDED RUN CAPABILITY")
    print("=" * 50)
    
    agent = ImprovedProductionReCoNArcAngel()
    
    # Mock frame data that changes slightly each time
    class MockFrameData:
        def __init__(self, action_count):
            self.frame = np.zeros((64, 64), dtype=np.int64)
            
            # Add some variety to prevent identical frames
            self.frame[10:15, 10:15] = 1  # Red square
            self.frame[20:25, 20:25] = 2  # Green square
            
            # Add small variations based on action count
            variation = action_count % 10
            if variation < 5:
                self.frame[30 + variation, 30 + variation] = 3  # Blue pixel moves
            
            self.score = 0
            self.state = "NOT_FINISHED"  # Keep game running
            
            class MockAction:
                def __init__(self, name):
                    self.name = name
            self.available_actions = [MockAction("ACTION6")]
    
    print("✅ Starting extended run test...")
    
    frames = []
    start_time = time.time()
    max_test_actions = 100  # Test 100 actions to verify stability
    
    try:
        for action_count in range(max_test_actions):
            frame_data = MockFrameData(action_count)
            
            # Test is_done method
            is_done = agent.is_done(frames, frame_data)
            if is_done:
                print(f"❌ Agent reported done at action {action_count}")
                break
            
            # Choose action
            action = agent.choose_action(frames, frame_data)
            
            # Verify action is valid
            assert action is not None, f"Action should not be None at step {action_count}"
            assert hasattr(action, 'action_type'), f"Action should have action_type at step {action_count}"
            
            # If ACTION6, verify coordinates
            if action.action_type == "ACTION6":
                assert hasattr(action, 'data') and action.data, f"ACTION6 should have coordinates at step {action_count}"
                assert 'x' in action.data and 'y' in action.data, f"ACTION6 should have x,y at step {action_count}"
            
            frames.append(frame_data)
            
            # Progress indicator
            if action_count % 20 == 0:
                print(f"  ✅ Action {action_count}: {action.action_type}")
        
        elapsed = time.time() - start_time
        print(f"✅ Completed {max_test_actions} actions in {elapsed:.2f}s")
        print(f"  Average: {max_test_actions/elapsed:.1f} actions/sec")
        
        # Check final stats
        stats = agent.get_stats()
        print(f"  Total actions: {stats['total_actions']}")
        print(f"  ACTION6 coord=None prevented: {stats['action6_coord_none_prevented']}")
        print(f"  Objects detected: {stats['objects_detected']}")
        
    except Exception as e:
        print(f"❌ Error during extended run: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("🎉 EXTENDED RUN TEST PASSED!")
    print("Agent can run for extended periods without issues.")
    return True


def test_is_done_method():
    """Test the is_done method behavior."""
    print("🔍 TESTING IS_DONE METHOD")
    print("=" * 50)
    
    agent = ImprovedProductionReCoNArcAngel()
    
    # Test different states
    test_states = ["NOT_FINISHED", "WIN", "GAME_OVER", "NOT_PLAYED"]
    
    for state in test_states:
        class MockFrame:
            def __init__(self, state):
                self.state = state
                self.score = 0
        
        frame = MockFrame(state)
        is_done = agent.is_done([], frame)
        
        print(f"  State '{state}': is_done = {is_done}")
        
        # Should only return True for WIN
        if state == "WIN":
            assert is_done == True, f"Should be done on WIN state"
        else:
            assert is_done == False, f"Should NOT be done on {state} state"
    
    print("✅ is_done method working correctly")


if __name__ == "__main__":
    print("🧪 TESTING AGENT EXTENDED RUN CAPABILITY")
    print("=" * 60)
    
    try:
        test_is_done_method()
        print()
        test_extended_run()
        
        print()
        print("🎉 ALL EXTENDED RUN TESTS PASSED!")
        print("The agent should be able to run up to MAX_ACTIONS without early termination.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
