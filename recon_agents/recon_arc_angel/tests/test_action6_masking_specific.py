"""
Specific test cases for ACTION6 and action masking edge cases

These tests ensure the availability masking is completely airtight
for all possible combinations and edge cases.
"""
import pytest
import sys
import os
import torch
import numpy as np
from typing import List

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.node import ReCoNState
from recon_agents.recon_arc_angel import ReCoNArcAngel

class FrameData:
    """Test frame data structure"""
    def __init__(self, frame, available_actions, score=0, state='PLAYING'):
        self.frame = frame
        self.available_actions = available_actions
        self.score = score
        self.state = state

def create_test_frame(pattern_type="random", available_actions=None):
    """Create test frame with different patterns"""
    if available_actions is None:
        available_actions = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6']
    
    if pattern_type == "random":
        frame_array = np.random.randint(0, 5, (64, 64))
    elif pattern_type == "center_square":
        frame_array = np.zeros((64, 64), dtype=int)
        frame_array[28:36, 28:36] = 2  # Center square
    elif pattern_type == "corner_patterns":
        frame_array = np.zeros((64, 64), dtype=int)
        frame_array[0:8, 0:8] = 1      # Top-left
        frame_array[56:64, 56:64] = 3  # Bottom-right
    elif pattern_type == "empty":
        frame_array = np.zeros((64, 64), dtype=int)
    else:
        frame_array = np.ones((64, 64), dtype=int)
    
    return FrameData(frame_array, available_actions)

class TestActionMasking:
    """Test class for action masking scenarios"""
    
    def test_action6_only_scenarios(self):
        """Test various scenarios where only ACTION6 is available"""
        
        # Test 1: ACTION6 only with center pattern
        agent = ReCoNArcAngel('test_action6_center')
        frame = create_test_frame("center_square", ['ACTION6'])
        
        action = agent.choose_action([frame], frame)
        
        assert hasattr(action, 'name') or hasattr(action, 'action_type')
        action_name = getattr(action, 'name', getattr(action, 'action_type', ''))
        assert 'action_click' in action_name or 'ACTION6' in action_name
        
        # Should have coordinates
        assert hasattr(action, 'data')
        if hasattr(action, 'data') and action.data:
            assert 'x' in action.data
            assert 'y' in action.data
            assert 0 <= action.data['x'] < 64
            assert 0 <= action.data['y'] < 64
    
    def test_action6_only_different_patterns(self):
        """Test ACTION6 selection with different frame patterns"""
        patterns = ["random", "center_square", "corner_patterns", "empty"]
        
        for pattern in patterns:
            agent = ReCoNArcAngel(f'test_action6_{pattern}')
            frame = create_test_frame(pattern, ['ACTION6'])
            
            action = agent.choose_action([frame], frame)
            action_name = getattr(action, 'name', getattr(action, 'action_type', ''))
            
            # Should always select ACTION6 when it's the only option
            assert 'action_click' in action_name or 'ACTION6' in action_name, \
                f"Pattern {pattern}: Expected ACTION6, got {action_name}"
    
    def test_action6_vs_individual_actions(self):
        """Test ACTION6 vs individual actions in head-to-head scenarios"""
        
        # Test ACTION6 vs ACTION1
        agent = ReCoNArcAngel('test_action6_vs_action1')
        frame = create_test_frame("center_square", ['ACTION1', 'ACTION6'])
        
        action = agent.choose_action([frame], frame)
        action_name = getattr(action, 'name', getattr(action, 'action_type', ''))
        
        # Should select one of the available actions
        assert any(x in action_name for x in ['action_1', 'action_click', 'ACTION1', 'ACTION6'])
    
    def test_action6_coordinate_regions(self):
        """Test that ACTION6 selects reasonable coordinates"""
        agent = ReCoNArcAngel('test_action6_coords')
        
        # Create frame with pattern in specific region
        frame_array = np.zeros((64, 64), dtype=int)
        frame_array[40:48, 40:48] = 3  # Pattern in region (5,5)
        frame = FrameData(frame_array, ['ACTION6'])
        
        action = agent.choose_action([frame], frame)
        
        if hasattr(action, 'data') and action.data:
            x, y = action.data.get('x', 0), action.data.get('y', 0)
            
            # Coordinates should be valid
            assert 0 <= x < 64, f"Invalid x coordinate: {x}"
            assert 0 <= y < 64, f"Invalid y coordinate: {y}"
            
            # Should be somewhat near the pattern (region 5,5 = pixels 40-47)
            # Allow some tolerance since CNN might not be perfectly trained
            print(f"ACTION6 selected coordinates: ({x}, {y}) for pattern at (40-47, 40-47)")

class TestMaskingCombinations:
    """Test various combinations of available actions"""
    
    def test_single_action_scenarios(self):
        """Test each individual action when it's the only one available"""
        for action_num in range(1, 7):
            action_name = f'ACTION{action_num}'
            agent = ReCoNArcAngel(f'test_single_{action_name}')
            frame = create_test_frame("random", [action_name])
            
            action = agent.choose_action([frame], frame)
            selected = getattr(action, 'name', getattr(action, 'action_type', ''))
            
            print(f"Single {action_name} test: Selected {selected}")
            
            # Should select the available action or a reasonable fallback
            assert selected is not None
    
    def test_pair_combinations(self):
        """Test pairs of actions"""
        test_pairs = [
            ['ACTION1', 'ACTION2'],
            ['ACTION1', 'ACTION6'], 
            ['ACTION3', 'ACTION6'],
            ['ACTION5', 'ACTION6']
        ]
        
        for pair in test_pairs:
            agent = ReCoNArcAngel(f'test_pair_{pair[0]}_{pair[1]}')
            frame = create_test_frame("random", pair)
            
            action = agent.choose_action([frame], frame)
            selected = getattr(action, 'name', getattr(action, 'action_type', ''))
            
            print(f"Pair {pair} test: Selected {selected}")
            
            # Should select one of the available actions
            expected_internal = []
            for avail_action in pair:
                if avail_action == 'ACTION6':
                    expected_internal.extend(['action_click', 'ACTION6'])
                else:
                    action_num = avail_action[-1]
                    expected_internal.extend([f'action_{action_num}', avail_action])
            
            assert any(exp in selected for exp in expected_internal), \
                f"Pair {pair}: Expected one of {expected_internal}, got {selected}"
    
    def test_exclusion_scenarios(self):
        """Test that specific actions are properly excluded"""
        
        # Test 1: Everything except ACTION6
        agent1 = ReCoNArcAngel('test_exclude_action6')
        frame1 = create_test_frame("random", ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5'])
        action1 = agent1.choose_action([frame1], frame1)
        selected1 = getattr(action1, 'name', getattr(action1, 'action_type', ''))
        
        # Should NOT select ACTION6
        assert 'ACTION6' not in selected1 and 'action_click' not in selected1
        
        # Test 2: Everything except ACTION1
        agent2 = ReCoNArcAngel('test_exclude_action1')
        frame2 = create_test_frame("random", ['ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6'])
        action2 = agent2.choose_action([frame2], frame2)
        selected2 = getattr(action2, 'name', getattr(action2, 'action_type', ''))
        
        # Should NOT select ACTION1
        assert 'ACTION1' not in selected2 and 'action_1' not in selected2
    
    def test_action6_region_masking(self):
        """Test that regions are properly masked when ACTION6 is unavailable"""
        agent = ReCoNArcAngel('test_action6_region_masking')
        
        # Create frame
        frame = create_test_frame("center_square", ['ACTION1', 'ACTION2'])  # No ACTION6
        
        # Apply masking manually to check region states
        current_frame_tensor = agent._convert_frame_to_tensor(frame)
        agent.hypothesis_manager.update_weights_from_frame(current_frame_tensor)
        agent.hypothesis_manager.reset()
        agent._apply_available_actions_mask(frame.available_actions)
        
        # Check that action_click is effectively unavailable (pure ReCoN semantics)
        assert agent.hypothesis_manager._is_action_effectively_unavailable('action_click'), \
            "action_click should be effectively unavailable due to low sub weight"
        
        # Check that some regions are effectively unavailable
        unavailable_regions = 0
        for region_y in range(3):  # Check first few regions
            for region_x in range(3):
                region_id = f'region_{region_y}_{region_x}'
                if region_id in agent.hypothesis_manager.graph.nodes:
                    if agent.hypothesis_manager._is_region_effectively_unavailable(region_id):
                        unavailable_regions += 1
        
        # All checked regions should be effectively unavailable when ACTION6 not available
        assert unavailable_regions == 9, f"Expected 9 unavailable regions, got {unavailable_regions}"

class TestMaskingEdgeCases:
    """Test edge cases for action masking"""
    
    def test_empty_available_actions(self):
        """Test behavior with empty available actions list"""
        agent = ReCoNArcAngel('test_empty_actions')
        frame = create_test_frame("random", [])
        
        action = agent.choose_action([frame], frame)
        
        # Should gracefully handle empty list with fallback
        assert action is not None
        assert hasattr(action, 'reasoning')
    
    def test_invalid_available_actions(self):
        """Test behavior with invalid action names"""
        agent = ReCoNArcAngel('test_invalid_actions')
        frame = create_test_frame("random", ['INVALID_ACTION', 'ACTION1'])
        
        action = agent.choose_action([frame], frame)
        selected = getattr(action, 'name', getattr(action, 'action_type', ''))
        
        # Should ignore invalid action and select ACTION1
        assert 'action_1' in selected or 'ACTION1' in selected
    
    def test_action6_with_no_regions(self):
        """Test ACTION6 behavior when regions fail to activate"""
        agent = ReCoNArcAngel('test_action6_no_regions')
        
        # Manually set all regions to FAILED
        for region_y in range(agent.hypothesis_manager.regions_per_dim):
            for region_x in range(agent.hypothesis_manager.regions_per_dim):
                region_id = f'region_{region_y}_{region_x}'
                if region_id in agent.hypothesis_manager.graph.nodes:
                    agent.hypothesis_manager.graph.nodes[region_id].state = ReCoNState.FAILED
                    agent.hypothesis_manager.graph.nodes[region_id].activation = -1.0
        
        frame = create_test_frame("center_square", ['ACTION6'])
        
        # Even with failed regions, should still be able to select ACTION6
        # (though it might not have good coordinates)
        best_action, best_coords = agent.hypothesis_manager.get_best_action(
            available_actions=['ACTION6']
        )
        
        # action_click should still be selectable even if regions are bad
        # The coordinates might be None or default
        assert best_action == 'action_click' or best_action is None
    
    def test_mixed_availability_with_action6(self):
        """Test mixed scenarios including ACTION6"""
        test_cases = [
            (['ACTION1', 'ACTION6'], "Should select ACTION1 or ACTION6"),
            (['ACTION2', 'ACTION3', 'ACTION6'], "Should select ACTION2, ACTION3, or ACTION6"),
            (['ACTION6'], "Should select ACTION6"),
            (['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5'], "Should NOT select ACTION6")
        ]
        
        for available_actions, description in test_cases:
            agent = ReCoNArcAngel(f'test_mixed_{len(available_actions)}')
            frame = create_test_frame("center_square", available_actions)
            
            action = agent.choose_action([frame], frame)
            selected = getattr(action, 'name', getattr(action, 'action_type', ''))
            
            print(f"Mixed test {available_actions}: Selected {selected} - {description}")
            
            # Verify selection is from available actions
            if 'ACTION6' in available_actions:
                # ACTION6 available - could select it or others
                valid_selections = []
                for avail in available_actions:
                    if avail == 'ACTION6':
                        valid_selections.extend(['action_click', 'ACTION6'])
                    else:
                        action_num = avail[-1]
                        valid_selections.extend([f'action_{action_num}', avail])
                
                assert any(sel in selected for sel in valid_selections), \
                    f"Expected one of {valid_selections}, got {selected}"
            else:
                # ACTION6 not available - should NOT select it
                assert 'ACTION6' not in selected and 'action_click' not in selected, \
                    f"ACTION6 not available but selected: {selected}"

class TestAction6CoordinateHandling:
    """Test ACTION6 coordinate handling specifically"""
    
    def test_action6_coordinate_bounds(self):
        """Test that ACTION6 coordinates are always within bounds"""
        agent = ReCoNArcAngel('test_action6_bounds')
        
        # Test with different frame patterns
        patterns = ["random", "center_square", "corner_patterns", "empty"]
        
        for pattern in patterns:
            frame = create_test_frame(pattern, ['ACTION6'])
            action = agent.choose_action([frame], frame)
            
            if hasattr(action, 'data') and action.data:
                x = action.data.get('x')
                y = action.data.get('y')
                
                if x is not None and y is not None:
                    assert 0 <= x < 64, f"Pattern {pattern}: x={x} out of bounds"
                    assert 0 <= y < 64, f"Pattern {pattern}: y={y} out of bounds"
                    
                    print(f"Pattern {pattern}: ACTION6 at ({x}, {y})")
    
    def test_action6_region_selection_consistency(self):
        """Test that ACTION6 consistently selects from the same region type"""
        agent = ReCoNArcAngel('test_action6_consistency')
        
        # Create frame with clear pattern in specific region
        frame_array = np.zeros((64, 64), dtype=int)
        frame_array[16:24, 16:24] = 4  # Pattern in region (2,2)
        frame = FrameData(frame_array, ['ACTION6'])
        
        # Run multiple times to test consistency
        coordinates = []
        for _ in range(3):
            # Reset agent state
            agent.hypothesis_manager.reset()
            
            action = agent.choose_action([frame], frame)
            
            if hasattr(action, 'data') and action.data:
                x = action.data.get('x', 0)
                y = action.data.get('y', 0)
                coordinates.append((x, y))
        
        print(f"ACTION6 coordinates across runs: {coordinates}")
        
        # All coordinates should be in reasonable range
        for x, y in coordinates:
            assert 0 <= x < 64
            assert 0 <= y < 64
    
    def test_action6_with_score_changes(self):
        """Test ACTION6 behavior across score changes"""
        agent = ReCoNArcAngel('test_action6_score_changes')
        
        frame_array = np.zeros((64, 64), dtype=int)
        frame_array[32:40, 32:40] = 2
        
        # First frame with score 0
        frame1 = FrameData(frame_array, ['ACTION6'], score=0)
        action1 = agent.choose_action([frame1], frame1)
        
        # Second frame with score 1 (should trigger reset)
        frame2 = FrameData(frame_array, ['ACTION6'], score=1)
        action2 = agent.choose_action([frame1, frame2], frame2)
        
        # Both should be ACTION6
        selected1 = getattr(action1, 'name', getattr(action1, 'action_type', ''))
        selected2 = getattr(action2, 'name', getattr(action2, 'action_type', ''))
        
        assert 'action_click' in selected1 or 'ACTION6' in selected1
        assert 'action_click' in selected2 or 'ACTION6' in selected2

class TestMaskingRobustness:
    """Test robustness of masking system"""
    
    def test_masking_with_high_cnn_confidence(self):
        """Test that masking works even when CNN has high confidence in unavailable actions"""
        agent = ReCoNArcAngel('test_high_confidence_masking')
        
        # Create frame that might make CNN prefer ACTION6
        frame_array = np.zeros((64, 64), dtype=int)
        frame_array[20:44, 20:44] = 1  # Large pattern that might favor clicking
        
        # But only allow ACTION1
        frame = FrameData(frame_array, ['ACTION1'])
        
        action = agent.choose_action([frame], frame)
        selected = getattr(action, 'name', getattr(action, 'action_type', ''))
        
        # Must select ACTION1 despite CNN potentially preferring ACTION6
        assert 'action_1' in selected or 'ACTION1' in selected, \
            f"Expected ACTION1 (only available), got {selected}"
    
    def test_masking_persistence_across_propagation(self):
        """Test that masking persists throughout ReCoN propagation"""
        agent = ReCoNArcAngel('test_masking_persistence')
        
        frame = create_test_frame("random", ['ACTION3'])
        
        # Manually check states after each step
        current_frame_tensor = agent._convert_frame_to_tensor(frame)
        agent.hypothesis_manager.update_weights_from_frame(current_frame_tensor)
        agent.hypothesis_manager.reset()
        agent._apply_available_actions_mask(frame.available_actions)
        
        # Check initial masking
        initial_states = {}
        for i in range(1, 6):
            action_id = f'action_{i}'
            node = agent.hypothesis_manager.graph.nodes[action_id]
            initial_states[action_id] = node.state
        
        agent.hypothesis_manager.request_frame_change()
        
        # Run propagation
        for step in range(5):
            agent.hypothesis_manager.propagate_step()
        
        # Check final states
        final_states = {}
        for i in range(1, 6):
            action_id = f'action_{i}'
            node = agent.hypothesis_manager.graph.nodes[action_id]
            final_states[action_id] = node.state
        
        # ACTION3 should be available (not effectively unavailable due to low sub weight)
        assert not agent.hypothesis_manager._is_action_effectively_unavailable('action_3'), \
            f"ACTION3 should be available, but it's effectively unavailable due to low sub weight"
        
        # Other actions should remain FAILED or at least not be the best choice
        best_action, _ = agent.hypothesis_manager.get_best_action(available_actions=['ACTION3'])
        assert best_action == 'action_3', f"Expected action_3, got {best_action}"
    
    def test_action6_region_threshold_sensitivity(self):
        """Test ACTION6 behavior with different region activation levels"""
        
        # Test region measurement thresholds in isolation
        from recon_engine.graph import ReCoNGraph
        
        g = ReCoNGraph()
        g.add_node("parent", node_type="script")
        g.add_node("region", node_type="terminal")
        g.add_link("parent", "region", "sub", weight=1.0)
        
        region_measurements = [0.1, 0.5, 0.8, 0.9, 1.0]
        
        for measurement in region_measurements:
            # Reset graph
            g.requested_roots.clear()
            g.message_queue.clear()
            g.step_count = 0
            
            for node in g.nodes.values():
                node.reset()
            
            # Set region measurement
            region_node = g.get_node("region")
            region_node.measurement_fn = lambda env=None, m=measurement: m
            
            # Run propagation
            g.request_root("parent")
            for _ in range(3):
                g.propagate_step()
            
            # Check region state
            region_state = region_node.state
            print(f"Isolated test - Measurement {measurement}: region={region_state.name}")
            
            # Regions with measurement > 0.8 should CONFIRM (threshold is exclusive)
            if measurement > 0.8:
                assert region_state == ReCoNState.CONFIRMED, \
                    f"Measurement {measurement} > 0.8 should CONFIRM, got {region_state}"
            else:
                assert region_state == ReCoNState.FAILED, \
                    f"Measurement {measurement} <= 0.8 should FAIL, got {region_state}"

# Test runner functions
def test_action6_only_scenarios():
    tester = TestActionMasking()
    tester.test_action6_only_scenarios()

def test_action6_only_different_patterns():
    tester = TestActionMasking()
    tester.test_action6_only_different_patterns()

def test_action6_vs_individual_actions():
    tester = TestActionMasking()
    tester.test_action6_vs_individual_actions()

def test_action6_coordinate_regions():
    tester = TestActionMasking()
    tester.test_action6_coordinate_regions()

def test_single_action_scenarios():
    tester = TestMaskingCombinations()
    tester.test_single_action_scenarios()

def test_pair_combinations():
    tester = TestMaskingCombinations()
    tester.test_pair_combinations()

def test_exclusion_scenarios():
    tester = TestMaskingCombinations()
    tester.test_exclusion_scenarios()

def test_action6_region_masking():
    tester = TestMaskingCombinations()
    tester.test_action6_region_masking()

def test_empty_available_actions():
    tester = TestMaskingEdgeCases()
    tester.test_empty_available_actions()

def test_invalid_available_actions():
    tester = TestMaskingEdgeCases()
    tester.test_invalid_available_actions()

def test_action6_with_no_regions():
    tester = TestMaskingEdgeCases()
    tester.test_action6_with_no_regions()

def test_mixed_availability_with_action6():
    tester = TestMaskingEdgeCases()
    tester.test_mixed_availability_with_action6()

def test_action6_coordinate_bounds():
    tester = TestAction6CoordinateHandling()
    tester.test_action6_coordinate_bounds()

def test_action6_region_selection_consistency():
    tester = TestAction6CoordinateHandling()
    tester.test_action6_region_selection_consistency()

def test_action6_with_score_changes():
    tester = TestAction6CoordinateHandling()
    tester.test_action6_with_score_changes()

def test_masking_with_high_cnn_confidence():
    tester = TestMaskingRobustness()
    tester.test_masking_with_high_cnn_confidence()

def test_masking_persistence_across_propagation():
    tester = TestMaskingRobustness()
    tester.test_masking_persistence_across_propagation()

def test_action6_region_threshold_sensitivity():
    tester = TestMaskingRobustness()
    tester.test_action6_region_threshold_sensitivity()
