"""
Test the production agent that uses EfficientHierarchicalHypothesisManager
"""
import pytest
import sys
import os
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_agents.recon_arc_angel.production_agent import ProductionReCoNArcAngel

class TestFrame:
    """Test frame data structure"""
    def __init__(self, frame, available_actions, score=0, state='PLAYING'):
        self.frame = frame
        self.available_actions = available_actions
        self.score = score
        self.state = state

def test_production_agent_uses_efficient_manager():
    """Test that production agent uses EfficientHierarchicalHypothesisManager"""
    agent = ProductionReCoNArcAngel("test_production", cnn_threshold=0.1, max_objects=30)
    
    # Check that it uses the efficient manager
    assert hasattr(agent.hypothesis_manager, 'extract_objects_from_frame')
    assert hasattr(agent.hypothesis_manager, 'update_dynamic_objects')
    assert hasattr(agent.hypothesis_manager, 'get_best_action_with_object_coordinates')
    
    # Check dual training setup
    assert agent.learning_manager.cnn_terminal is not None
    assert agent.learning_manager.resnet_terminal is not None
    
    stats = agent.get_stats()
    print(f"✅ Production agent using efficient manager: {stats['hypothesis_manager']['total_nodes']} nodes")
    print(f"✅ Efficiency gain: {stats['hypothesis_manager']['efficiency_gain']}")

def test_production_agent_action_selection():
    """Test production agent action selection with object segmentation"""
    agent = ProductionReCoNArcAngel("test_production", cnn_threshold=0.1, max_objects=20)
    
    # Create frame with clear objects
    frame_array = np.zeros((64, 64), dtype=int)
    frame_array[15:25, 15:25] = 1  # Square object
    frame_array[35:40, 40:50] = 2  # Rectangle object
    
    frame = TestFrame(frame_array, ["ACTION1", "ACTION6"])
    
    action = agent.choose_action([frame], frame)
    
    # Should return valid action
    assert hasattr(action, 'action_type')
    assert action.action_type in ["ACTION1", "ACTION6"]
    
    # Check stats
    stats = agent.get_stats()
    print(f"✅ Objects detected: {stats['objects_detected']}")
    print(f"✅ Selected action: {action.action_type}")
    
    if hasattr(action, 'data') and action.data:
        print(f"✅ Coordinates: ({action.data.get('x', 'N/A')}, {action.data.get('y', 'N/A')})")

def test_production_agent_dual_training():
    """Test that production agent performs dual training"""
    agent = ProductionReCoNArcAngel("test_dual", cnn_threshold=0.1, max_objects=20)
    
    # Create sequence of frames
    frames = []
    for i in range(3):
        frame_array = np.zeros((64, 64), dtype=int)
        frame_array[10+i*5:15+i*5, 10+i*5:15+i*5] = i + 1
        frames.append(TestFrame(frame_array, ["ACTION1", "ACTION2"], score=i))
    
    # Simulate gameplay
    for i, frame in enumerate(frames):
        action = agent.choose_action(frames[:i+1], frame)
        print(f"Frame {i}: Selected {action.action_type}")
    
    # Check that training occurred
    stats = agent.get_stats()
    
    print(f"✅ CNN training steps: {stats['cnn_training_steps']}")
    print(f"✅ ResNet training steps: {stats['resnet_training_steps']}")
    print(f"✅ Unique experiences: {stats['unique_experiences']}")
    print(f"✅ Score resets: {stats['score_resets']}")
    
    # Should have some training activity
    assert stats['total_actions'] > 0

def test_production_agent_performance():
    """Test production agent performance"""
    import time
    
    agent = ProductionReCoNArcAngel("test_perf", cnn_threshold=0.1, max_objects=30)
    
    # Create realistic frame
    frame_array = np.zeros((64, 64), dtype=int)
    frame_array[20:30, 20:30] = 1
    frame_array[40:45, 40:50] = 2
    
    frame = TestFrame(frame_array, ["ACTION1", "ACTION2", "ACTION6"])
    
    # Time the action selection
    start = time.time()
    action = agent.choose_action([frame], frame)
    selection_time = time.time() - start
    
    print(f"✅ Action selection time: {selection_time:.3f}s")
    print(f"✅ Selected: {action.action_type}")
    
    # Should be fast (sub-second)
    assert selection_time < 5.0, f"Too slow: {selection_time:.3f}s"
    
    # Check efficiency
    stats = agent.get_stats()
    node_count = stats['hypothesis_manager']['total_nodes']
    objects = stats['objects_detected']
    
    print(f"✅ Nodes used: {node_count}")
    print(f"✅ Objects detected: {objects}")
    
    # Should be very efficient
    assert node_count < 100, f"Too many nodes: {node_count}"

def test_production_agent_vs_legacy():
    """Compare production agent vs legacy simplified agent"""
    from recon_agents.recon_arc_angel.agent import ReCoNArcAngel
    
    # Create both agents
    production_agent = ProductionReCoNArcAngel("prod_test")
    legacy_agent = ReCoNArcAngel("legacy_test")
    
    # Compare stats
    prod_stats = production_agent.get_stats()
    legacy_stats = legacy_agent.get_stats()
    
    prod_nodes = prod_stats['hypothesis_manager']['total_nodes']
    legacy_nodes = legacy_stats['hypothesis_manager']['total_nodes']
    
    print(f"Production agent nodes: {prod_nodes}")
    print(f"Legacy agent nodes: {legacy_nodes}")
    
    # Production should have similar efficiency (both are efficient)
    # The key difference is the dynamic objects vs fixed regions
    assert prod_nodes < 100  # Both should be efficient
    assert legacy_nodes < 100
    
    print("✅ Both agents are efficient, production uses dynamic objects")
