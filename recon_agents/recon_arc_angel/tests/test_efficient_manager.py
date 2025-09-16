"""
Test the efficient hierarchy manager that combines REFINED_PLAN with BlindSquirrel
"""
import pytest
import sys
import os
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_agents.recon_arc_angel.efficient_hierarchy_manager import EfficientHierarchicalHypothesisManager

def test_efficient_manager_build_speed():
    """Test that efficient manager builds quickly"""
    import time
    
    start = time.time()
    manager = EfficientHierarchicalHypothesisManager(cnn_threshold=0.1, max_objects=50)
    manager.build_structure()
    build_time = time.time() - start
    
    print(f"✅ Build time: {build_time:.3f}s")
    assert build_time < 5.0, f"Build too slow: {build_time:.3f}s"
    
    stats = manager.get_stats()
    print(f"✅ Nodes: {stats['total_nodes']}")
    print(f"✅ Efficiency gain: {stats['efficiency_gain']}")

def test_efficient_manager_frame_update_speed():
    """Test that frame updates are fast"""
    import time
    
    manager = EfficientHierarchicalHypothesisManager(cnn_threshold=0.1, max_objects=20)
    manager.build_structure()
    
    # Create test frame
    frame = torch.zeros(16, 64, 64)
    frame[1, 10:20, 10:20] = 1.0  # Object 1
    frame[2, 30:35, 30:40] = 1.0  # Object 2
    
    # Move to GPU if available
    if torch.cuda.is_available():
        frame = frame.cuda()
    
    # Test update speed
    start = time.time()
    manager.update_weights_from_cnn(frame)
    update_time = time.time() - start
    
    print(f"✅ Frame update time: {update_time:.3f}s")
    assert update_time < 1.0, f"Update too slow: {update_time:.3f}s"
    
    print(f"✅ Objects detected: {len(manager.current_objects)}")
    print(f"✅ Final node count: {len(manager.graph.nodes)}")

def test_efficient_manager_action_selection():
    """Test action selection with efficient manager"""
    manager = EfficientHierarchicalHypothesisManager(cnn_threshold=0.1, max_objects=20)
    manager.build_structure()
    
    # Create frame with clear objects
    frame = torch.zeros(16, 64, 64)
    frame[3, 25:35, 25:35] = 1.0  # Clear object
    
    if torch.cuda.is_available():
        frame = frame.cuda()
    
    # Update and propagate
    manager.update_weights_from_cnn(frame)
    manager.reset()
    manager.apply_availability_mask(["ACTION1", "ACTION6"])
    manager.request_frame_change()
    
    # Run propagation
    for _ in range(8):  # Enough steps for script→terminal→script flow
        manager.propagate_step()
    
    # Get best action
    best_action, best_coords = manager.get_best_action_with_object_coordinates(["ACTION1", "ACTION6"])
    
    print(f"✅ Selected action: {best_action}")
    if best_coords:
        print(f"✅ Coordinates: {best_coords}")
    
    # Should select a valid action
    assert best_action in ["action_1", "action_click", None]
    
    # If ACTION6, should have valid coordinates
    if best_action == "action_click" and best_coords:
        y, x = best_coords
        assert 0 <= y < 64 and 0 <= x < 64

def test_efficient_vs_fixed_hierarchy_comparison():
    """Compare efficient vs fixed hierarchy approach"""
    
    # Efficient approach
    efficient_manager = EfficientHierarchicalHypothesisManager(cnn_threshold=0.1, max_objects=30)
    efficient_manager.build_structure()
    
    # Test with various frame complexities
    test_frames = [
        ("simple", torch.zeros(16, 64, 64)),
        ("medium", None),
        ("complex", None)
    ]
    
    # Medium frame
    medium_frame = torch.zeros(16, 64, 64)
    for i in range(5):
        y, x = np.random.randint(5, 59, 2)
        medium_frame[i+1, y:y+4, x:x+4] = 1.0
    test_frames[1] = ("medium", medium_frame)
    
    # Complex frame (but not random noise)
    complex_frame = torch.zeros(16, 64, 64)
    for i in range(10):
        y, x = np.random.randint(2, 62, 2)
        complex_frame[(i%8)+1, y:y+2, x:x+2] = 1.0
    test_frames[2] = ("complex", complex_frame)
    
    print("Hierarchy comparison:")
    print("Fixed REFINED_PLAN approach: 266,320 nodes")
    print()
    
    for name, frame in test_frames:
        if torch.cuda.is_available():
            frame = frame.cuda()
        
        efficient_manager.update_weights_from_cnn(frame)
        
        stats = efficient_manager.get_stats()
        node_count = stats["total_nodes"]
        object_count = stats["current_objects"]
        reduction = 266320 / node_count
        
        print(f"Frame {name}:")
        print(f"  Efficient approach: {node_count} nodes ({object_count} objects)")
        print(f"  Reduction factor: {reduction:.0f}x smaller")
        print(f"  Memory efficiency: {(266320 - node_count) / 266320 * 100:.1f}% less memory")
        
        # Should be dramatically more efficient
        assert node_count < 1000, f"Still too many nodes for {name}: {node_count}"
        assert reduction > 100, f"Not enough reduction for {name}: {reduction:.0f}x"
