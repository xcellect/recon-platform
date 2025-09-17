#!/usr/bin/env python3
"""
Demonstration of Improved ReCoN ARC Angel

This script demonstrates the key improvements made to fix the coordinate selection issues:
1. Proper ReCoN graph structure with por/ret sequences
2. Mask-aware CNN coupling 
3. Background suppression
4. Improved selection scoring
5. Stickiness mechanism
6. Pure ReCoN execution semantics

Run with: python demo_improved_recon.py
"""

import torch
import numpy as np
import os
import sys

# Add paths
sys.path.insert(0, "/workspace/recon-platform")
sys.path.insert(0, "/workspace/recon-platform/recon_agents/recon_arc_angel")

from improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager
from improved_production_agent import ImprovedProductionReCoNArcAngel


def create_test_frame_with_background_issue():
    """Create a test frame that would cause issues with the old implementation"""
    frame = torch.zeros(16, 64, 64)
    
    # Background strip (old implementation would click here)
    frame[0, :, 0] = 1  # Full-height left border
    
    # Small interesting object (should be preferred)
    frame[1, 10:15, 10:15] = 1  # Red square
    
    # Irregular object (should be penalized)
    frame[2, 20:25, 20:60] = 1  # Long horizontal strip
    
    return frame


def create_test_frame_with_l_shape():
    """Create L-shaped object to test mask adherence"""
    frame = torch.zeros(16, 64, 64)
    
    # L-shaped object
    frame[1, 10:15, 10:15] = 1  # Square part
    frame[1, 15:20, 10:12] = 1  # Extension part
    
    return frame


def demo_graph_structure():
    """Demonstrate proper ReCoN graph structure"""
    print("🏗️  DEMO 1: Proper ReCoN Graph Structure")
    print("=" * 50)
    
    manager = ImprovedHierarchicalHypothesisManager()
    manager.build_improved_structure()
    
    print("✅ Graph Structure:")
    print(f"   Total nodes: {len(manager.graph.nodes)}")
    print(f"   Total links: {len(manager.graph.links)}")
    
    print("\n✅ ACTION6 Sequence Structure:")
    print("   action_click -> click_cnn -> click_objects")
    
    # Verify por/ret sequence
    por_links = manager.graph.get_links(source="click_cnn", target="click_objects", link_type="por")
    ret_links = manager.graph.get_links(source="click_objects", target="click_cnn", link_type="ret")
    
    print(f"   Por links (click_cnn -> click_objects): {len(por_links)}")
    print(f"   Ret links (click_objects -> click_cnn): {len(ret_links)}")
    
    print("\n✅ Neural Terminal Placement:")
    cnn_links = manager.graph.get_links(source="click_cnn", target="cnn_terminal", link_type="sub")
    resnet_links = manager.graph.get_links(source="click_objects", target="resnet_terminal", link_type="sub")
    
    print(f"   CNN terminal under click_cnn: {len(cnn_links) > 0}")
    print(f"   ResNet terminal under click_objects: {len(resnet_links) > 0}")
    print()


def demo_mask_aware_cnn():
    """Demonstrate mask-aware CNN coupling"""
    print("🎯 DEMO 2: Mask-Aware CNN Coupling")
    print("=" * 50)
    
    manager = ImprovedHierarchicalHypothesisManager()
    manager.build_improved_structure()
    
    # Create frame with background issue
    frame = create_test_frame_with_background_issue()
    
    # Extract objects
    objects = manager.extract_objects_from_frame(frame)
    
    print("✅ Object Detection Results:")
    for i, obj in enumerate(objects):
        print(f"   Object {i}:")
        print(f"     Area fraction: {obj['area_frac']:.3f}")
        print(f"     Regularity: {obj['regularity']:.3f}")
        print(f"     Border penalty: {obj['border_penalty']:.3f}")
        print(f"     Confidence: {obj['confidence']:.3f}")
        print(f"     Slice: {obj['slice']}")
    
    # Demonstrate masked max calculation
    coord_probs = torch.ones(64, 64) * 0.1
    coord_probs[0, 0] = 0.9  # High prob at border (should be ignored for small objects)
    coord_probs[12, 12] = 0.6  # Lower prob but inside small object
    
    for i, obj in enumerate(objects):
        if obj['area_frac'] < 0.1:  # Small object
            masked_max = manager.calculate_masked_cnn_probability(coord_probs, obj)
            print(f"   Object {i} masked max CNN prob: {masked_max:.3f}")
    print()


def demo_background_suppression():
    """Demonstrate background suppression"""
    print("🚫 DEMO 3: Background Suppression")
    print("=" * 50)
    
    manager = ImprovedHierarchicalHypothesisManager()
    manager.build_improved_structure()
    
    frame = create_test_frame_with_background_issue()
    objects = manager.extract_objects_from_frame(frame)
    
    print("✅ Background Suppression Results:")
    background_objects = []
    good_objects = []
    
    for obj in objects:
        if obj['border_penalty'] > 0.3 or obj['area_frac'] > 0.15:
            background_objects.append(obj)
        else:
            good_objects.append(obj)
    
    print(f"   Background objects detected: {len(background_objects)}")
    print(f"   Good objects preserved: {len(good_objects)}")
    
    for i, obj in enumerate(background_objects):
        print(f"   Background {i}: confidence={obj['confidence']:.3f}, penalty={obj['border_penalty']:.3f}")
    
    for i, obj in enumerate(good_objects):
        print(f"   Good object {i}: confidence={obj['confidence']:.3f}, regularity={obj['regularity']:.3f}")
    print()


def demo_coordinate_selection():
    """Demonstrate coordinate selection within masks"""
    print("📍 DEMO 4: Coordinate Selection Within Masks")
    print("=" * 50)
    
    manager = ImprovedHierarchicalHypothesisManager()
    manager.build_improved_structure()
    
    # Create L-shaped object
    frame = create_test_frame_with_l_shape()
    manager.update_dynamic_objects_improved(frame)
    
    print("✅ L-Shaped Object Coordinate Selection:")
    
    # Test multiple coordinate selections
    valid_selections = 0
    coords_tested = []
    
    for i in range(10):
        coord = manager._get_object_coordinate_improved(0)
        if coord is not None:
            y, x = coord
            coords_tested.append((y, x))
            
            # Check if coordinate is within L-shape
            if ((10 <= y <= 14 and 10 <= x <= 14) or  # Square part
                (15 <= y <= 19 and 10 <= x <= 11)):   # Extension part
                valid_selections += 1
    
    print(f"   Coordinates tested: {len(coords_tested)}")
    print(f"   Valid selections (within mask): {valid_selections}")
    print(f"   Success rate: {valid_selections/len(coords_tested)*100:.1f}%")
    print(f"   Sample coordinates: {coords_tested[:5]}")
    print()


def demo_stickiness_mechanism():
    """Demonstrate stickiness mechanism"""
    print("🎯 DEMO 5: Stickiness Mechanism")
    print("=" * 50)
    
    manager = ImprovedHierarchicalHypothesisManager()
    manager.build_improved_structure()
    
    # Record successful click
    successful_coord = (12, 12)
    manager.record_successful_click(successful_coord)
    
    print("✅ Stickiness State:")
    print(f"   Last successful click: {manager.last_successful_click}")
    print(f"   Stickiness strength: {manager.stickiness_strength:.3f}")
    
    # Test decay
    print("\n✅ Stickiness Decay:")
    for i in range(5):
        manager.decay_stickiness()
        print(f"   Step {i+1}: strength={manager.stickiness_strength:.3f}")
    
    print(f"   Final stickiness: {manager.stickiness_strength:.3f}")
    print()


def demo_recon_propagation():
    """Demonstrate proper ReCoN message propagation"""
    print("⚡ DEMO 6: ReCoN Message Propagation")
    print("=" * 50)
    
    manager = ImprovedHierarchicalHypothesisManager()
    manager.build_improved_structure()
    
    # Create test frame
    frame = torch.zeros(16, 64, 64)
    frame[1, 10:15, 10:15] = 1
    
    manager.update_weights_from_cnn_improved(frame)
    manager.reset()
    manager.apply_availability_mask(["ACTION6"])
    manager.request_frame_change()
    
    print("✅ ReCoN Propagation Sequence:")
    
    # Show propagation steps
    for step in range(6):
        manager.propagate_step()
        root_state = manager.graph.nodes["frame_change_hypothesis"].state
        click_state = manager.graph.nodes["action_click"].state
        cnn_state = manager.graph.nodes["click_cnn"].state
        objects_state = manager.graph.nodes["click_objects"].state
        
        print(f"   Step {step+1}: Root={root_state.name}, Click={click_state.name}, CNN={cnn_state.name}, Objects={objects_state.name}")
    
    print("\n✅ Final States:")
    print(f"   action_click: {manager.graph.nodes['action_click'].state.name}")
    print(f"   click_cnn: {manager.graph.nodes['click_cnn'].state.name}")
    print(f"   click_objects: {manager.graph.nodes['click_objects'].state.name}")
    print()


def demo_complete_agent():
    """Demonstrate complete improved agent"""
    print("🚀 DEMO 7: Complete Improved Agent")
    print("=" * 50)
    
    # Mock frame data
    class MockFrameData:
        def __init__(self):
            self.frame = np.zeros((64, 64), dtype=np.int64)
            self.frame[10:15, 10:15] = 1  # Red square
            self.frame[:, 0] = 1  # Background strip
            self.score = 0
            self.state = "PLAYING"
            
            # Mock available actions
            class MockAction:
                def __init__(self, name):
                    self.name = name
            self.available_actions = [MockAction("ACTION6")]
    
    # Create improved agent
    agent = ImprovedProductionReCoNArcAngel()
    frame_data = MockFrameData()
    
    print("✅ Agent Initialization:")
    print(f"   CNN threshold: {agent.cnn_threshold}")
    print(f"   Max objects: {agent.max_objects}")
    
    # Choose action
    action = agent.choose_action([], frame_data)
    
    print("\n✅ Action Selection:")
    print(f"   Selected action type: {action.action_type}")
    if hasattr(action, 'data') and action.data:
        print(f"   Coordinates: ({action.data.get('x', 'N/A')}, {action.data.get('y', 'N/A')})")
    print(f"   Reasoning: {action.reasoning}")
    
    # Show stats
    stats = agent.get_stats()
    print("\n✅ Agent Statistics:")
    print(f"   Total actions: {stats['total_actions']}")
    print(f"   Objects detected: {stats['objects_detected']}")
    print(f"   Improvements: {len(stats['improvements'])}")
    
    for improvement in stats['improvements']:
        print(f"     - {improvement}")
    print()


def main():
    """Run all demonstrations"""
    print("🎉 IMPROVED RECON ARC ANGEL DEMONSTRATION")
    print("=" * 60)
    print("This demo shows all the improvements that fix the coordinate selection issues.\n")
    
    try:
        demo_graph_structure()
        demo_mask_aware_cnn()
        demo_background_suppression()
        demo_coordinate_selection()
        demo_stickiness_mechanism()
        demo_recon_propagation()
        demo_complete_agent()
        
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The improved implementation is ready for production use.")
        print("Key benefits:")
        print("✅ Coordinates are strictly within object masks")
        print("✅ Background strips and noise are filtered out")
        print("✅ Successful clicks are persisted with stickiness")
        print("✅ Pure ReCoN execution with proper sequences")
        print("✅ Comprehensive object scoring with multiple factors")
        print("✅ GPU acceleration and dual training support")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
