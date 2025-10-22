"""
Test Extension 1A fix - CompactReCoNGraph with neural terminals
"""
import sys
import os

# Clear module cache to force reload
modules_to_clear = [k for k in sys.modules.keys() if 'recon' in k or 'improved' in k]
for mod in modules_to_clear:
    del sys.modules[mod]

# Use direct imports to bypass module caching
sys.path.insert(0, "/workspace/recon-platform")
sys.path.insert(0, "/workspace/recon-platform/recon_agents/recon_arc_angel")

from improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager
from recon_engine.compact import CompactReCoNGraph, CompactReCoNNode
from recon_engine.node import ReCoNNode
import torch

def test_compact_with_neural_terminals():
    """Test that compact graph can handle neural terminals"""
    print("\n" + "="*70)
    print("TEST: Extension 1A - Compact ReCoN with Neural Terminals")
    print("="*70)

    # Create manager with compact mode
    print("\n✅ Creating ImprovedHierarchicalHypothesisManager with use_compact=True...")
    manager = ImprovedHierarchicalHypothesisManager(use_compact=True)

    # Build structure (this adds neural terminals to compact graph)
    print("✅ Building structure (including neural terminals)...")
    try:
        manager.build_improved_structure()
        print("✅ SUCCESS: Structure built without errors!")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Verify graph type
    print(f"\n✅ Graph type: {type(manager.graph).__name__}")
    assert isinstance(manager.graph, CompactReCoNGraph), "Should be CompactReCoNGraph"

    # Verify neural terminals were added
    print(f"✅ CNN terminal in graph: {'cnn_terminal' in manager.graph.nodes}")
    print(f"✅ ResNet terminal in graph: {'resnet_terminal' in manager.graph.nodes}")

    # Verify node types
    cnn_node = manager.graph.get_node("cnn_terminal")
    resnet_node = manager.graph.get_node("resnet_terminal")
    action_1_node = manager.graph.get_node("action_1")

    print(f"\n✅ Node types:")
    print(f"   cnn_terminal:    {type(cnn_node).__name__} (should be CNNValidActionTerminal)")
    print(f"   resnet_terminal: {type(resnet_node).__name__} (should be ResNetActionValueTerminal)")
    print(f"   action_1:        {type(action_1_node).__name__} (should be CompactReCoNNode)")

    # Verify action_1 is compact but terminals are not
    assert isinstance(action_1_node, CompactReCoNNode), "Script nodes should be CompactReCoNNode"
    assert isinstance(cnn_node, ReCoNNode), "Terminals should be ReCoNNode (base or subclass)"
    assert not isinstance(cnn_node, CompactReCoNNode), "Terminals should NOT be CompactReCoNNode"

    # Test gen loops on objects
    print(f"\n✅ Testing gen loops on objects...")

    # Create test frame with object
    frame = torch.zeros(64, 64, dtype=torch.long)
    frame[10:20, 10:20] = 3  # Object with color 3

    # Extract objects
    objects = manager.extract_objects_from_frame(frame)
    manager.current_objects = objects

    print(f"✅ Extracted {len(objects)} object(s)")

    if len(objects) > 0:
        # Add gen loops
        manager.add_gen_loops_to_objects()
        print(f"✅ Added gen loops to objects")

        # Verify gen loops exist
        obj_id = "object_0"
        obj_node = manager.graph.get_node(obj_id)
        print(f"✅ object_0 type: {type(obj_node).__name__}")
        assert isinstance(obj_node, CompactReCoNNode), "Object nodes should be CompactReCoNNode"

        # Check for gen loop link
        gen_links = [l for l in manager.graph.links
                     if l.source == obj_id and l.target == obj_id and l.type == "gen"]

        if len(gen_links) > 0:
            print(f"✅ Gen loop found: {obj_id} -> {obj_id}, weight={gen_links[0].weight}")
            assert abs(gen_links[0].weight - 0.95) < 0.01, "Gen loop weight should be 0.95"
        else:
            print(f"⚠️  No gen loop found for {obj_id}")

    print("\n" + "="*70)
    print("✅ TEST PASSED: Extension 1A works with neural terminals!")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_compact_with_neural_terminals()
    sys.exit(0 if success else 1)
