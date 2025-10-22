"""
Tests for Global Workspace Theory Extension

Tests the GWT broadcast mechanism that implements winner-take-all
lateral inhibition for clearer action selection.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recon_agents.recon_arc_angel.global_workspace import GlobalWorkspace
from recon_agents.recon_arc_angel.improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager
from recon_agents.recon_arc_angel.improved_production_agent import ImprovedProductionReCoNArcAngel


class TestGlobalWorkspace:
    """Test suite for Global Workspace Theory extension"""

    def test_global_workspace_creation(self):
        """Test that GlobalWorkspace can be created with default parameters"""
        gw = GlobalWorkspace()
        assert gw.broadcast_strength == 0.3
        assert gw.winner_boost == 0.2
        assert gw.stats['total_broadcasts'] == 0

    def test_global_workspace_custom_parameters(self):
        """Test GlobalWorkspace with custom parameters"""
        gw = GlobalWorkspace(broadcast_strength=0.5, winner_boost=0.3)
        assert gw.broadcast_strength == 0.5
        assert gw.winner_boost == 0.3

    def test_broadcast_winner_suppresses_losers(self):
        """Test that winner suppresses competitor activations"""
        # Create hierarchy manager with GWT enabled
        manager = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=True
        )
        manager.build_improved_structure()

        # Create test candidates with known scores
        candidates = [
            ("action_1", 0.8, None, None),  # Winner
            ("action_2", 0.6, None, None),  # Loser 1
            ("action_3", 0.5, None, None),  # Loser 2
        ]

        # Set initial activations for test
        for action_id, score, _, _ in candidates:
            if action_id in manager.graph.nodes:
                node = manager.graph.nodes[action_id]
                node.activation = score

        # Broadcast winner (index 0)
        manager.global_workspace.broadcast_winner(0, candidates, manager.graph)

        # Verify winner was boosted
        action_1_activation = float(manager.graph.nodes["action_1"].activation)
        assert action_1_activation >= 0.8, f"Winner activation should be >= 0.8, got {action_1_activation}"

        # Verify losers were suppressed
        action_2_activation = float(manager.graph.nodes["action_2"].activation)
        action_3_activation = float(manager.graph.nodes["action_3"].activation)

        assert action_2_activation < 0.6, f"Loser 1 should be suppressed below 0.6, got {action_2_activation}"
        assert action_3_activation < 0.5, f"Loser 2 should be suppressed below 0.5, got {action_3_activation}"

        print(f"✅ Global Workspace broadcast working:")
        print(f"  Winner activation: {action_1_activation:.3f} (was 0.800)")
        print(f"  Loser 1 activation: {action_2_activation:.3f} (was 0.600)")
        print(f"  Loser 2 activation: {action_3_activation:.3f} (was 0.500)")

    def test_broadcast_updates_statistics(self):
        """Test that broadcast updates GWT statistics"""
        manager = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=True
        )
        manager.build_improved_structure()

        candidates = [
            ("action_1", 0.9, None, None),
            ("action_2", 0.7, None, None),
        ]

        # Set activations
        for action_id, score, _, _ in candidates:
            if action_id in manager.graph.nodes:
                manager.graph.nodes[action_id].activation = score

        # Initial stats
        assert manager.global_workspace.stats['total_broadcasts'] == 0

        # Broadcast
        manager.global_workspace.broadcast_winner(0, candidates, manager.graph)

        # Verify stats updated
        assert manager.global_workspace.stats['total_broadcasts'] == 1
        assert manager.global_workspace.stats['avg_winner_boost'] > 0
        assert manager.global_workspace.stats['last_winner_score'] == 0.9
        assert manager.global_workspace.stats['last_runnerup_score'] == 0.7

        print(f"✅ Statistics tracking working:")
        print(f"  Total broadcasts: {manager.global_workspace.stats['total_broadcasts']}")
        print(f"  Avg winner boost: {manager.global_workspace.stats['avg_winner_boost']:.4f}")
        print(f"  Avg loser suppression: {manager.global_workspace.stats['avg_loser_suppression']:.4f}")
        print(f"  Decision confidence: {manager.global_workspace.stats['avg_decision_confidence']:.3f}")

    def test_hierarchy_manager_integration(self):
        """Test that hierarchy manager properly integrates GWT"""
        # With GWT enabled
        manager_with_gw = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=True
        )
        manager_with_gw.build_improved_structure()

        assert manager_with_gw.enable_global_workspace is True
        assert manager_with_gw.global_workspace is not None

        # Without GWT enabled
        manager_without_gw = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=False
        )
        manager_without_gw.build_improved_structure()

        assert manager_without_gw.enable_global_workspace is False
        assert manager_without_gw.global_workspace is None

        print(f"✅ Hierarchy manager integration working")

    def test_production_agent_integration(self):
        """Test that production agent properly integrates GWT"""
        # With GWT enabled
        agent_with_gw = ImprovedProductionReCoNArcAngel(
            game_id="test_game",
            enable_global_workspace=True
        )

        assert agent_with_gw.hypothesis_manager.enable_global_workspace is True
        assert agent_with_gw.hypothesis_manager.global_workspace is not None

        # Without GWT enabled
        agent_without_gw = ImprovedProductionReCoNArcAngel(
            game_id="test_game",
            enable_global_workspace=False
        )

        assert agent_without_gw.hypothesis_manager.enable_global_workspace is False
        assert agent_without_gw.hypothesis_manager.global_workspace is None

        print(f"✅ Production agent integration working")

    def test_get_stats_includes_gw(self):
        """Test that get_stats includes GWT statistics"""
        manager = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=True
        )
        manager.build_improved_structure()

        stats = manager.get_stats()
        assert 'global_workspace' in stats
        assert stats['global_workspace']['enabled'] is True
        assert 'broadcast_strength' in stats['global_workspace']
        assert 'total_broadcasts' in stats['global_workspace']

        print(f"✅ Stats reporting working:")
        print(f"  GWT enabled: {stats['global_workspace']['enabled']}")
        print(f"  Broadcast strength: {stats['global_workspace']['broadcast_strength']}")

    def test_gw_disabled_stats(self):
        """Test that stats correctly report when GWT is disabled"""
        manager = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=False
        )
        manager.build_improved_structure()

        stats = manager.get_stats()
        assert 'global_workspace' in stats
        assert stats['global_workspace']['enabled'] is False

        print(f"✅ GWT disabled stats working")

    def test_broadcast_with_no_activation_attribute(self):
        """Test that broadcast handles nodes without activation attribute gracefully"""
        gw = GlobalWorkspace()

        # Create a mock graph with nodes that don't have activation
        class MockGraph:
            def __init__(self):
                self.nodes = {
                    'action_1': type('obj', (object,), {})(),
                    'action_2': type('obj', (object,), {})(),
                }

        mock_graph = MockGraph()
        candidates = [
            ("action_1", 0.8, None, None),
            ("action_2", 0.6, None, None),
        ]

        # Should not raise an error
        gw.broadcast_winner(0, candidates, mock_graph)

        # Stats should still update
        assert gw.stats['total_broadcasts'] == 1

        print(f"✅ Graceful handling of nodes without activation")

    def test_multiple_broadcasts_update_averages(self):
        """Test that multiple broadcasts correctly update running averages"""
        manager = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=True
        )
        manager.build_improved_structure()

        # First broadcast
        candidates1 = [
            ("action_1", 0.9, None, None),
            ("action_2", 0.7, None, None),
        ]
        for action_id, score, _, _ in candidates1:
            if action_id in manager.graph.nodes:
                manager.graph.nodes[action_id].activation = score

        manager.global_workspace.broadcast_winner(0, candidates1, manager.graph)

        # Second broadcast
        candidates2 = [
            ("action_2", 0.8, None, None),
            ("action_1", 0.6, None, None),
        ]
        for action_id, score, _, _ in candidates2:
            if action_id in manager.graph.nodes:
                manager.graph.nodes[action_id].activation = score

        manager.global_workspace.broadcast_winner(0, candidates2, manager.graph)

        # Verify multiple broadcasts tracked
        assert manager.global_workspace.stats['total_broadcasts'] == 2
        assert manager.global_workspace.stats['avg_winner_boost'] > 0
        assert manager.global_workspace.stats['avg_loser_suppression'] > 0

        print(f"✅ Multiple broadcasts working:")
        print(f"  Total broadcasts: {manager.global_workspace.stats['total_broadcasts']}")
        print(f"  Avg winner boost: {manager.global_workspace.stats['avg_winner_boost']:.4f}")

    def test_decision_confidence_metric(self):
        """Test that decision confidence metric tracks winner vs runner-up gap"""
        manager = ImprovedHierarchicalHypothesisManager(
            enable_global_workspace=True
        )
        manager.build_improved_structure()

        # Clear winner scenario
        candidates_clear = [
            ("action_1", 0.9, None, None),  # Winner
            ("action_2", 0.3, None, None),  # Distant runner-up
        ]
        for action_id, score, _, _ in candidates_clear:
            if action_id in manager.graph.nodes:
                manager.graph.nodes[action_id].activation = score

        manager.global_workspace.broadcast_winner(0, candidates_clear, manager.graph)

        # Decision confidence should be high (0.9 - 0.3 = 0.6)
        confidence = manager.global_workspace.stats['avg_decision_confidence']
        assert confidence >= 0.5, f"Decision confidence should be high, got {confidence}"

        print(f"✅ Decision confidence metric working:")
        print(f"  Clear winner confidence: {confidence:.3f} (expected ~0.6)")


def test_all_extensions_together():
    """Integration test: All extensions (1A, 1B, 1C, 3B) working together"""
    agent = ImprovedProductionReCoNArcAngel(
        game_id="test_integration",
        use_compact=True,              # Extension 1A
        timing_mode="discrete",        # Extension 1C
        enable_link_learning=True,     # Extension 1B
        enable_global_workspace=True   # Extension 3B
    )

    # Verify all extensions enabled
    assert agent.hypothesis_manager.use_compact is True
    assert agent.hypothesis_manager.timing_mode == "discrete"
    assert agent.link_learner is not None
    assert agent.hypothesis_manager.enable_global_workspace is True
    assert agent.hypothesis_manager.global_workspace is not None

    print(f"✅ All extensions working together:")
    print(f"  Extension 1A (Compact): {agent.hypothesis_manager.use_compact}")
    print(f"  Extension 1C (Timing): {agent.hypothesis_manager.timing_mode}")
    print(f"  Extension 1B (Link Learning): {agent.link_learner is not None}")
    print(f"  Extension 3B (Global Workspace): {agent.hypothesis_manager.enable_global_workspace}")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Global Workspace Theory Extension Tests")
    print("=" * 60)
    print()

    test_suite = TestGlobalWorkspace()

    tests = [
        ("Creation", test_suite.test_global_workspace_creation),
        ("Custom Parameters", test_suite.test_global_workspace_custom_parameters),
        ("Broadcast Suppression", test_suite.test_broadcast_winner_suppresses_losers),
        ("Statistics Tracking", test_suite.test_broadcast_updates_statistics),
        ("Hierarchy Integration", test_suite.test_hierarchy_manager_integration),
        ("Production Agent Integration", test_suite.test_production_agent_integration),
        ("Stats Reporting", test_suite.test_get_stats_includes_gw),
        ("Disabled Stats", test_suite.test_gw_disabled_stats),
        ("Graceful Handling", test_suite.test_broadcast_with_no_activation_attribute),
        ("Multiple Broadcasts", test_suite.test_multiple_broadcasts_update_averages),
        ("Decision Confidence", test_suite.test_decision_confidence_metric),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            print("-" * 60)
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Integration test
    print(f"\nRunning: All Extensions Integration")
    print("-" * 60)
    try:
        test_all_extensions_together()
        passed += 1
    except Exception as e:
        print(f"❌ FAILED: All Extensions Integration")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All Global Workspace tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        exit(1)
