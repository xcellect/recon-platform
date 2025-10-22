"""
Quick validation tests for Extensions 1A, 1B, 1C

Run: pytest tests/test_extensions.py -v
"""

import pytest
import torch
from recon_agents.recon_arc_angel import (
    LinkWeightLearner,
    ImprovedHierarchicalHypothesisManager,
    ImprovedProductionReCoNArcAngel
)


class TestExtension1A:
    """Test Extension 1A: Compact Implementation with Gen Loops"""

    def test_compact_graph_creation(self):
        """Test that compact graph can be created"""
        manager = ImprovedHierarchicalHypothesisManager(use_compact=True)
        manager.build_improved_structure()

        assert manager.use_compact is True
        assert "CompactReCoNGraph" in type(manager.graph).__name__

    def test_gen_loops_added(self):
        """Test that gen loops can be added to compact nodes"""
        manager = ImprovedHierarchicalHypothesisManager(use_compact=True)
        manager.build_improved_structure()

        # Add a test object node to the graph (compact nodes support gen loops)
        from recon_engine.compact import CompactReCoNNode
        test_obj = manager.graph.add_node("test_object", "script")
        assert isinstance(test_obj, CompactReCoNNode), "Should be CompactReCoNNode"

        # Manually add gen loop
        manager.graph.add_link("test_object", "test_object", "gen", weight=0.95)

        # Verify gen loop was added
        gen_links = [l for l in manager.graph.links
                     if l.source == "test_object" and l.target == "test_object" and l.type == "gen"]
        assert len(gen_links) > 0, "Gen loop should be added"
        assert abs(gen_links[0].weight - 0.95) < 0.01, "Gen loop weight should be 0.95"


class TestExtension1B:
    """Test Extension 1B: Link Weight Learning"""

    def test_link_learner_creation(self):
        """Test link learner can be created"""
        learner = LinkWeightLearner(learning_rate=0.1)
        assert learner.lr == 0.1
        assert learner.blend_ratio == 0.3

    def test_weight_update(self):
        """Test weight updates from outcomes"""
        learner = LinkWeightLearner(learning_rate=0.5)

        # Initial weight should be 0.5
        initial = learner.get_blended_weight("A", "B", "sub", cnn_prior=0.7)

        # Update with success
        learner.update_from_outcome("A", "B", "sub", outcome=1.0)

        # Weight should increase toward 1.0
        updated = learner.learned_weights[("A", "B", "sub")]
        assert updated > 0.5, "Weight should increase after success"

    def test_metrics(self):
        """Test metrics tracking"""
        learner = LinkWeightLearner()

        for i in range(5):
            learner.update_from_outcome("A", f"B{i}", "sub", outcome=1.0)

        metrics = learner.get_metrics()
        assert metrics['total_links_learned'] == 5
        assert metrics['total_updates'] == 5
        assert len(metrics['top_learned_links']) <= 5


class TestExtension1C:
    """Test Extension 1C: Timing Modes"""

    def test_discrete_timing(self):
        """Test discrete timing configuration"""
        manager = ImprovedHierarchicalHypothesisManager(timing_mode="discrete")
        manager.build_improved_structure()

        action_1 = manager.graph.get_node("action_1")
        assert action_1.timing_mode == "discrete"
        assert action_1.discrete_wait_steps == 2

    def test_activation_timing(self):
        """Test activation-based timing"""
        manager = ImprovedHierarchicalHypothesisManager(timing_mode="activation")
        manager.build_improved_structure()

        action_1 = manager.graph.get_node("action_1")
        assert action_1.timing_mode == "activation"
        assert action_1.activation_decay_rate == 0.8

    def test_hybrid_timing(self):
        """Test hybrid timing mode"""
        manager = ImprovedHierarchicalHypothesisManager(timing_mode="hybrid")
        manager.build_improved_structure()

        action_1 = manager.graph.get_node("action_1")
        action_click = manager.graph.get_node("action_click")

        assert action_1.timing_mode == "discrete"
        assert action_click.timing_mode == "activation"


class TestAllExtensions:
    """Test all extensions working together"""

    def test_agent_with_all_extensions(self):
        """Test creating agent with all extensions enabled"""
        agent = ImprovedProductionReCoNArcAngel(
            use_compact=True,
            enable_link_learning=True,
            timing_mode="hybrid"
        )

        assert agent.hypothesis_manager.use_compact is True
        assert agent.link_learner is not None
        assert agent.hypothesis_manager.timing_mode == "hybrid"

    def test_stats_tracking(self):
        """Test that extension stats are tracked"""
        agent = ImprovedProductionReCoNArcAngel(enable_link_learning=True)

        # Check extension stats exist
        assert 'link_learning_updates' in agent.stats
        assert 'learned_links_count' in agent.stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
