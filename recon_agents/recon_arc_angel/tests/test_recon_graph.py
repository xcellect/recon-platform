"""
Test ReCoN graph phase - 1:30-2:30

Test building root → 5 action hypotheses + action_click,
add region hypotheses with terminals, wire sub weights from p_action/p_region.
"""
import pytest
import sys
import os
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.graph import ReCoNGraph
from recon_engine.node import ReCoNState
from recon_engine.neural_terminal import CNNValidActionTerminal

def test_basic_action_hypothesis_graph():
    """Test that we can build a basic action hypothesis graph"""
    g = ReCoNGraph()
    
    # Root hypothesis
    g.add_node("frame_change_hypothesis", node_type="script")
    
    # Action hypotheses
    for i in range(1, 6):
        g.add_node(f"action_{i}", node_type="script")
        g.add_link("frame_change_hypothesis", f"action_{i}", "sub", weight=1.0)
    
    # Click action hypothesis
    g.add_node("action_click", node_type="script")
    g.add_link("frame_change_hypothesis", "action_click", "sub", weight=1.0)
    
    # Verify structure
    assert len(g.nodes) == 7  # root + 5 actions + click
    assert len(g.links) == 12  # 6 sub + 6 reciprocal sur links
    
    # Verify all children are connected to root via sub links
    sub_links = g.get_links(source="frame_change_hypothesis", link_type="sub")
    assert len(sub_links) == 6
    
    child_ids = {link.target for link in sub_links}
    expected_children = {"action_1", "action_2", "action_3", "action_4", "action_5", "action_click"}
    assert child_ids == expected_children

def test_region_hypothesis_graph():
    """Test that we can build region hypotheses under action_click"""
    g = ReCoNGraph()
    
    # Root and click action
    g.add_node("frame_change_hypothesis", node_type="script")
    g.add_node("action_click", node_type="script")
    g.add_link("frame_change_hypothesis", "action_click", "sub", weight=1.0)
    
    # Add 8x8 = 64 region hypotheses under action_click
    for region_y in range(8):
        for region_x in range(8):
            region_id = f"region_{region_y}_{region_x}"
            g.add_node(region_id, node_type="script")
            g.add_link("action_click", region_id, "sub", weight=1.0)
    
    # Verify structure
    click_sub_links = g.get_links(source="action_click", link_type="sub")
    assert len(click_sub_links) == 64
    
    # Check some specific regions exist
    region_ids = {link.target for link in click_sub_links}
    assert "region_0_0" in region_ids
    assert "region_3_4" in region_ids
    assert "region_7_7" in region_ids

def test_cnn_terminal_integration():
    """Test that CNN terminal can be integrated with action hypotheses"""
    g = ReCoNGraph()
    
    # Root hypothesis
    g.add_node("frame_change_hypothesis", node_type="script")
    
    # CNN terminal that provides action probabilities
    g.add_node("cnn_terminal", node_type="terminal")
    g.add_link("frame_change_hypothesis", "cnn_terminal", "sub", weight=1.0)
    
    # Create actual CNN terminal
    cnn_terminal = CNNValidActionTerminal("cnn_terminal")
    
    # Replace the graph's terminal with our CNN terminal
    g.nodes["cnn_terminal"] = cnn_terminal
    
    # Test that terminal can be measured
    dummy_frame = torch.zeros(16, 64, 64)
    dummy_frame[1, 20:30, 20:30] = 1.0
    
    measurement = cnn_terminal.measure(dummy_frame)
    assert measurement.numel() == 4101
    
    # Test processing
    result = cnn_terminal._process_measurement(measurement)
    assert result["sur"] == "confirm"
    assert "action_probabilities" in result
    assert "coordinate_probabilities" in result

class ActionHypothesisGraph:
    """Helper class to build the action hypothesis graph structure"""
    
    def __init__(self):
        self.graph = ReCoNGraph()
        self.cnn_terminal = None
        self.region_aggregator = None
        
    def build_basic_structure(self):
        """Build the basic root -> actions structure"""
        # Root hypothesis
        self.graph.add_node("frame_change_hypothesis", node_type="script")
        
        # Individual action hypotheses (ACTION1-ACTION5)
        for i in range(1, 6):
            action_id = f"action_{i}"
            self.graph.add_node(action_id, node_type="script")
            self.graph.add_link("frame_change_hypothesis", action_id, "sub", weight=1.0)
        
        # Click action hypothesis (ACTION6)
        self.graph.add_node("action_click", node_type="script")
        self.graph.add_link("frame_change_hypothesis", "action_click", "sub", weight=1.0)
        
        return self
    
    def add_cnn_terminal(self):
        """Add CNN terminal for action/coordinate prediction"""
        # CNN terminal
        self.cnn_terminal = CNNValidActionTerminal("cnn_terminal")
        self.graph.nodes["cnn_terminal"] = self.cnn_terminal
        
        # Connect to root for global frame analysis
        self.graph.add_link("frame_change_hypothesis", "cnn_terminal", "sub", weight=1.0)
        
        return self
    
    def add_region_hypotheses(self):
        """Add 8x8 region hypotheses under action_click"""
        for region_y in range(8):
            for region_x in range(8):
                region_id = f"region_{region_y}_{region_x}"
                self.graph.add_node(region_id, node_type="script")
                self.graph.add_link("action_click", region_id, "sub", weight=1.0)
        
        return self
    
    def update_weights_from_cnn(self, frame: torch.Tensor):
        """Update link weights based on CNN predictions"""
        if self.cnn_terminal is None:
            raise ValueError("CNN terminal not added")
        
        # Get CNN predictions
        measurement = self.cnn_terminal.measure(frame)
        result = self.cnn_terminal._process_measurement(measurement)
        
        action_probs = result["action_probabilities"]
        coord_probs = result["coordinate_probabilities"]
        
        # Update action hypothesis weights
        for i in range(5):
            action_id = f"action_{i + 1}"
            weight = float(action_probs[i])
            
            # Find and update the sub link weight
            for link in self.graph.get_links(source="frame_change_hypothesis", target=action_id):
                if link.type == "sub":
                    link.weight = weight
        
        # Update click action weight (sum of coordinate probabilities)
        click_weight = float(coord_probs.sum() / (64 * 64))  # Normalize
        for link in self.graph.get_links(source="frame_change_hypothesis", target="action_click"):
            if link.type == "sub":
                link.weight = click_weight
        
        # Update region weights if regions exist
        if "region_0_0" in self.graph.nodes:
            from recon_agents.recon_arc_angel.region_aggregator import RegionAggregator
            aggregator = RegionAggregator()
            region_scores = aggregator.aggregate_to_regions(coord_probs)
            
            for region_y in range(8):
                for region_x in range(8):
                    region_id = f"region_{region_y}_{region_x}"
                    weight = float(region_scores[region_y, region_x])
                    
                    # Update sub link weight from action_click to region
                    for link in self.graph.get_links(source="action_click", target=region_id):
                        if link.type == "sub":
                            link.weight = weight

def test_action_hypothesis_graph_builder():
    """Test the ActionHypothesisGraph builder class"""
    builder = ActionHypothesisGraph()
    builder.build_basic_structure()
    
    # Check basic structure
    assert len(builder.graph.nodes) == 7  # root + 6 actions
    assert "frame_change_hypothesis" in builder.graph.nodes
    assert "action_1" in builder.graph.nodes
    assert "action_click" in builder.graph.nodes

def test_action_hypothesis_graph_with_cnn():
    """Test ActionHypothesisGraph with CNN terminal"""
    builder = ActionHypothesisGraph()
    builder.build_basic_structure().add_cnn_terminal()
    
    # Check CNN terminal added
    assert "cnn_terminal" in builder.graph.nodes
    assert builder.cnn_terminal is not None
    
    # Test weight updates
    dummy_frame = torch.zeros(16, 64, 64)
    dummy_frame[2, 15:25, 15:25] = 1.0
    
    builder.update_weights_from_cnn(dummy_frame)
    
    # Check that weights were updated
    action_1_links = builder.graph.get_links(source="frame_change_hypothesis", target="action_1")
    assert len(action_1_links) > 0
    
    # Weight should be between 0 and 1 (sigmoid output)
    sub_link = next(link for link in action_1_links if link.type == "sub")
    assert 0 <= sub_link.weight <= 1

def test_full_hierarchy_with_regions():
    """Test full hierarchy with regions"""
    builder = ActionHypothesisGraph()
    builder.build_basic_structure().add_cnn_terminal().add_region_hypotheses()
    
    # Check full structure
    assert len(builder.graph.nodes) == 7 + 1 + 64  # root + actions + cnn + regions
    
    # Test weight updates with regions
    dummy_frame = torch.zeros(16, 64, 64)
    dummy_frame[1, 40:50, 40:50] = 1.0  # Should activate region (5, 5)
    
    builder.update_weights_from_cnn(dummy_frame)
    
    # Check that region weights were updated
    region_5_5_links = builder.graph.get_links(source="action_click", target="region_5_5")
    assert len(region_5_5_links) > 0
    
    sub_link = next(link for link in region_5_5_links if link.type == "sub")
    assert 0 <= sub_link.weight <= 1
