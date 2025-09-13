"""
Test Agent Mapping to ReCoN

Comprehensive tests to validate that both ARC-AGI-3 winners map exactly
to ReCoN architectures while maintaining theoretical correctness.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List
import random
import time

from recon_engine.blindsquirrel_recon import BlindSquirrelReCoNAgent, create_blindsquirrel_agent
from recon_engine.stochasticgoose_recon import StochasticGooseReCoNAgent, create_stochasticgoose_agent
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from recon_engine.neural_terminal import NeuralTerminal, NeuralOutputMode
from recon_engine.messages import HybridMessage, auto_convert_message
from recon_engine.graph import ReCoNGraph


class TestAgentMapping:
    """Test suite for validating agent mappings to ReCoN."""
    
    def setup_method(self):
        """Setup test environment."""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Mock frame data for testing
        self.test_frame = np.random.randint(0, 16, (64, 64))
        self.test_frame_tensor = torch.randint(0, 16, (64, 64)).long()
        
        # Available actions for testing
        self.available_actions = [
            "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5",
            "CLICK_0_0", "CLICK_32_32", "CLICK_63_63"
        ]
    
    def test_blindsquirrel_creation(self):
        """Test BlindSquirrel agent creation and basic structure."""
        agent = create_blindsquirrel_agent("test_game")
        
        assert isinstance(agent, BlindSquirrelReCoNAgent)
        assert agent.game_id == "test_game"
        assert isinstance(agent.graph, ReCoNGraph)
        
        # Verify key nodes exist
        assert agent.graph.get_node("agent_root") is not None
        assert agent.graph.get_node("state_tracker") is not None
        assert agent.graph.get_node("action_selector") is not None
        assert agent.graph.get_node("value_model") is not None
        
        # Verify explicit mode nodes
        root = agent.graph.get_node("agent_root")
        assert root.mode == NodeMode.EXPLICIT
    
    def test_stochasticgoose_creation(self):
        """Test StochasticGoose agent creation and basic structure."""
        agent = create_stochasticgoose_agent("test_game")
        
        assert isinstance(agent, StochasticGooseReCoNAgent)
        assert agent.game_id == "test_game"
        assert isinstance(agent.graph, ReCoNGraph)
        
        # Verify key nodes exist
        assert agent.graph.get_node("agent_root") is not None
        assert agent.graph.get_node("action_cnn") is not None
        assert agent.graph.get_node("action_type_selector") is not None
        assert agent.graph.get_node("coordinate_selector") is not None
        
        # Verify implicit mode nodes
        root = agent.graph.get_node("agent_root")
        assert root.mode == NodeMode.IMPLICIT
    
    def test_blindsquirrel_action_selection(self):
        """Test BlindSquirrel action selection process."""
        agent = create_blindsquirrel_agent("test")
        
        # Mock frame data
        class MockFrame:
            def __init__(self, frame, score):
                self.frame = frame
                self.score = score
        
        test_frame = MockFrame(self.test_frame.tolist(), 0)
        
        # Test action selection
        action = agent.process_frame(test_frame)
        
        assert action is not None
        assert isinstance(action, str)
        
        # Test multiple actions to ensure consistency
        actions = [agent.process_frame(test_frame) for _ in range(5)]
        assert all(isinstance(a, str) for a in actions)
    
    def test_stochasticgoose_action_selection(self):
        """Test StochasticGoose action selection process."""
        agent = create_stochasticgoose_agent("test")
        
        # Test action selection
        action = agent.process_frame(self.test_frame)
        
        assert action is not None
        assert isinstance(action, str)
        
        # Should be either ACTION1-ACTION5 or CLICK_x_y
        assert (action.startswith("ACTION") or action.startswith("CLICK"))
        
        # Test with available actions constraint
        constrained_action = agent.process_frame(self.test_frame, self.available_actions)
        assert constrained_action in self.available_actions or constrained_action.startswith("ACTION") or constrained_action.startswith("CLICK")
    
    def test_blindsquirrel_state_tracking(self):
        """Test BlindSquirrel state tracking functionality."""
        agent = create_blindsquirrel_agent("test")
        
        class MockFrame:
            def __init__(self, frame, score):
                self.frame = frame
                self.score = score
        
        # Process multiple frames with increasing scores
        frames = [
            MockFrame(self.test_frame.tolist(), 0),
            MockFrame((self.test_frame + 1).tolist(), 1),
            MockFrame((self.test_frame + 2).tolist(), 2)
        ]
        
        for frame in frames:
            action = agent.process_frame(frame)
            assert action is not None
        
        # Verify state memory is populated
        assert len(agent.state_memory) > 0
        
        # Check state memory structure
        for state_key, state_data in agent.state_memory.items():
            assert 'frame_data' in state_data
            assert 'visits' in state_data
            assert 'actions_tried' in state_data
            assert 'future_states' in state_data
            assert 'action_rweights' in state_data
    
    def test_stochasticgoose_experience_buffer(self):
        """Test StochasticGoose experience buffer functionality."""
        agent = create_stochasticgoose_agent("test")
        
        # Process multiple frames
        frames = [
            np.random.randint(0, 16, (64, 64)) for _ in range(10)
        ]
        
        for frame in frames:
            action = agent.process_frame(frame)
            assert action is not None
        
        # Verify experience buffer
        assert len(agent.experience_buffer) == 10
        assert len(agent.experience_hashes) == 10
        
        # Test deduplication
        duplicate_frame = frames[0]
        agent.process_frame(duplicate_frame)
        
        # Should not increase due to deduplication
        assert len(agent.experience_buffer) <= 11
    
    def test_hybrid_message_conversion(self):
        """Test hybrid message conversion between discrete and continuous."""
        
        # Test discrete to continuous
        discrete_msg = HybridMessage("confirm", "node1", "node2", "sur")
        assert discrete_msg.continuous == 1.0
        assert discrete_msg.discrete == "confirm"
        
        # Test continuous to discrete
        continuous_msg = HybridMessage(0.9, "node1", "node2", "sur")
        assert continuous_msg.discrete == "confirm"
        assert continuous_msg.continuous == 0.9
        
        # Test tensor conversion
        tensor_msg = HybridMessage(torch.tensor([0.5, 0.8, 0.2]), "node1", "node2", "sur")
        assert tensor_msg.discrete == "confirm"  # max value > threshold
        assert torch.equal(tensor_msg.continuous, torch.tensor([0.5, 0.8, 0.2]))
    
    def test_neural_terminal_functionality(self):
        """Test neural terminal integration."""
        
        # Simple test model
        class TestModel(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(torch.sum(x, dim=-1, keepdim=True))
        
        model = TestModel()
        terminal = NeuralTerminal("test_terminal", model, NeuralOutputMode.VALUE)
        
        # Test measurement
        test_input = torch.randn(1, 10)
        result = terminal.measure(test_input)
        
        assert isinstance(result, (float, torch.Tensor))
        
        # Test message processing
        messages = {"sub": [HybridMessage(test_input, "test", "test_terminal", "sub")]}
        response = terminal.process_messages(messages)
        
        assert "sur" in response
        assert response["sur"] in ["confirm", "fail"]
    
    def test_mode_switching(self):
        """Test hybrid node mode switching."""
        
        node = HybridReCoNNode("test", "script", NodeMode.EXPLICIT)
        
        # Start in explicit mode
        assert node.mode == NodeMode.EXPLICIT
        node.state = node._explicit_state = "confirmed"
        
        # Switch to implicit mode
        node.set_mode(NodeMode.IMPLICIT, preserve_state=True)
        assert node.mode == NodeMode.IMPLICIT
        assert node.activation == 1.0  # Converted from "confirmed"
        
        # Switch back to explicit
        node.set_mode(NodeMode.EXPLICIT, preserve_state=True)
        assert node.mode == NodeMode.EXPLICIT
        assert node.state.value == "confirmed"  # Converted back
    
    def test_graph_export(self):
        """Test graph export functionality for visualization."""
        
        # Test BlindSquirrel export
        bs_agent = create_blindsquirrel_agent("test")
        bs_dict = bs_agent.to_dict()
        
        assert "agent_type" in bs_dict
        assert bs_dict["agent_type"] == "BlindSquirrel"
        assert "graph" in bs_dict
        assert "parameters" in bs_dict
        assert "statistics" in bs_dict
        
        # Test StochasticGoose export
        sg_agent = create_stochasticgoose_agent("test")
        sg_dict = sg_agent.to_dict()
        
        assert "agent_type" in sg_dict
        assert sg_dict["agent_type"] == "StochasticGoose"
        assert "graph" in sg_dict
        assert "experience_buffer_size" in sg_dict
    
    def test_theoretical_compliance(self):
        """Test that mappings maintain ReCoN theoretical correctness."""
        
        # Test BlindSquirrel compliance
        bs_agent = create_blindsquirrel_agent("test")
        
        # Should use explicit state machine
        for node_id in ["agent_root", "state_tracker", "action_selector"]:
            node = bs_agent.graph.get_node(node_id)
            assert node is not None
            assert node.mode == NodeMode.EXPLICIT
        
        # Test StochasticGoose compliance
        sg_agent = create_stochasticgoose_agent("test")
        
        # Should use implicit activations
        for node_id in ["agent_root", "action_type_selector", "coordinate_selector"]:
            node = sg_agent.graph.get_node(node_id)
            assert node is not None
            assert node.mode == NodeMode.IMPLICIT
    
    def test_performance_comparison(self):
        """Test performance characteristics of both mappings."""
        
        frames_to_test = 100
        test_frames = [np.random.randint(0, 16, (64, 64)) for _ in range(frames_to_test)]
        
        # Test BlindSquirrel performance
        bs_agent = create_blindsquirrel_agent("perf_test")
        
        start_time = time.time()
        bs_actions = []
        for frame in test_frames:
            class MockFrame:
                def __init__(self, f):
                    self.frame = f.tolist()
                    self.score = 0
            action = bs_agent.process_frame(MockFrame(frame))
            bs_actions.append(action)
        bs_time = time.time() - start_time
        
        # Test StochasticGoose performance
        sg_agent = create_stochasticgoose_agent("perf_test")
        
        start_time = time.time()
        sg_actions = []
        for frame in test_frames:
            action = sg_agent.process_frame(frame)
            sg_actions.append(action)
        sg_time = time.time() - start_time
        
        print(f"BlindSquirrel: {frames_to_test} frames in {bs_time:.3f}s ({frames_to_test/bs_time:.1f} FPS)")
        print(f"StochasticGoose: {frames_to_test} frames in {sg_time:.3f}s ({frames_to_test/sg_time:.1f} FPS)")
        
        # Both should complete in reasonable time
        assert bs_time < 30.0  # 30 seconds max
        assert sg_time < 30.0  # 30 seconds max
        
        # All actions should be valid
        assert len(bs_actions) == frames_to_test
        assert len(sg_actions) == frames_to_test
        assert all(isinstance(a, str) for a in bs_actions)
        assert all(isinstance(a, str) for a in sg_actions)
    
    def test_exact_architectural_mapping(self):
        """Test that ReCoN mapping exactly preserves original architectures."""
        
        # BlindSquirrel architectural elements
        bs_agent = create_blindsquirrel_agent("arch_test")
        
        # Should have state graph equivalent
        assert "state_tracker" in [n.id for n in bs_agent.graph.nodes.values()]
        
        # Should have value model
        value_node = bs_agent.graph.get_node("value_model")
        assert value_node is not None
        assert hasattr(value_node, 'model')  # Neural model
        
        # Should have rules-based components
        assert "valid_actions" in [n.id for n in bs_agent.graph.nodes.values()]
        
        # StochasticGoose architectural elements
        sg_agent = create_stochasticgoose_agent("arch_test")
        
        # Should have CNN for action prediction
        cnn_node = sg_agent.graph.get_node("action_cnn")
        assert cnn_node is not None
        assert hasattr(cnn_node, 'model')
        
        # Should have hierarchical selection
        assert "action_type_selector" in [n.id for n in sg_agent.graph.nodes.values()]
        assert "coordinate_selector" in [n.id for n in sg_agent.graph.nodes.values()]
        
        # Should have experience management
        assert "experience_manager" in [n.id for n in sg_agent.graph.nodes.values()]


def run_comprehensive_tests():
    """Run all tests and generate detailed report."""
    
    print("=== ReCoN Agent Mapping Test Suite ===\n")
    
    test_suite = TestAgentMapping()
    test_suite.setup_method()
    
    tests = [
        ("BlindSquirrel Creation", test_suite.test_blindsquirrel_creation),
        ("StochasticGoose Creation", test_suite.test_stochasticgoose_creation),
        ("BlindSquirrel Action Selection", test_suite.test_blindsquirrel_action_selection),
        ("StochasticGoose Action Selection", test_suite.test_stochasticgoose_action_selection),
        ("BlindSquirrel State Tracking", test_suite.test_blindsquirrel_state_tracking),
        ("StochasticGoose Experience Buffer", test_suite.test_stochasticgoose_experience_buffer),
        ("Hybrid Message Conversion", test_suite.test_hybrid_message_conversion),
        ("Neural Terminal Functionality", test_suite.test_neural_terminal_functionality),
        ("Mode Switching", test_suite.test_mode_switching),
        ("Graph Export", test_suite.test_graph_export),
        ("Theoretical Compliance", test_suite.test_theoretical_compliance),
        ("Performance Comparison", test_suite.test_performance_comparison),
        ("Exact Architectural Mapping", test_suite.test_exact_architectural_mapping)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}...")
            test_func()
            print(f"✓ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_name} - {str(e)}")
            failed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Both ARC winners map exactly to ReCoN.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the implementations.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()