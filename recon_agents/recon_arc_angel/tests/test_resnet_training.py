"""
Test ResNet training implementation (BlindSquirrel style)
"""
import pytest
import sys
import os
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_agents.recon_arc_angel.learning_manager import LearningManager
from recon_engine.neural_terminal import CNNValidActionTerminal, ResNetActionValueTerminal

def test_resnet_training_integration():
    """Test that ResNet training is now integrated"""
    
    # Create learning manager with both terminals
    learning_manager = LearningManager(buffer_size=100, batch_size=5, train_frequency=3)
    
    # Set up CNN terminal
    cnn_terminal = CNNValidActionTerminal("test_cnn", use_gpu=True)
    learning_manager.set_cnn_terminal(cnn_terminal)
    
    # Set up ResNet terminal
    resnet_terminal = ResNetActionValueTerminal("test_resnet")
    learning_manager.set_resnet_terminal(resnet_terminal)
    
    # Verify both are set up
    assert learning_manager.cnn_terminal is not None
    assert learning_manager.resnet_terminal is not None
    assert learning_manager.optimizer is not None
    assert learning_manager.resnet_optimizer is not None
    
    print("✅ Dual training setup complete")

def test_state_transition_tracking():
    """Test that state transitions are tracked for ResNet training"""
    learning_manager = LearningManager()
    resnet_terminal = ResNetActionValueTerminal("test_resnet")
    learning_manager.set_resnet_terminal(resnet_terminal)
    
    # Add state transitions
    frame1 = torch.zeros(16, 64, 64)
    frame1[1, 10, 10] = 1.0
    
    frame2 = torch.zeros(16, 64, 64)
    frame2[2, 20, 20] = 1.0
    
    learning_manager.add_state_transition(frame1, frame2, "action_1", None, "test_game", 0)
    
    # Check that states and transitions were recorded
    assert len(learning_manager.state_graph) == 2  # prev and curr state
    assert len(learning_manager.transitions) == 1  # one transition
    
    print(f"✅ State tracking: {len(learning_manager.state_graph)} states, {len(learning_manager.transitions)} transitions")

def test_resnet_training_on_score_increase():
    """Test that ResNet training is triggered on score increase"""
    learning_manager = LearningManager(batch_size=3)
    
    # Set up ResNet
    resnet_terminal = ResNetActionValueTerminal("test_resnet")
    learning_manager.set_resnet_terminal(resnet_terminal)
    
    # Add enough transitions for training
    for i in range(5):
        frame1 = torch.zeros(16, 64, 64)
        frame1[1, i, i] = 1.0
        
        frame2 = torch.zeros(16, 64, 64)
        frame2[2, i, i] = 1.0
        
        learning_manager.add_state_transition(frame1, frame2, "action_1", None, "test_game", 0)
    
    # Trigger score increase (should train ResNet)
    initial_score = learning_manager.current_score
    score_increased = learning_manager.on_score_change(1, "test_game")
    
    assert score_increased == True
    assert learning_manager.current_score == 1
    
    print("✅ ResNet training triggered on score increase")

def test_dual_training_workflow():
    """Test complete dual training workflow"""
    learning_manager = LearningManager(buffer_size=50, batch_size=3, train_frequency=3)
    
    # Set up both terminals
    cnn_terminal = CNNValidActionTerminal("test_cnn", use_gpu=True)
    resnet_terminal = ResNetActionValueTerminal("test_resnet")
    
    learning_manager.set_cnn_terminal(cnn_terminal)
    learning_manager.set_resnet_terminal(resnet_terminal)
    
    print("🎓 Testing dual training workflow...")
    
    # Simulate gameplay with both training types
    for level in range(2):  # Two levels
        print(f"\n--- Level {level} ---")
        
        # Add experiences within level
        for step in range(6):
            frame1 = torch.zeros(16, 64, 64)
            frame1[1, level*10 + step, level*10 + step] = 1.0
            
            frame2 = torch.zeros(16, 64, 64)
            frame2[2, level*10 + step, level*10 + step] = 1.0
            
            # CNN experience (StochasticGoose style)
            learning_manager.add_experience(frame1, frame2, "action_1")
            
            # ResNet state transition (BlindSquirrel style)
            learning_manager.add_state_transition(frame1, frame2, "action_1", None, "test_game", level)
            
            learning_manager.step()
            
            # CNN training (periodic)
            if learning_manager.should_train():
                cnn_metrics = learning_manager.train_step()
                print(f"  CNN training: loss={cnn_metrics.get('total_loss', 0):.4f}")
        
        # Level completion (triggers ResNet training)
        if level < 1:  # Don't train on final level
            score_increased = learning_manager.on_score_change(level + 1, "test_game")
            if score_increased:
                print(f"  ResNet training triggered for level {level + 1}")
    
    # Check final stats
    stats = learning_manager.get_stats()
    print(f"\n✅ Final CNN training steps: {stats['total_training_steps']}")
    print(f"✅ Buffer resets: {stats['buffer_resets']}")
    print(f"✅ State graph size: {len(learning_manager.state_graph)}")
    print(f"✅ Transitions tracked: {len(learning_manager.transitions)}")

def test_training_performance():
    """Test that training doesn't slow down the system too much"""
    import time
    
    learning_manager = LearningManager(buffer_size=50, batch_size=5, train_frequency=5)
    
    # Set up terminals
    cnn_terminal = CNNValidActionTerminal("perf_cnn", use_gpu=True)
    resnet_terminal = ResNetActionValueTerminal("perf_resnet")
    
    learning_manager.set_cnn_terminal(cnn_terminal)
    learning_manager.set_resnet_terminal(resnet_terminal)
    
    # Time the training operations
    frame1 = torch.zeros(16, 64, 64)
    frame2 = torch.zeros(16, 64, 64)
    frame2[1, 20, 20] = 1.0
    
    # Add experiences quickly
    start = time.time()
    for i in range(10):
        f1 = frame1.clone()
        f1[1, i, i] = 1.0
        f2 = frame2.clone() 
        f2[2, i, i] = 1.0
        
        learning_manager.add_experience(f1, f2, "action_1")
        learning_manager.add_state_transition(f1, f2, "action_1", None, "perf_game", 0)
        learning_manager.step()
        
        if learning_manager.should_train():
            train_start = time.time()
            learning_manager.train_step()
            train_time = time.time() - train_start
            print(f"✅ CNN training time: {train_time:.3f}s")
            assert train_time < 5.0, f"CNN training too slow: {train_time:.3f}s"
    
    total_time = time.time() - start
    print(f"✅ Total workflow time: {total_time:.3f}s")
    
    # Test ResNet training time (on score increase)
    resnet_start = time.time()
    learning_manager.on_score_change(1, "perf_game")
    resnet_time = time.time() - resnet_start
    print(f"✅ ResNet training time: {resnet_time:.3f}s")
    
    # ResNet training can be slower (BlindSquirrel trains less frequently)
    assert resnet_time < 30.0, f"ResNet training too slow: {resnet_time:.3f}s"
