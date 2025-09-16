"""
Test CNN integration phase - 0:30-1:30

Test that we can instantiate CNNValidActionTerminal once, cache outputs per frame,
and implement region aggregator from 64×64 probs to region scores for 8×8 grid.
"""
import pytest
import sys
import os
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.neural_terminal import CNNValidActionTerminal

def test_cnn_terminal_instantiation():
    """Test that CNNValidActionTerminal can be instantiated and produces expected output shape"""
    terminal = CNNValidActionTerminal("action_terminal")
    
    # Create a dummy 16-channel 64x64 input (one-hot encoded frame)
    dummy_frame = torch.zeros(16, 64, 64)
    # Set some pixels to create a pattern
    dummy_frame[1, 10:20, 10:20] = 1.0  # Color 1 in a square
    
    # Test measurement
    output = terminal.measure(dummy_frame)
    
    # Should return tensor with 4101 elements (5 actions + 4096 coordinates)
    assert isinstance(output, torch.Tensor)
    assert output.numel() == 4101, f"Expected 4101 elements, got {output.numel()}"

def test_cnn_terminal_caching():
    """Test that CNNValidActionTerminal caches outputs for same frame"""
    terminal = CNNValidActionTerminal("action_terminal")
    
    dummy_frame = torch.zeros(16, 64, 64)
    dummy_frame[1, 10:20, 10:20] = 1.0
    
    # First measurement
    output1 = terminal.measure(dummy_frame)
    
    # Second measurement with same frame should be cached
    output2 = terminal.measure(dummy_frame)
    
    # Should be identical (from cache)
    assert torch.equal(output1, output2)
    
    # Cache should have one entry
    assert len(terminal._output_cache) == 1

def test_cnn_terminal_produces_probabilities():
    """Test that CNNValidActionTerminal produces action and coordinate probabilities"""
    terminal = CNNValidActionTerminal("action_terminal")
    
    dummy_frame = torch.zeros(16, 64, 64)
    dummy_frame[2, 30:34, 30:34] = 1.0  # Color 2 in a small square
    
    # Process measurement to get probabilities
    measurement = terminal.measure(dummy_frame)
    result = terminal._process_measurement(measurement)
    
    # Should return confirm with action and coordinate probabilities
    assert result["sur"] == "confirm"
    assert "action_probabilities" in result
    assert "coordinate_probabilities" in result
    
    action_probs = result["action_probabilities"]
    coord_probs = result["coordinate_probabilities"]
    
    # Check shapes
    assert action_probs.shape == torch.Size([5]), f"Action probs shape: {action_probs.shape}"
    assert coord_probs.shape == torch.Size([64, 64]), f"Coord probs shape: {coord_probs.shape}"
    
    # Check that probabilities are in valid range [0, 1] after sigmoid
    assert torch.all(action_probs >= 0) and torch.all(action_probs <= 1)
    assert torch.all(coord_probs >= 0) and torch.all(coord_probs <= 1)

class RegionAggregator:
    """Region aggregator to convert 64x64 coordinate probabilities to 8x8 region scores"""
    
    def __init__(self, region_size: int = 8):
        self.region_size = region_size
        self.grid_size = 64
        assert self.grid_size % region_size == 0, "Grid size must be divisible by region size"
        self.regions_per_dim = self.grid_size // region_size
    
    def aggregate_to_regions(self, coord_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert 64x64 coordinate probabilities to 8x8 region scores.
        
        Args:
            coord_probs: Tensor of shape (64, 64) with coordinate probabilities
            
        Returns:
            Tensor of shape (8, 8) with region scores
        """
        assert coord_probs.shape == (64, 64), f"Expected (64, 64), got {coord_probs.shape}"
        
        # Reshape to (8, 8, 8, 8) to group into regions
        reshaped = coord_probs.view(
            self.regions_per_dim, self.region_size,
            self.regions_per_dim, self.region_size
        )
        
        # Take max over each 8x8 region (could also use mean)
        region_scores = reshaped.max(dim=3)[0].max(dim=1)[0]
        
        assert region_scores.shape == (8, 8), f"Expected (8, 8), got {region_scores.shape}"
        return region_scores

def test_region_aggregator():
    """Test that RegionAggregator correctly converts 64x64 to 8x8"""
    aggregator = RegionAggregator(region_size=8)
    
    # Create test coordinate probabilities with a hot spot
    coord_probs = torch.zeros(64, 64)
    coord_probs[16:24, 16:24] = 0.9  # High probability in region (2, 2)
    coord_probs[48:56, 48:56] = 0.7  # Medium probability in region (6, 6)
    
    region_scores = aggregator.aggregate_to_regions(coord_probs)
    
    # Check shape
    assert region_scores.shape == (8, 8)
    
    # Check that hot spots are captured
    assert region_scores[2, 2] == 0.9, f"Region (2,2) score: {region_scores[2, 2]}"
    assert region_scores[6, 6] == 0.7, f"Region (6,6) score: {region_scores[6, 6]}"
    
    # Check that other regions have lower scores
    assert region_scores[0, 0] < 0.1, f"Region (0,0) score: {region_scores[0, 0]}"

def test_region_aggregator_with_cnn_output():
    """Test RegionAggregator with actual CNN terminal output"""
    terminal = CNNValidActionTerminal("action_terminal")
    aggregator = RegionAggregator(region_size=8)
    
    # Create dummy frame
    dummy_frame = torch.zeros(16, 64, 64)
    dummy_frame[3, 20:30, 20:30] = 1.0  # Color 3 in a square
    
    # Get CNN output
    measurement = terminal.measure(dummy_frame)
    result = terminal._process_measurement(measurement)
    coord_probs = result["coordinate_probabilities"]
    
    # Aggregate to regions
    region_scores = aggregator.aggregate_to_regions(coord_probs)
    
    # Should produce valid 8x8 region scores
    assert region_scores.shape == (8, 8)
    assert torch.all(region_scores >= 0) and torch.all(region_scores <= 1)
    
    # At least one region should have some activation
    assert region_scores.max() > 0.0
