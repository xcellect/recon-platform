"""
Region Aggregator

Converts 64x64 coordinate probabilities to 8x8 region scores for hierarchical processing.
This enables the ReCoN graph to work with manageable numbers of coordinate hypotheses
while preserving spatial locality.
"""

import torch
from typing import Tuple


class RegionAggregator:
    """
    Region aggregator to convert 64x64 coordinate probabilities to 8x8 region scores.
    
    This supports the hierarchical coordinate refinement approach where we first
    select among 8x8 regions, then refine within the selected region.
    """
    
    def __init__(self, region_size: int = 8, grid_size: int = 64):
        """
        Initialize the region aggregator.
        
        Args:
            region_size: Size of each region (default 8 for 8x8 regions)
            grid_size: Size of the full grid (default 64 for 64x64 coordinates)
        """
        self.region_size = region_size
        self.grid_size = grid_size
        
        if self.grid_size % region_size != 0:
            raise ValueError(f"Grid size {grid_size} must be divisible by region size {region_size}")
        
        self.regions_per_dim = self.grid_size // region_size
    
    def aggregate_to_regions(self, coord_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert full-resolution coordinate probabilities to region scores.
        
        Args:
            coord_probs: Tensor of shape (grid_size, grid_size) with coordinate probabilities
            
        Returns:
            Tensor of shape (regions_per_dim, regions_per_dim) with region scores
        """
        expected_shape = (self.grid_size, self.grid_size)
        if coord_probs.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {coord_probs.shape}")
        
        # Reshape to group pixels into regions
        # (64, 64) -> (8, 8, 8, 8) where first two dims are region indices
        reshaped = coord_probs.view(
            self.regions_per_dim, self.region_size,
            self.regions_per_dim, self.region_size
        )
        
        # Take max over each region (could also use mean or sum)
        # This preserves the highest activation within each region
        region_scores = reshaped.max(dim=3)[0].max(dim=1)[0]
        
        expected_output_shape = (self.regions_per_dim, self.regions_per_dim)
        assert region_scores.shape == expected_output_shape
        
        return region_scores
    
    def get_region_bounds(self, region_y: int, region_x: int) -> Tuple[int, int, int, int]:
        """
        Get pixel bounds for a given region.
        
        Args:
            region_y: Region row index (0 to regions_per_dim-1)
            region_x: Region column index (0 to regions_per_dim-1)
            
        Returns:
            Tuple of (y_start, y_end, x_start, x_end) pixel coordinates
        """
        if not (0 <= region_y < self.regions_per_dim):
            raise ValueError(f"region_y {region_y} out of bounds [0, {self.regions_per_dim})")
        if not (0 <= region_x < self.regions_per_dim):
            raise ValueError(f"region_x {region_x} out of bounds [0, {self.regions_per_dim})")
        
        y_start = region_y * self.region_size
        y_end = y_start + self.region_size
        x_start = region_x * self.region_size  
        x_end = x_start + self.region_size
        
        return y_start, y_end, x_start, x_end
    
    def region_to_pixel(self, region_y: int, region_x: int, 
                       local_y: int, local_x: int) -> Tuple[int, int]:
        """
        Convert region coordinates + local coordinates to global pixel coordinates.
        
        Args:
            region_y: Region row index
            region_x: Region column index  
            local_y: Local y coordinate within region (0 to region_size-1)
            local_x: Local x coordinate within region (0 to region_size-1)
            
        Returns:
            Tuple of (global_y, global_x) pixel coordinates
        """
        if not (0 <= local_y < self.region_size):
            raise ValueError(f"local_y {local_y} out of bounds [0, {self.region_size})")
        if not (0 <= local_x < self.region_size):
            raise ValueError(f"local_x {local_x} out of bounds [0, {self.region_size})")
        
        y_start, _, x_start, _ = self.get_region_bounds(region_y, region_x)
        global_y = y_start + local_y
        global_x = x_start + local_x
        
        return global_y, global_x
    
    def pixel_to_region(self, pixel_y: int, pixel_x: int) -> Tuple[int, int, int, int]:
        """
        Convert global pixel coordinates to region + local coordinates.
        
        Args:
            pixel_y: Global y coordinate
            pixel_x: Global x coordinate
            
        Returns:
            Tuple of (region_y, region_x, local_y, local_x)
        """
        if not (0 <= pixel_y < self.grid_size):
            raise ValueError(f"pixel_y {pixel_y} out of bounds [0, {self.grid_size})")
        if not (0 <= pixel_x < self.grid_size):
            raise ValueError(f"pixel_x {pixel_x} out of bounds [0, {self.grid_size})")
        
        region_y = pixel_y // self.region_size
        region_x = pixel_x // self.region_size
        local_y = pixel_y % self.region_size
        local_x = pixel_x % self.region_size
        
        return region_y, region_x, local_y, local_x
