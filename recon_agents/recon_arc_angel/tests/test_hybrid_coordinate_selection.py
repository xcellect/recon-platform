"""
Test hybrid coordinate selection combining BlindSquirrel + StochasticGoose
"""
import pytest
import sys
import os
import torch
import numpy as np
import scipy.ndimage

# Add parent directories to path
sys.path.insert(0, "/workspace/recon-platform")

from recon_engine.neural_terminal import CNNValidActionTerminal

class HybridCoordinateSelector:
    """
    Hybrid coordinate selection combining BlindSquirrel efficiency with StochasticGoose precision.
    
    Modes:
    - efficiency_mode: Pure BlindSquirrel (object centroids)
    - precision_mode: Pure StochasticGoose (CNN-guided pixels)  
    - hybrid_mode: BlindSquirrel objects + StochasticGoose precision within objects
    """
    
    def __init__(self, mode: str = "hybrid", cnn_terminal=None):
        self.mode = mode  # "efficiency", "precision", "hybrid"
        self.cnn_terminal = cnn_terminal
        
    def extract_objects(self, frame: np.ndarray, min_area: int = 2):
        """Extract objects using BlindSquirrel segmentation"""
        objects = []
        
        for color in range(16):
            labeled, _ = scipy.ndimage.label(frame == color)
            slices = scipy.ndimage.find_objects(labeled)
            
            for i, slc in enumerate(slices):
                if slc is None:
                    continue
                
                mask = (labeled[slc] == (i + 1))
                area = np.sum(mask)
                
                if area < min_area:
                    continue
                
                h = slc[0].stop - slc[0].start
                w = slc[1].stop - slc[1].start
                regularity = area / (h * w)
                
                ys, xs = np.nonzero(mask)
                y_centroid = ys.mean() + slc[0].start
                x_centroid = xs.mean() + slc[1].start
                
                objects.append({
                    "color": color,
                    "slice": slc,
                    "mask": mask,
                    "area": area,
                    "regularity": regularity,
                    "centroid": (int(y_centroid), int(x_centroid))
                })
        
        # Sort by importance
        objects.sort(key=lambda o: (-o["regularity"], -o["area"]))
        return objects
    
    def select_coordinate(self, frame: torch.Tensor) -> tuple:
        """Select coordinate using the configured mode"""
        
        # Convert frame for object processing
        if frame.dim() == 3:  # One-hot (16, 64, 64)
            frame_np = frame.argmax(dim=0).cpu().numpy()
        else:
            frame_np = frame.cpu().numpy()
        
        if self.mode == "efficiency":
            return self._efficiency_mode(frame_np)
        elif self.mode == "precision":
            return self._precision_mode(frame)
        elif self.mode == "hybrid":
            return self._hybrid_mode(frame, frame_np)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _efficiency_mode(self, frame_np: np.ndarray) -> tuple:
        """Pure BlindSquirrel: object centroids"""
        objects = self.extract_objects(frame_np)
        
        if not objects:
            return (32, 32)  # Center fallback
        
        # Select best object by regularity
        best_object = objects[0]  # Already sorted by importance
        return best_object["centroid"]
    
    def _precision_mode(self, frame: torch.Tensor) -> tuple:
        """Pure StochasticGoose: CNN-guided pixel selection"""
        if self.cnn_terminal is None:
            return (32, 32)
        
        # Get CNN coordinate probabilities
        measurement = self.cnn_terminal.measure(frame)
        result = self.cnn_terminal._process_measurement(measurement)
        coord_probs = result["coordinate_probabilities"]  # 64x64
        
        # Sample from CNN probabilities (StochasticGoose style)
        coord_probs_flat = coord_probs.flatten()
        coord_probs_np = coord_probs_flat.cpu().numpy()
        
        # Sample coordinate
        selected_idx = np.random.choice(len(coord_probs_np), p=coord_probs_np / coord_probs_np.sum())
        y = selected_idx // 64
        x = selected_idx % 64
        
        return (y, x)
    
    def _hybrid_mode(self, frame: torch.Tensor, frame_np: np.ndarray) -> tuple:
        """Hybrid: BlindSquirrel objects + StochasticGoose precision within objects"""
        objects = self.extract_objects(frame_np)
        
        if not objects or self.cnn_terminal is None:
            return self._efficiency_mode(frame_np)
        
        # Get CNN coordinate probabilities
        measurement = self.cnn_terminal.measure(frame)
        result = self.cnn_terminal._process_measurement(measurement)
        coord_probs = result["coordinate_probabilities"]  # 64x64
        
        # Score objects using CNN probabilities
        for obj in objects:
            slc = obj["slice"]
            y_start, y_end = slc[0].start, slc[0].stop
            x_start, x_end = slc[1].start, slc[1].stop
            
            # Ensure bounds
            y_start, y_end = max(0, y_start), min(64, y_end)
            x_start, x_end = max(0, x_start), min(64, x_end)
            
            if y_end > y_start and x_end > x_start:
                obj_cnn_score = coord_probs[y_start:y_end, x_start:x_end].max().item()
            else:
                obj_cnn_score = 0.0
            
            # Combine object properties + CNN confidence
            obj["cnn_score"] = obj_cnn_score
            obj["total_score"] = obj["regularity"] * obj_cnn_score
        
        # Select best object
        best_object = max(objects, key=lambda o: o["total_score"])
        
        # Get precise coordinate within best object using CNN
        slc = best_object["slice"]
        y_start, y_end = slc[0].start, slc[0].stop
        x_start, x_end = slc[1].start, slc[1].stop
        
        object_cnn_probs = coord_probs[y_start:y_end, x_start:x_end]
        
        if object_cnn_probs.numel() > 0:
            # Sample within object using CNN probabilities
            flat_probs = object_cnn_probs.flatten()
            flat_probs_np = flat_probs.cpu().numpy()
            
            if flat_probs_np.sum() > 0:
                selected_idx = np.random.choice(len(flat_probs_np), 
                                              p=flat_probs_np / flat_probs_np.sum())
                local_y = selected_idx // object_cnn_probs.shape[1]
                local_x = selected_idx % object_cnn_probs.shape[1]
                
                global_y = y_start + local_y
                global_x = x_start + local_x
                
                return (global_y, global_x)
        
        # Fallback to object centroid
        return best_object["centroid"]

def test_hybrid_coordinate_selector():
    """Test the hybrid coordinate selector"""
    
    # Create CNN terminal for testing
    cnn_terminal = CNNValidActionTerminal("test_cnn", use_gpu=True)
    
    # Test all three modes
    modes = ["efficiency", "precision", "hybrid"]
    
    # Create test frame
    frame = torch.zeros(16, 64, 64)
    frame[1, 20:30, 20:30] = 1.0  # Large object
    frame[2, 45, 45] = 1.0         # Single pixel
    
    if torch.cuda.is_available():
        frame = frame.cuda()
    
    for mode in modes:
        selector = HybridCoordinateSelector(mode, cnn_terminal)
        
        # Test coordinate selection
        coord = selector.select_coordinate(frame)
        y, x = coord
        
        assert 0 <= y < 64 and 0 <= x < 64
        print(f"✅ {mode:10s} mode: coordinate ({x:2d}, {y:2d})")
    
    print("✅ All modes working correctly")

def test_mode_performance_comparison():
    """Compare performance of different modes"""
    import time
    
    cnn_terminal = CNNValidActionTerminal("test_cnn", use_gpu=True)
    
    # Create test frame
    frame = torch.zeros(16, 64, 64)
    frame[1, 15:25, 15:25] = 1.0
    frame[2, 35:40, 35:40] = 1.0
    
    if torch.cuda.is_available():
        frame = frame.cuda()
    
    modes = ["efficiency", "precision", "hybrid"]
    times = {}
    
    for mode in modes:
        selector = HybridCoordinateSelector(mode, cnn_terminal)
        
        # Warm up
        selector.select_coordinate(frame)
        
        # Time multiple runs
        start = time.time()
        for _ in range(10):
            coord = selector.select_coordinate(frame)
        times[mode] = (time.time() - start) / 10
        
        print(f"✅ {mode:10s} mode: {times[mode]:.4f}s per selection")
    
    # All modes should be reasonably fast (sub-second)
    for mode, time_per_selection in times.items():
        assert time_per_selection < 0.1, f"{mode} too slow: {time_per_selection:.4f}s"
    
    print("✅ All modes are fast enough for production use")

def test_coordinate_quality_comparison():
    """Test coordinate quality for different game types"""
    
    cnn_terminal = CNNValidActionTerminal("test_cnn", use_gpu=True)
    
    # Test case 1: Large uniform objects (BlindSquirrel assumption holds)
    uniform_frame = torch.zeros(16, 64, 64)
    uniform_frame[1, 10:30, 10:30] = 1.0  # Large uniform square
    
    if torch.cuda.is_available():
        uniform_frame = uniform_frame.cuda()
    
    efficiency_selector = HybridCoordinateSelector("efficiency", cnn_terminal)
    hybrid_selector = HybridCoordinateSelector("hybrid", cnn_terminal)
    
    # Test multiple selections
    efficiency_coords = []
    hybrid_coords = []
    
    for _ in range(5):
        eff_coord = efficiency_selector.select_coordinate(uniform_frame)
        hyb_coord = hybrid_selector.select_coordinate(uniform_frame)
        
        efficiency_coords.append(eff_coord)
        hybrid_coords.append(hyb_coord)
    
    print(f"Uniform object - Efficiency coords: {efficiency_coords}")
    print(f"Uniform object - Hybrid coords: {hybrid_coords}")
    
    # For uniform objects, both should give reasonable results
    for coord in efficiency_coords + hybrid_coords:
        y, x = coord
        assert 10 <= y <= 29 and 10 <= x <= 29, f"Coord {coord} outside object bounds"
    
    print("✅ Both modes handle uniform objects correctly")
