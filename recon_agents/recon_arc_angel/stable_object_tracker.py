"""
Stable Object Tracker

Maintains stable object identity across frames using IoU-based matching.
This prevents per-object bookkeeping from attaching to wrong objects when
the object list is rebuilt and re-sorted each frame.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class PersistentObject:
    """Represents a persistent object across frames."""
    object_id: str
    current_mask: np.ndarray  # Full-frame 64x64 boolean mask
    current_slice: tuple      # Current bounding box slice
    current_color: int
    
    # Hypothesis state
    status: str = "PENDING"  # PENDING, TESTING, CONFIRMED, FAILED, REJECTED
    stale_tries: int = 0
    last_click_time: Optional[float] = None
    last_click_coords: Optional[Tuple[int, int]] = None
    
    # Properties for scoring
    area: int = 0
    regularity: float = 0.0
    contrast: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.current_mask is not None:
            self.area = int(self.current_mask.sum())


class StableObjectTracker:
    """
    Tracks objects across frames with stable identity using IoU matching.
    
    Maintains persistent object IDs and hypothesis states to enable
    proper per-object bookkeeping and systematic hypothesis reduction.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_objects: int = 50):
        """
        Initialize stable object tracker.
        
        Args:
            iou_threshold: Minimum IoU to consider objects as same across frames
            max_objects: Maximum number of objects to track
        """
        self.iou_threshold = iou_threshold
        self.max_objects = max_objects
        
        # Persistent object storage
        self.persistent_objects: Dict[str, PersistentObject] = {}
        self.next_object_id = 0
        
        # Frame-to-frame matching
        self.current_frame_objects = []  # Current frame's raw objects
        self.object_id_mapping = {}  # current_idx -> persistent_object_id
    
    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Intersection over Union between two masks."""
        try:
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            
            if union == 0:
                return 0.0
            
            return float(intersection) / float(union)
        except Exception:
            return 0.0
    
    def _create_full_frame_mask(self, obj_dict: dict) -> np.ndarray:
        """Create full-frame mask from object dictionary."""
        full_mask = np.zeros((64, 64), dtype=bool)
        
        slc = obj_dict["slice"]
        local_mask = obj_dict["mask"]
        
        y_start, y_end = slc[0].start, slc[0].stop
        x_start, x_end = slc[1].start, slc[1].stop
        
        # Ensure bounds
        y_start, y_end = max(0, y_start), min(64, y_end)
        x_start, x_end = max(0, x_start), min(64, x_end)
        
        if y_end > y_start and x_end > x_start:
            local_h, local_w = local_mask.shape
            frame_h = y_end - y_start
            frame_w = x_end - x_start
            
            if local_h == frame_h and local_w == frame_w:
                full_mask[y_start:y_end, x_start:x_end] = local_mask
            else:
                # Fallback: mark entire bounding box
                full_mask[y_start:y_end, x_start:x_end] = True
        
        return full_mask
    
    def update_objects(self, raw_objects: List[dict]) -> Dict[str, PersistentObject]:
        """
        Update object tracking with new frame objects.
        
        Args:
            raw_objects: List of object dictionaries from current frame
            
        Returns:
            Dictionary of persistent objects with stable IDs
        """
        self.current_frame_objects = raw_objects
        self.object_id_mapping = {}
        
        # Create full-frame masks for current objects
        current_masks = []
        for obj in raw_objects:
            full_mask = self._create_full_frame_mask(obj)
            current_masks.append(full_mask)
        
        # Match current objects to persistent objects using IoU
        matched_persistent = set()
        unmatched_current = list(range(len(raw_objects)))
        
        for current_idx, current_mask in enumerate(current_masks):
            best_iou = 0.0
            best_persistent_id = None
            
            # Find best IoU match among unmatched persistent objects
            for persistent_id, persistent_obj in self.persistent_objects.items():
                if persistent_id in matched_persistent:
                    continue
                
                iou = self._calculate_iou(current_mask, persistent_obj.current_mask)
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_persistent_id = persistent_id
            
            if best_persistent_id is not None:
                # Match found - update persistent object
                persistent_obj = self.persistent_objects[best_persistent_id]
                raw_obj = raw_objects[current_idx]
                
                # Update persistent object with new frame data
                persistent_obj.current_mask = current_mask
                persistent_obj.current_slice = raw_obj["slice"]
                persistent_obj.current_color = raw_obj["colour"]
                persistent_obj.area = raw_obj["area"]
                persistent_obj.regularity = raw_obj["regularity"]
                persistent_obj.contrast = raw_obj["contrast"]
                persistent_obj.confidence = raw_obj["confidence"]
                
                # Create mapping
                self.object_id_mapping[current_idx] = best_persistent_id
                matched_persistent.add(best_persistent_id)
                unmatched_current.remove(current_idx)
        
        # Create new persistent objects for unmatched current objects
        for current_idx in unmatched_current:
            if len(self.persistent_objects) < self.max_objects:
                # Create new persistent object
                persistent_id = f"obj_{self.next_object_id}"
                self.next_object_id += 1
                
                raw_obj = raw_objects[current_idx]
                current_mask = current_masks[current_idx]
                
                persistent_obj = PersistentObject(
                    object_id=persistent_id,
                    current_mask=current_mask,
                    current_slice=raw_obj["slice"],
                    current_color=raw_obj["colour"],
                    area=raw_obj["area"],
                    regularity=raw_obj["regularity"],
                    contrast=raw_obj["contrast"],
                    confidence=raw_obj["confidence"]
                )
                
                self.persistent_objects[persistent_id] = persistent_obj
                self.object_id_mapping[current_idx] = persistent_id
        
        # Remove persistent objects that weren't matched (disappeared)
        disappeared_ids = [pid for pid in self.persistent_objects.keys() 
                          if pid not in matched_persistent and 
                          pid not in self.object_id_mapping.values()]
        
        for pid in disappeared_ids:
            del self.persistent_objects[pid]
        
        return self.persistent_objects
    
    def get_persistent_object_by_current_index(self, current_idx: int) -> Optional[PersistentObject]:
        """Get persistent object by current frame index."""
        persistent_id = self.object_id_mapping.get(current_idx)
        if persistent_id:
            return self.persistent_objects.get(persistent_id)
        return None
    
    def get_persistent_object_containing_coord(self, coord: Tuple[int, int]) -> Optional[PersistentObject]:
        """Get persistent object whose mask contains the given coordinate."""
        y, x = coord
        
        for persistent_obj in self.persistent_objects.values():
            if persistent_obj.current_mask[y, x]:
                return persistent_obj
        
        return None
    
    def record_click_attempt(self, coord: Tuple[int, int], success: bool):
        """Record a click attempt on the object containing the coordinate."""
        persistent_obj = self.get_persistent_object_containing_coord(coord)
        
        if persistent_obj is not None:
            persistent_obj.last_click_time = time.time()
            persistent_obj.last_click_coords = coord
            
            if success:
                # Reset stale tries on success
                persistent_obj.stale_tries = 0
                if persistent_obj.status == "PENDING":
                    persistent_obj.status = "CONFIRMED"
            else:
                # Increment stale tries on failure
                persistent_obj.stale_tries += 1
                
                # Mark as FAILED after too many stale tries
                if persistent_obj.stale_tries >= 2:  # K=2
                    persistent_obj.status = "FAILED"
    
    def get_testable_hypotheses(self) -> List[PersistentObject]:
        """Get objects that can be tested (not FAILED)."""
        return [obj for obj in self.persistent_objects.values() 
                if obj.status in ["PENDING", "TESTING", "CONFIRMED"]]
    
    def get_stats(self) -> Dict[str, any]:
        """Get tracker statistics."""
        status_counts = {}
        for obj in self.persistent_objects.values():
            status_counts[obj.status] = status_counts.get(obj.status, 0) + 1
        
        return {
            "total_persistent_objects": len(self.persistent_objects),
            "current_frame_objects": len(self.current_frame_objects),
            "mapping_size": len(self.object_id_mapping),
            "status_counts": status_counts,
            "iou_threshold": self.iou_threshold,
            "next_object_id": self.next_object_id
        }
