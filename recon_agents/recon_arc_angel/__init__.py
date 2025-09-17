"""
ReCoN ARC Angel Agent

Key components:
- CNNValidActionTerminal for action/coordinate prediction
- ResNetActionValueTerminal for object segmentation
- Hierarchical coordinate refinement (8x8 -> 64x64) 
- Deduplicated experience buffer with frame change detection
- Pure ReCoN execution with continuous sur magnitudes
- GPU acceleration and dual training
"""

from .learning_manager import LearningManager
from .improved_hierarchy_manager import ImprovedHierarchicalHypothesisManager
from .improved_production_agent import ImprovedProductionReCoNArcAngel

# Export production-ready agent and components that still exist
__all__ = [
    'ImprovedProductionReCoNArcAngel', 
    'ImprovedHierarchicalHypothesisManager', 
    'LearningManager', 
]
