"""
ReCoN ARC Angel Agent

A TDD implementation of the StochasticGoose parity agent using ReCoN principles.
This agent extends the paper's active perception approach to reproduce StochasticGoose
performance on ARC-AGI tasks using pure ReCoN message passing.

Key components:
- CNNValidActionTerminal for action/coordinate prediction
- Hierarchical coordinate refinement (8x8 -> 64x64) 
- Deduplicated experience buffer with frame change detection
- Pure ReCoN execution with continuous sur magnitudes
"""

from .agent import ReCoNArcAngel
from .hypothesis_manager import HypothesisManager
from .learning_manager import LearningManager
from .region_aggregator import RegionAggregator
from .efficient_hierarchy_manager import EfficientHierarchicalHypothesisManager

# Export production-ready efficient implementation (primary) and working reference implementations
__all__ = [
    'EfficientHierarchicalHypothesisManager',  # 🚀 PRIMARY: REFINED_PLAN + BlindSquirrel efficiency
    'ReCoNArcAngel',  # Working simplified agent (reference baseline)
    'HypothesisManager',  # Working simplified manager (reference baseline)
    'LearningManager',  # Dual training component (CNN + ResNet)
    'RegionAggregator',  # Working region aggregation (reference baseline)
]
