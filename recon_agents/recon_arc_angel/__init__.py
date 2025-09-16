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
from .production_agent import ProductionReCoNArcAngel

# Export production-ready agent and components, plus reference implementations
__all__ = [
    'ProductionReCoNArcAngel',  # 🚀 MAIN: Complete production agent with dual training
    'EfficientHierarchicalHypothesisManager',  # 🚀 PRIMARY: REFINED_PLAN + BlindSquirrel efficiency
    'LearningManager',  # 🎓 Dual training component (CNN + ResNet)
    'ReCoNArcAngel',  # 📚 Reference: Simplified agent baseline
    'HypothesisManager',  # 📚 Reference: Simplified manager baseline
    'RegionAggregator',  # 📚 Reference: Region aggregation baseline
]
