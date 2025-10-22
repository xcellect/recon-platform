"""
Global Workspace Theory Implementation for ReCoN Graphs

Implements Global Workspace Theory (GWT) for action selection:
- Winner-take-all broadcasting
- Lateral inhibition of competitors
- Clearer decision boundaries
- Reduces action oscillation

Based on Baars' Global Workspace Theory and winner-take-all neural networks.

References:
- Baars, B. J. (1988). A cognitive theory of consciousness.
- Dehaene, S., & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing.
"""

from typing import List, Tuple, Optional, Any, Dict
import torch


class GlobalWorkspace:
    """
    Global Workspace Theory implementation for ReCoN graphs.

    Implements winner-take-all broadcasting where the winning hypothesis
    broadcasts its activation to suppress competitors, creating clearer
    decision boundaries and reducing indecision.

    This aligns with consciousness theories where a dominant mental state
    gains "global access" and inhibits competing states.
    """

    def __init__(self, broadcast_strength: float = 0.3, winner_boost: float = 0.2):
        """
        Initialize the Global Workspace.

        Args:
            broadcast_strength: How much the winner suppresses competitors (0-1)
                               0.3 = 30% suppression based on winner's strength
            winner_boost: How much to boost the winner's activation (0-1)
                         0.2 = 20% activation increase
        """
        self.broadcast_strength = broadcast_strength
        self.winner_boost = winner_boost

        # Statistics for tracking GWT effects
        self.stats = {
            'total_broadcasts': 0,
            'total_winner_boost': 0.0,
            'total_loser_suppression': 0.0,
            'avg_winner_boost': 0.0,
            'avg_loser_suppression': 0.0,
            'avg_decision_confidence': 0.0,
            'last_winner_score': 0.0,
            'last_runnerup_score': 0.0
        }

    def broadcast_winner(self, winner_idx: int, candidates: List[Tuple],
                        graph: Any) -> None:
        """
        Broadcast winning action to suppress competitors (Global Workspace).

        Implements lateral inhibition: the winning hypothesis gains global access
        and suppresses non-winning hypotheses, creating clearer decision boundaries.

        Args:
            winner_idx: Index of winning action in candidates list
            candidates: List of (action_id, score, coords, obj_idx) tuples
            graph: ReCoN graph to modify (ReCoNGraph or CompactReCoNGraph)
        """
        if not candidates or winner_idx >= len(candidates):
            return

        winner_action, winner_score, _, _ = candidates[winner_idx]

        # Track winner score for metrics
        self.stats['last_winner_score'] = float(winner_score)

        # Find runner-up score for decision confidence metric
        runner_up_score = 0.0
        for idx, (_, score, _, _) in enumerate(candidates):
            if idx != winner_idx:
                runner_up_score = max(runner_up_score, float(score))

        self.stats['last_runnerup_score'] = runner_up_score
        decision_confidence = winner_score - runner_up_score
        self.stats['avg_decision_confidence'] = (
            (self.stats['avg_decision_confidence'] * self.stats['total_broadcasts'] +
             decision_confidence) / (self.stats['total_broadcasts'] + 1)
            if self.stats['total_broadcasts'] > 0 else decision_confidence
        )

        # Broadcast: suppress all non-winners based on winner strength
        total_suppression = 0.0
        num_losers = 0

        for idx, (action_id, score, _, _) in enumerate(candidates):
            if idx == winner_idx:
                continue  # Don't suppress the winner

            if action_id in graph.nodes:
                node = graph.nodes[action_id]

                # Calculate lateral inhibition strength
                # Stronger winners suppress more (biologically plausible)
                suppression_factor = self.broadcast_strength * float(winner_score)

                # Apply suppression to activation (if node has activation attribute)
                if hasattr(node, 'activation'):
                    old_activation = float(node.activation)
                    # Reduce activation but keep it non-negative
                    new_activation = max(0.0, old_activation * (1.0 - suppression_factor))
                    node.activation = new_activation

                    # Track suppression for metrics
                    suppression_amount = old_activation - new_activation
                    total_suppression += suppression_amount
                    num_losers += 1

        # Optionally boost the winner (positive reinforcement)
        winner_boost_amount = 0.0
        if winner_action in graph.nodes:
            winner_node = graph.nodes[winner_action]
            if hasattr(winner_node, 'activation'):
                old_activation = float(winner_node.activation)
                # Boost activation but cap at 1.0
                new_activation = min(1.0, old_activation * (1.0 + self.winner_boost))
                winner_node.activation = new_activation
                winner_boost_amount = new_activation - old_activation

        # Update running statistics
        self.stats['total_broadcasts'] += 1
        self.stats['total_winner_boost'] += winner_boost_amount
        self.stats['total_loser_suppression'] += total_suppression

        # Update averages
        if self.stats['total_broadcasts'] > 0:
            self.stats['avg_winner_boost'] = (
                self.stats['total_winner_boost'] / self.stats['total_broadcasts']
            )

        if num_losers > 0:
            self.stats['avg_loser_suppression'] = (
                total_suppression / num_losers
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get Global Workspace statistics"""
        return {
            **self.stats,
            'broadcast_strength': self.broadcast_strength,
            'winner_boost': self.winner_boost,
            'enabled': True
        }

    def reset_stats(self):
        """Reset statistics (useful for per-game tracking)"""
        self.stats = {
            'total_broadcasts': 0,
            'total_winner_boost': 0.0,
            'total_loser_suppression': 0.0,
            'avg_winner_boost': 0.0,
            'avg_loser_suppression': 0.0,
            'avg_decision_confidence': 0.0,
            'last_winner_score': 0.0,
            'last_runnerup_score': 0.0
        }
