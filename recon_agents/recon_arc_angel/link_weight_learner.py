"""
EXTENSION 1B: Link Weight Learning from Outcomes

Learns ReCoN link weights based on action outcomes, enabling
bidirectional neural-symbolic feedback.
"""

import numpy as np
from typing import Dict, Tuple, List


class LinkWeightLearner:
    """
    Learn link weights from action outcomes via exponential moving average.

    Blends learned weights with CNN priors for improved action selection.
    """

    def __init__(self, learning_rate: float = 0.1, blend_ratio: float = 0.3):
        """
        Args:
            learning_rate: How fast weights adapt to outcomes (0.0-1.0)
            blend_ratio: How much to trust learned vs CNN (0.0=all CNN, 1.0=all learned)
        """
        self.lr = learning_rate
        self.blend_ratio = blend_ratio

        # Track: (source, target, link_type) -> learned weight
        self.learned_weights: Dict[Tuple[str, str, str], float] = {}

        # Track outcomes for metrics
        self.outcome_history: Dict[Tuple[str, str, str], List[float]] = {}
        self.total_updates = 0

    def get_blended_weight(self, source: str, target: str, link_type: str,
                          cnn_prior: float) -> float:
        """
        Get weight blending learned experience with CNN prior.

        Returns:
            Blended weight: (1-α)·CNN + α·learned, where α is blend_ratio
        """
        key = (source, target, link_type)
        learned = self.learned_weights.get(key, 0.5)  # Neutral default

        # Blend: more CNN at first, more learned over time
        return (1 - self.blend_ratio) * cnn_prior + self.blend_ratio * learned

    def update_from_outcome(self, source: str, target: str, link_type: str,
                           outcome: float) -> float:
        """
        Update link weight based on outcome.

        Args:
            source: Source node ID
            target: Target node ID
            link_type: Link type (typically "sub")
            outcome: Reward (1.0=success, 0.0=failure)

        Returns:
            New learned weight
        """
        key = (source, target, link_type)
        current = self.learned_weights.get(key, 0.5)

        # EMA update: move toward 1.0 if success, 0.0 if failure
        if outcome > 0:
            target_weight = 1.0
        else:
            target_weight = 0.0

        new_weight = current + self.lr * (target_weight - current)
        self.learned_weights[key] = np.clip(new_weight, 0.0, 1.0)

        # Track history
        if key not in self.outcome_history:
            self.outcome_history[key] = []
        self.outcome_history[key].append(outcome)
        self.total_updates += 1

        return self.learned_weights[key]

    def get_metrics(self) -> Dict:
        """
        Get learning metrics for visualization.

        Returns:
            Dict with learning statistics
        """
        if not self.learned_weights:
            return {
                "total_links_learned": 0,
                "total_updates": 0,
                "top_learned_links": [],
                "avg_improvement": 0.0
            }

        # Find most confident learned links (furthest from 0.5)
        sorted_links = sorted(
            self.learned_weights.items(),
            key=lambda x: abs(x[1] - 0.5),
            reverse=True
        )

        top_links = []
        for key, weight in sorted_links[:5]:
            source, target, link_type = key
            history = self.outcome_history.get(key, [])
            success_rate = np.mean(history) if history else 0.0

            top_links.append({
                "link": f"{source} → {target}",
                "learned_weight": weight,
                "success_rate": success_rate,
                "n_updates": len(history)
            })

        # Compute average improvement
        all_success_rates = [
            np.mean(hist) for hist in self.outcome_history.values() if hist
        ]
        avg_improvement = np.mean(all_success_rates) if all_success_rates else 0.0

        return {
            "total_links_learned": len(self.learned_weights),
            "total_updates": self.total_updates,
            "top_learned_links": top_links,
            "avg_improvement": avg_improvement
        }

    def reset(self):
        """Reset all learned weights."""
        self.learned_weights.clear()
        self.outcome_history.clear()
        self.total_updates = 0
