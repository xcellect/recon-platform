"""
Pure ReCoN Hypothesis Implementation - No Manual State Control

This implementation strictly follows ReCoN principles:
1. Single root request propagates through sub links automatically
2. States emerge from message passing (Table 1), not manual setting
3. Por/ret links provide natural inhibition
4. Gen loops provide persistence for failed states (cooldown)
5. CNN priors modulate flow via link weights, not manual control
6. No Python-side state management
"""

import sys
import os
sys.path.append('/workspace/recon-platform')

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from recon_engine.node import ReCoNNode, ReCoNState
from recon_engine.messages import ReCoNMessage, MessageType
from recon_engine.graph import ReCoNGraph
import numpy as np
import torch


class ActionHypothesis(ReCoNNode):
    """
    Pure ReCoN action hypothesis node.

    Hypothesis: "Action X will be productive in this context"

    No manual state control - only ReCoN message passing.
    """

    def __init__(self, node_id: str, action_idx: int, predicted_change_prob: float):
        super().__init__(node_id, "script")  # Script node for hypothesis testing

        self.action_idx = action_idx
        self.predicted_change_prob = predicted_change_prob

        # Hypothesis testing state (read-only)
        self.tested = False
        self.actually_changed = False
        self.confirmation_count = 0
        self.failure_count = 0

        # Context when hypothesis was formed
        self.context_frame = None
        # Terminal child id (set by manager)
        self.terminal_id: Optional[str] = None

    def get_confidence(self) -> float:
        """
        Get confidence in this hypothesis based on testing history.
        """
        if not self.tested:
            return self.predicted_change_prob  # CNN prediction

        total_tests = self.confirmation_count + self.failure_count
        if total_tests == 0:
            return self.predicted_change_prob

        # Combine CNN prediction with empirical results
        empirical_rate = self.confirmation_count / total_tests
        # Weight empirical evidence more as we get more data
        weight = min(total_tests / 10.0, 0.8)
        return weight * empirical_rate + (1 - weight) * self.predicted_change_prob


class SequenceHypothesis(ReCoNNode):
    """
    Pure ReCoN sequence hypothesis node.

    Hypothesis: "This sequence of actions leads to progress"
    """

    def __init__(self, node_id: str, action_sequence: List[int]):
        super().__init__(node_id, "script")

        self.action_sequence = action_sequence
        self.current_step = 0
        self.completed_successfully = False

        # Child action hypotheses (read-only references)
        self.action_hypotheses = []

    def add_action_hypothesis(self, hypothesis: ActionHypothesis):
        """Add an action hypothesis reference (no graph modification)."""
        self.action_hypotheses.append(hypothesis)


class PureReCoNHypothesisManager:
    """
    Pure ReCoN Hypothesis Manager - No Manual State Control

    Manages hypothesis testing using ONLY ReCoN message passing:
    - Single root request at hypothesis_root
    - States emerge from message flow (Table 1)
    - CNN priors via link weights
    - Cooldown via gen loops
    - No Python-side state management
    """

    def __init__(self):
        self.graph = ReCoNGraph()
        self.action_hypotheses = {}  # action_idx -> ActionHypothesis
        self.sequence_hypotheses = []
        self.hypothesis_counter = 0

        # CNN priors (affect link weights only)
        self.alpha_valid: Dict[int, float] = {}
        self.alpha_value: Dict[int, float] = {}

        # Available actions filter
        self._available_actions: Optional[Set[int]] = None

        # Map action_idx to terminal node id
        self._action_to_terminal: Dict[int, str] = {}

        # Build basic pure ReCoN architecture
        self._build_pure_recon_architecture()

    def _build_pure_recon_architecture(self):
        """Build pure ReCoN architecture - single root with natural flow."""
        # Single hypothesis root - the only node we request
        if "hypothesis_root" not in self.graph.nodes:
            self.graph.add_node("hypothesis_root", "script")

    def create_action_hypothesis(self, action_idx: int, predicted_prob: float,
                                 context_frame: Optional[np.ndarray]) -> ActionHypothesis:
        """
        Create action hypothesis and connect to pure ReCoN graph.
        """
        self.hypothesis_counter += 1
        node_id = f"action_hyp_{self.hypothesis_counter}"

        hypothesis = ActionHypothesis(node_id, action_idx, predicted_prob)
        if context_frame is not None:
            hypothesis.context_frame = context_frame.copy()

        # Add to graph
        self.graph.add_node(hypothesis)

        # Create terminal measurement node
        term_id = f"{node_id}_term"
        terminal = TerminalMeasurementNode(term_id)
        self.graph.add_node(terminal)

        # Connect via sub link (natural parent-child relationship)
        self.graph.add_link(node_id, term_id, "sub")
        hypothesis.terminal_id = term_id

        # Connect to hypothesis root - all actions are alternatives
        self.graph.add_link("hypothesis_root", node_id, "sub")

        # Add gen loop for persistent failed states (natural cooldown)
        self.graph.add_link(node_id, node_id, "gen")

        # Store references
        self.action_hypotheses[action_idx] = hypothesis
        self._action_to_terminal[action_idx] = term_id

        return hypothesis

    def create_sequence_hypothesis(self, action_sequence: List[int]) -> SequenceHypothesis:
        """Create sequence hypothesis with por/ret ordering."""
        self.hypothesis_counter += 1
        node_id = f"seq_hyp_{self.hypothesis_counter}"

        hypothesis = SequenceHypothesis(node_id, action_sequence)

        # Create action hypotheses if needed
        for idx in action_sequence:
            if idx not in self.action_hypotheses:
                ah = self.create_action_hypothesis(idx, 0.5, np.zeros((64, 64)))
            else:
                ah = self.action_hypotheses[idx]
            hypothesis.add_action_hypothesis(ah)

        # Add to graph
        self.graph.add_node(hypothesis)

        # Connect to root
        self.graph.add_link("hypothesis_root", node_id, "sub")

        # Connect sequence to each action
        for act_h in hypothesis.action_hypotheses:
            self.graph.add_link(node_id, act_h.id, "sub")

        # Create por/ret chain for natural ordering
        for i in range(len(hypothesis.action_hypotheses) - 1):
            prev_id = hypothesis.action_hypotheses[i].id
            next_id = hypothesis.action_hypotheses[i + 1].id
            # Por: previous inhibits next until complete
            self.graph.add_link(prev_id, next_id, "por")
            # Ret: next inhibits previous confirmation
            self.graph.add_link(next_id, prev_id, "ret")

        self.sequence_hypotheses.append(hypothesis)
        return hypothesis

    def create_alternatives_hypothesis(self, action_indices: List[int]) -> str:
        """Create alternatives with por ordering based on α_value."""
        self.hypothesis_counter += 1
        node_id = f"alt_hyp_{self.hypothesis_counter}"
        self.graph.add_node(node_id, "script")

        # Connect to root
        self.graph.add_link("hypothesis_root", node_id, "sub")

        # Create action hypotheses if needed
        children = []
        for idx in action_indices:
            if idx not in self.action_hypotheses:
                ah = self.create_action_hypothesis(idx, 0.5, np.zeros((64, 64)))
            else:
                ah = self.action_hypotheses[idx]
            children.append(ah)

        # Sort by α_value for por ordering (high value inhibits low value)
        scored = sorted(children, key=lambda n: self.alpha_value.get(n.action_idx, 0.0), reverse=True)

        # Connect all as alternatives
        for child in scored:
            # Sub link weighted by α_valid
            alpha_valid = self.alpha_valid.get(child.action_idx, 1.0)
            self.graph.add_link(node_id, child.id, "sub", weight=alpha_valid)

            # Sur link weighted by α_value
            alpha_value = self.alpha_value.get(child.action_idx, 1.0)
            self.graph.add_link(child.id, node_id, "sur", weight=alpha_value)

        # Add por inhibition: higher α_value inhibits lower α_value
        for i in range(len(scored) - 1):
            for j in range(i + 1, len(scored)):
                high_child = scored[i]
                low_child = scored[j]
                # Higher value inhibits lower value via por
                self.graph.add_link(high_child.id, low_child.id, "por")

        return node_id

    def set_available_actions(self, allowed_action_indices: Optional[List[int]]):
        """Set available actions filter."""
        if allowed_action_indices is None:
            self._available_actions = None
        else:
            self._available_actions = set(allowed_action_indices)

    def set_alpha_valid(self, mapping: Dict[int, float]) -> None:
        """Set α_valid priors - affects sub link weights for request delay."""
        self.alpha_valid.update(mapping)

        # Update existing sub link weights
        for action_idx, alpha in mapping.items():
            hypothesis = self.action_hypotheses.get(action_idx)
            if hypothesis:
                # Find sub links to this hypothesis and update weights
                for link in self.graph.links:
                    if link.target == hypothesis.id and link.type == "sub":
                        link.weight = alpha

    def set_alpha_value(self, mapping: Dict[int, float]) -> None:
        """Set α_value priors - affects sur link weights and por ordering."""
        self.alpha_value.update(mapping)

        # Update existing sur link weights
        for action_idx, alpha in mapping.items():
            hypothesis = self.action_hypotheses.get(action_idx)
            if hypothesis:
                # Find sur links from this hypothesis and update weights
                for link in self.graph.links:
                    if link.source == hypothesis.id and link.type == "sur":
                        link.weight = alpha

    def feed_cnn_priors(self, valid_probs: Dict[int, float], value_probs: Dict[int, float]) -> None:
        """
        Feed CNN priors - affects link weights only, no manual state control.
        """
        self.set_alpha_valid(valid_probs)
        self.set_alpha_value(value_probs)

    def request_hypothesis_test(self, hypothesis_id: str = "hypothesis_root"):
        """Request hypothesis testing - single root request only."""
        if hypothesis_id in self.graph.nodes:
            self.graph.request_root(hypothesis_id)

    def set_terminal_measurement(self, action_idx: int, changed: bool) -> None:
        """Set terminal measurement - triggers natural ReCoN flow."""
        term_id = self._action_to_terminal.get(action_idx)
        if term_id:
            terminal = self.graph.get_node(term_id)
            if isinstance(terminal, TerminalMeasurementNode):
                terminal.set_measurement(bool(changed))

        # Update hypothesis statistics (read-only tracking)
        hypothesis = self.action_hypotheses.get(action_idx)
        if hypothesis:
            hypothesis.tested = True
            hypothesis.actually_changed = changed
            if changed:
                hypothesis.confirmation_count += 1
            else:
                hypothesis.failure_count += 1

    def propagate_step(self):
        """Propagate one step - pure ReCoN message passing only."""
        self.graph.propagate_step()

        # No manual state control - states emerge from message passing

    def get_best_action_hypothesis(self) -> Optional[ActionHypothesis]:
        """Get action hypothesis with highest confidence."""
        if not self.action_hypotheses:
            return None

        best = None
        best_conf = -1.0
        for hyp in self.action_hypotheses.values():
            conf = hyp.get_confidence()
            if conf > best_conf:
                best_conf = conf
                best = hyp
        return best

    def update_hypothesis_result(self, action_idx: int, frame_changed: bool) -> None:
        """Update hypothesis result via terminal measurement."""
        self.set_terminal_measurement(action_idx, bool(frame_changed))


class TerminalMeasurementNode(ReCoNNode):
    """
    Terminal node that performs measurements when requested.
    Pure ReCoN - no manual state control.
    """
    def __init__(self, node_id: str):
        super().__init__(node_id, "terminal")
        self._measurement: Optional[bool] = None

    def set_measurement(self, changed: Optional[bool]):
        """Set measurement result."""
        self._measurement = changed

    def measure(self) -> float:
        """Return measurement result."""
        if self._measurement is True:
            return 1.0
        elif self._measurement is False:
            return 0.0
        return 0.0


# Compatibility alias - allows existing code to work with pure implementation
HypothesisManager = PureReCoNHypothesisManager