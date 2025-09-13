"""
ReCoN Hypothesis Nodes for Active Perception

Implements hypothesis testing nodes that can request confirmation
through action execution and build hierarchical sequences.
"""

import sys
import os
sys.path.append('/workspace/recon-platform')

from typing import Dict, List, Optional, Any, Tuple
from recon_engine.node import ReCoNNode, ReCoNState
from recon_engine.messages import ReCoNMessage, MessageType
from recon_engine.graph import ReCoNGraph
import numpy as np


class ActionHypothesis(ReCoNNode):
    """
    ReCoN node representing a hypothesis about an action.

    Hypothesis: "Action X will be productive in this context"
    """

    def __init__(self, node_id: str, action_idx: int, predicted_change_prob: float):
        super().__init__(node_id, "script")  # Script node for hypothesis testing

        self.action_idx = action_idx
        self.predicted_change_prob = predicted_change_prob

        # Hypothesis testing state
        self.tested = False
        self.actually_changed = False
        self.confirmation_count = 0
        self.failure_count = 0

        # Context when hypothesis was formed
        self.context_frame = None

    def process_message(self, message: ReCoNMessage) -> List[ReCoNMessage]:
        """
        Process incoming messages for hypothesis testing.

        Request -> Test the hypothesis
        Confirm -> Hypothesis was correct
        Fail -> Hypothesis was wrong
        """
        responses = []

        if message.type == MessageType.REQUEST:
            # Hypothesis is being tested
            self.state = ReCoNState.REQUESTED
            # Send request to execute the action (hierarchical sub relationship)
            responses.append(ReCoNMessage(MessageType.REQUEST, self.id, "action_executor", "sub"))

        elif message.type == MessageType.CONFIRM:
            # Action execution succeeded (frame changed)
            self.state = ReCoNState.CONFIRMED
            self.tested = True
            self.actually_changed = True
            self.confirmation_count += 1

        elif message.type == MessageType.FAIL:
            # Action execution failed (no frame change)
            self.state = ReCoNState.FAILED
            self.tested = True
            self.actually_changed = False
            self.failure_count += 1

        elif message.type == MessageType.INHIBIT_REQUEST:
            # Stop testing this hypothesis
            self.state = ReCoNState.INACTIVE

        return responses

    def get_confidence(self) -> float:
        """
        Get confidence in this hypothesis based on testing history.

        Returns:
            confidence: Value between 0 and 1
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
    ReCoN node representing a sequence of actions.

    Hypothesis: "This sequence of actions leads to progress"
    """

    def __init__(self, node_id: str, action_sequence: List[int]):
        super().__init__(node_id, "script")

        self.action_sequence = action_sequence
        self.current_step = 0
        self.completed_successfully = False

        # Child action hypotheses
        self.action_hypotheses = []

    def add_action_hypothesis(self, hypothesis: ActionHypothesis):
        """Add an action hypothesis to this sequence."""
        self.action_hypotheses.append(hypothesis)

    def process_message(self, message: ReCoNMessage) -> List[ReCoNMessage]:
        """
        Process messages for sequence execution.
        """
        responses = []

        if message.type == MessageType.REQUEST:
            # Start executing the sequence
            self.state = ReCoNState.REQUESTED
            self.current_step = 0

            if self.action_hypotheses:
                # Request first action in sequence (sequential por relationship)
                first_hypothesis = self.action_hypotheses[0]
                responses.append(ReCoNMessage(MessageType.REQUEST, self.id, first_hypothesis.id, "por"))

        elif message.type == MessageType.CONFIRM:
            # Current step succeeded, move to next
            self.current_step += 1

            if self.current_step >= len(self.action_hypotheses):
                # Sequence completed successfully
                self.state = ReCoNState.CONFIRMED
                self.completed_successfully = True
            else:
                # Request next step (sequential por relationship)
                next_hypothesis = self.action_hypotheses[self.current_step]
                responses.append(ReCoNMessage(MessageType.REQUEST, self.id, next_hypothesis.id, "por"))

        elif message.type == MessageType.FAIL:
            # Sequence failed
            self.state = ReCoNState.FAILED

        return responses


class HypothesisManager:
    """
    Manages the creation and testing of hypotheses using ReCoN graph.
    """

    def __init__(self):
        self.graph = ReCoNGraph()
        self.action_hypotheses = {}  # action_idx -> ActionHypothesis
        self.sequence_hypotheses = []
        self.hypothesis_counter = 0

        # Build basic architecture
        self._build_architecture()

    def _build_architecture(self):
        """Build ReCoN architecture for hypothesis testing."""
        # Root hypothesis controller
        root = self.graph.add_node("hypothesis_root", "script")

        # Action executor (interface to actual game actions)
        executor = self.graph.add_node("action_executor", "script")

        # Connect root to executor
        self.graph.add_link("hypothesis_root", "action_executor", "sub")

    def create_action_hypothesis(self, action_idx: int, predicted_prob: float,
                                context_frame: np.ndarray) -> ActionHypothesis:
        """
        Create a new action hypothesis.

        Args:
            action_idx: Action to test (0-5)
            predicted_prob: CNN predicted change probability
            context_frame: Frame context when hypothesis formed

        Returns:
            hypothesis: New ActionHypothesis node
        """
        self.hypothesis_counter += 1
        node_id = f"action_hyp_{self.hypothesis_counter}"

        hypothesis = ActionHypothesis(node_id, action_idx, predicted_prob)
        hypothesis.context_frame = context_frame.copy()

        # Add to graph
        self.graph.add_node(hypothesis)

        # Connect to executor
        self.graph.add_link("action_executor", node_id, "sub")

        # Store reference
        self.action_hypotheses[action_idx] = hypothesis

        return hypothesis

    def create_sequence_hypothesis(self, action_sequence: List[int]) -> SequenceHypothesis:
        """
        Create a hypothesis about a sequence of actions.

        Args:
            action_sequence: List of action indices

        Returns:
            hypothesis: New SequenceHypothesis node
        """
        self.hypothesis_counter += 1
        node_id = f"seq_hyp_{self.hypothesis_counter}"

        hypothesis = SequenceHypothesis(node_id, action_sequence)

        # Create action hypotheses for each step if they don't exist
        for action_idx in action_sequence:
            if action_idx not in self.action_hypotheses:
                # Create with neutral probability since it's part of a sequence
                action_hyp = self.create_action_hypothesis(action_idx, 0.5, np.zeros((64, 64)))
            else:
                action_hyp = self.action_hypotheses[action_idx]

            hypothesis.add_action_hypothesis(action_hyp)

        # Add to graph
        self.graph.add_node(hypothesis)

        # Connect to root
        self.graph.add_link("hypothesis_root", node_id, "sub")

        # Store reference
        self.sequence_hypotheses.append(hypothesis)

        return hypothesis

    def request_hypothesis_test(self, hypothesis_id: str):
        """Request testing of a specific hypothesis."""
        if hypothesis_id in self.graph.nodes:
            self.graph.request_root(hypothesis_id)

    def get_best_action_hypothesis(self) -> Optional[ActionHypothesis]:
        """
        Get the action hypothesis with highest confidence.

        Returns:
            hypothesis: Best ActionHypothesis or None
        """
        if not self.action_hypotheses:
            return None

        best_hypothesis = None
        best_confidence = 0.0

        for hypothesis in self.action_hypotheses.values():
            confidence = hypothesis.get_confidence()
            if confidence > best_confidence:
                best_confidence = confidence
                best_hypothesis = hypothesis

        return best_hypothesis

    def update_hypothesis_result(self, action_idx: int, frame_changed: bool):
        """
        Update hypothesis with actual result.

        Args:
            action_idx: Action that was executed
            frame_changed: Whether the frame actually changed
        """
        if action_idx in self.action_hypotheses:
            hypothesis = self.action_hypotheses[action_idx]

            if frame_changed:
                hypothesis.process_message(ReCoNMessage(MessageType.CONFIRM, "executor", hypothesis.id, "sur"))
            else:
                hypothesis.process_message(ReCoNMessage(MessageType.FAIL, "executor", hypothesis.id, "sur"))

    def propagate_step(self):
        """Perform one step of ReCoN message propagation."""
        self.graph.propagate_step()

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about current hypotheses."""
        info = {
            'num_action_hypotheses': len(self.action_hypotheses),
            'num_sequence_hypotheses': len(self.sequence_hypotheses),
            'action_confidences': {}
        }

        for action_idx, hypothesis in self.action_hypotheses.items():
            info['action_confidences'][action_idx] = {
                'confidence': hypothesis.get_confidence(),
                'tested': hypothesis.tested,
                'confirmations': hypothesis.confirmation_count,
                'failures': hypothesis.failure_count
            }

        return info