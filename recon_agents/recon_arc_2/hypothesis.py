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
        # Terminal child id (set by manager)
        self.terminal_id: Optional[str] = None

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
            # Send request to terminal measurement if available; otherwise to executor
            target = self.terminal_id if self.terminal_id else "action_executor"
            responses.append(ReCoNMessage(MessageType.REQUEST, self.id, target, "sub"))

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
                # Request first action in sequence 
                first_hypothesis = self.action_hypotheses[0]
                responses.append(ReCoNMessage(MessageType.REQUEST, self.id, first_hypothesis.id, "sub"))

        elif message.type == MessageType.CONFIRM:
            # Current step succeeded, move to next
            self.current_step += 1

            if self.current_step >= len(self.action_hypotheses):
                # Sequence completed successfully
                self.state = ReCoNState.CONFIRMED
                self.completed_successfully = True
            else:
                # Request next step 
                next_hypothesis = self.action_hypotheses[self.current_step]
                responses.append(ReCoNMessage(MessageType.REQUEST, self.id, next_hypothesis.id, "sub"))

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

        # Pending measurement results per action index (True=confirm, False=fail)
        self._pending_measurements: Dict[int, bool] = {}

        # Map action_idx to (sequence_obj, position, last_index) for quick updates
        self._action_to_sequence: Dict[int, Tuple["SequenceHypothesis", int, int]] = {}
        # Map action_idx to terminal node id
        self._action_to_terminal: Dict[int, str] = {}

        # Build basic architecture
        self._build_architecture()

    def _build_architecture(self):
        """Build ReCoN architecture for hypothesis testing."""
        # Root hypothesis controller
        if "hypothesis_root" not in self.graph.nodes:
            self.graph.add_node("hypothesis_root", "script")

        # Action executor (interface to actual game actions)
        if "action_executor" not in self.graph.nodes:
            self.graph.add_node("action_executor", "script")

        # Connect root to executor if not already
        try:
            self.graph.add_link("hypothesis_root", "action_executor", "sub")
        except Exception:
            pass

    def create_action_hypothesis(self, action_idx: int, predicted_prob: float,
                                 context_frame: np.ndarray) -> ActionHypothesis:
        """
        Create a new action hypothesis and its terminal measurement child.
        """
        self.hypothesis_counter += 1
        node_id = f"action_hyp_{self.hypothesis_counter}"

        hypothesis = ActionHypothesis(node_id, action_idx, predicted_prob)
        hypothesis.context_frame = context_frame.copy()

        # Add to graph
        self.graph.add_node(hypothesis)

        # Create terminal measurement node and connect as child
        term_id = f"{node_id}_term"
        terminal = TerminalMeasurementNode(term_id)
        self.graph.add_node(terminal)
        self.graph.add_link(node_id, term_id, "sub")
        hypothesis.terminal_id = term_id

        # Connect to executor
        try:
            self.graph.add_link("action_executor", node_id, "sub")
        except Exception:
            pass

        # Store reference
        self.action_hypotheses[action_idx] = hypothesis

        # Map action to terminal id
        self._action_to_terminal[action_idx] = term_id

        return hypothesis

    def create_sequence_hypothesis(self, action_sequence: List[int]) -> SequenceHypothesis:
        """Create a hypothesis about a sequence of actions."""
        self.hypothesis_counter += 1
        node_id = f"seq_hyp_{self.hypothesis_counter}"

        hypothesis = SequenceHypothesis(node_id, action_sequence)

        # Create action hypotheses for each step if they don't exist
        for idx in action_sequence:
            if idx not in self.action_hypotheses:
                ah = self.create_action_hypothesis(idx, 0.5, np.zeros((64, 64)))
            else:
                ah = self.action_hypotheses[idx]
            hypothesis.add_action_hypothesis(ah)

        # Add to graph and connect to root
        self.graph.add_node(hypothesis)
        try:
            self.graph.add_link("hypothesis_root", node_id, "sub")
        except Exception:
            pass

        # Wire por/ret chain across steps
        for i in range(len(hypothesis.action_hypotheses) - 1):
            prev_id = hypothesis.action_hypotheses[i].id
            next_id = hypothesis.action_hypotheses[i + 1].id
            try:
                self.graph.add_link(prev_id, next_id, "por")
                self.graph.add_link(next_id, prev_id, "ret")
            except Exception:
                pass

        # Track mapping from action indices to this sequence
        last_index = len(hypothesis.action_hypotheses) - 1
        for pos, act_h in enumerate(hypothesis.action_hypotheses):
            for idx, ref in self.action_hypotheses.items():
                if ref.id == act_h.id:
                    self._action_to_sequence[idx] = (hypothesis, pos, last_index)
                    break

        self.sequence_hypotheses.append(hypothesis)
        return hypothesis

    def request_hypothesis_test(self, hypothesis_id: str):
        if hypothesis_id in self.graph.nodes:
            self.graph.request_root(hypothesis_id)

    def set_terminal_measurement(self, action_idx: int, changed: bool) -> None:
        term_id = self._action_to_terminal.get(action_idx)
        if not term_id:
            return
        node = self.graph.nodes.get(term_id)
        if isinstance(node, TerminalMeasurementNode):
            node.set_measurement(bool(changed))

    def propagate_step(self):
        """Propagate one step in the ReCoN graph (terminals will emit sur when requested)."""
        self.graph.propagate_step()


class TerminalMeasurementNode(ReCoNNode):
    """
    Terminal node that performs a measurement when requested.
    Measurement is set externally via set_measurement(True/False).
    """
    def __init__(self, node_id: str):
        super().__init__(node_id, "terminal")
        self._measurement: Optional[bool] = None

    def set_measurement(self, changed: Optional[bool]):
        self._measurement = changed

    def measure(self) -> float:
        # Return 1.0 for change, 0.0 for no change; default 0.0 if not set
        if self._measurement is True:
            return 1.0
        if self._measurement is False:
            return 0.0
        return 0.0

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

        # Create terminal measurement node and connect as child
        term_id = f"{node_id}_term"
        terminal = TerminalMeasurementNode(term_id)
        self.graph.add_node(terminal)
        self.graph.add_link(node_id, term_id, "sub")
        hypothesis.terminal_id = term_id

        # Connect to executor
        self.graph.add_link("action_executor", node_id, "sub")

        # Store reference
        self.action_hypotheses[action_idx] = hypothesis

        # Map action to terminal id
        self._action_to_terminal[action_idx] = term_id

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

        # Wire por/ret chain across steps to enforce ordering and last-confirm
        for i in range(len(hypothesis.action_hypotheses) - 1):
            prev_id = hypothesis.action_hypotheses[i].id
            next_id = hypothesis.action_hypotheses[i + 1].id
            # Successor inhibition until predecessor is true
            self.graph.add_link(prev_id, next_id, "por")
            # Confirmation inhibition from successors to predecessors
            self.graph.add_link(next_id, prev_id, "ret")

        # Track mapping from action indices to this sequence
        last_index = len(hypothesis.action_hypotheses) - 1
        for pos, act_h in enumerate(hypothesis.action_hypotheses):
            # Find the action_idx for this hypothesis
            for idx, ref in self.action_hypotheses.items():
                if ref.id == act_h.id:
                    self._action_to_sequence[idx] = (hypothesis, pos, last_index)
                    break

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
        Update hypothesis with actual result by setting the terminal measurement.

        Args:
            action_idx: Action that was executed
            frame_changed: Whether the frame actually changed
        """
        if action_idx in self.action_hypotheses:
            # Route via terminal node; apply on next propagate
            self.set_terminal_measurement(action_idx, frame_changed)

    def propagate_step(self):
        """Perform one step of ReCoN message propagation."""
        # First propagate messages within the graph
        self.graph.propagate_step()

        # Apply any pending measurements as bottom-up confirmation/failure
        if self._pending_measurements:
            to_clear = []
            for action_idx, changed in list(self._pending_measurements.items()):
                hypothesis = self.action_hypotheses.get(action_idx)
                if hypothesis is None:
                    to_clear.append(action_idx)
                    continue

                if changed:
                    hypothesis.process_message(ReCoNMessage(MessageType.CONFIRM, "terminal", hypothesis.id, "sur"))
                else:
                    hypothesis.process_message(ReCoNMessage(MessageType.FAIL, "terminal", hypothesis.id, "sur"))
                to_clear.append(action_idx)

            for k in to_clear:
                self._pending_measurements.pop(k, None)

    def set_action_measurement(self, action_idx: int, frame_changed: bool) -> None:
        """
        Queue a measurement result for the given action hypothesis.
        The result is applied on the next propagate_step as a sur confirm/fail.
        """
        self._pending_measurements[action_idx] = bool(frame_changed)

        # Also update any parent sequence bookkeeping immediately
        seq_info = self._action_to_sequence.get(action_idx)
        if seq_info is not None:
            sequence_obj, pos, last = seq_info
            if frame_changed:
                if pos < last:
                    sequence_obj.state = ReCoNState.REQUESTED
                    sequence_obj.current_step = pos + 1
                else:
                    sequence_obj.state = ReCoNState.CONFIRMED
                    sequence_obj.completed_successfully = True
            else:
                sequence_obj.state = ReCoNState.FAILED

    def set_terminal_measurement(self, action_idx: int, changed: bool) -> None:
        """Set measurement on the terminal node so it can emit sur confirm/fail when requested."""
        term_id = self._action_to_terminal.get(action_idx)
        if not term_id:
            return
        node = self.graph.nodes.get(term_id)
        if isinstance(node, TerminalMeasurementNode):
            node.set_measurement(bool(changed))

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