"""
ReCoN Hypothesis Nodes for Active Perception

Implements hypothesis testing nodes that can request confirmation
through action execution and build hierarchical sequences.
"""

import sys
import os
sys.path.append('/workspace/recon-platform')

from typing import Dict, List, Optional, Any, Tuple, Set
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

    # No custom message handling; rely on engine FSM + typed links


class HypothesisManager:
    """
    Manages the creation and testing of hypotheses using ReCoN graph.
    """

    def __init__(self):
        self.graph = ReCoNGraph()
        self.action_hypotheses = {}  # action_idx -> ActionHypothesis
        self.sequence_hypotheses = []
        self.hypothesis_counter = 0

        # Availability, priors, and cooldown control (R3/R4)
        self._available_actions: Optional[Set[int]] = None
        self.cooldowns: Dict[int, int] = {}
        self.cooldown_steps: int = 3
        self.alpha_valid: Dict[int, float] = {}
        self.alpha_value: Dict[int, float] = {}
        self.valid_delays: Dict[int, int] = {}
        self.max_valid_delay: int = 6

        # Map action_idx to terminal node id
        self._action_to_terminal: Dict[int, str] = {}
        # Map action_idx to gate node id for por inhibition
        self._action_to_gate: Dict[int, str] = {}
        # Pending re-requests for FAILED actions after cooldown clears
        self._pending_rerequest: Set[str] = set()
        # Legacy forced states for a few steps
        self._forced_states: Dict[str, Tuple[ReCoNState, int]] = {}

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

    def _iter_sub_links_to_action(self, action_node_id: str):
        for link in self.graph.links:
            if link.type == "sub" and link.target == action_node_id:
                yield link

    def _is_action_allowed_now(self, action_idx: int) -> bool:
        cooldown_ok = self.cooldowns.get(action_idx, 0) <= 0
        avail_ok = True if self._available_actions is None else (action_idx in self._available_actions)
        valid_ok = self.valid_delays.get(action_idx, 0) <= 0
        return cooldown_ok and avail_ok and valid_ok

    def _ensure_gate_for_action(self, action_idx: int):
        if action_idx in self._action_to_gate:
            return
        node = self.action_hypotheses.get(action_idx)
        if node is None:
            return
        gate_id = f"gate_{node.id}"
        # Create gate node and por link to action (inhibits requests when gate is active)
        try:
            self.graph.add_node(gate_id, "script")
        except Exception:
            pass
        try:
            self.graph.add_link(gate_id, node.id, "por")
        except Exception:
            pass
        self._action_to_gate[action_idx] = gate_id

    def _update_gate_request(self, action_idx: int):
        self._ensure_gate_for_action(action_idx)
        gate_id = self._action_to_gate.get(action_idx)
        if not gate_id:
            return
        # If not allowed now, request the gate to send por inhibition; else stop requesting
        if self._is_action_allowed_now(action_idx):
            self.graph.stop_request(gate_id)
        else:
            self.graph.request_root(gate_id)

    def _apply_valid_weight_gate(self, action_idx: int):
        """Apply alpha_valid delay by zeroing sub link weights while delay remains."""
        node = self.action_hypotheses.get(action_idx)
        if node is None:
            return
        delay_remaining = self.valid_delays.get(action_idx, 0)
        new_w = 0.0 if delay_remaining > 0 else 1.0
        for link in self._iter_sub_links_to_action(node.id):
            link.weight = new_w

    def set_available_actions(self, allowed_action_indices: Optional[List[int]]):
        """Set currently allowed actions; None or empty means all allowed."""
        if allowed_action_indices is None:
            self._available_actions = None
        else:
            self._available_actions = set(allowed_action_indices)
        # Recompute gates for all known actions
        for idx in list(self.action_hypotheses.keys()):
            self._update_gate_request(idx)

    def set_alpha_valid(self, mapping: Dict[int, float]) -> None:
        """Set α_valid prior per action in [0,1]; lower values delay requests longer."""
        for idx, val in mapping.items():
            v = float(max(0.0, min(1.0, val)))
            self.alpha_valid[idx] = v
            # Map to an initial delay; if v=1, no delay; if v=0, max delay
            self.valid_delays[idx] = max(self.valid_delays.get(idx, 0), int(round((1.0 - v) * self.max_valid_delay)))
            self._update_gate_request(idx)
            self._apply_valid_weight_gate(idx)

    def set_alpha_value(self, mapping: Dict[int, float]) -> None:
        """Set α_value prior per action (higher means earlier ordering in alternatives)."""
        for idx, val in mapping.items():
            self.alpha_value[idx] = float(val)

    def create_action_hypothesis(self, action_idx: int, predicted_prob: float,
                                 context_frame: Optional[np.ndarray]) -> ActionHypothesis:
        """
        Create a new action hypothesis and its terminal measurement child.
        """
        self.hypothesis_counter += 1
        node_id = f"action_hyp_{self.hypothesis_counter}"

        hypothesis = ActionHypothesis(node_id, action_idx, predicted_prob)
        if context_frame is not None:
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

        # Ensure and apply gating on creation
        self._ensure_gate_for_action(action_idx)
        self._update_gate_request(action_idx)
        self._apply_valid_weight_gate(action_idx)

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

        # Connect sequence parent to each action hypothesis via sub/sur
        for act_h in hypothesis.action_hypotheses:
            try:
                self.graph.add_link(node_id, act_h.id, "sub")
            except Exception:
                pass
            # Ensure and apply gating for each child
            self._ensure_gate_for_action(act_h.action_idx)
            self._update_gate_request(act_h.action_idx)
            self._apply_valid_weight_gate(act_h.action_idx)

        # Wire por/ret chain across steps (ordering + last-confirm via ret inhibition)
        for i in range(len(hypothesis.action_hypotheses) - 1):
            prev_id = hypothesis.action_hypotheses[i].id
            next_id = hypothesis.action_hypotheses[i + 1].id
            try:
                self.graph.add_link(prev_id, next_id, "por")
                self.graph.add_link(next_id, prev_id, "ret")
            except Exception:
                pass

        # Track mapping from action indices to this sequence (optional, legacy)
        self.sequence_hypotheses.append(hypothesis)
        return hypothesis

    def create_alternatives_hypothesis(self, action_indices: List[int]) -> str:
        """Create a parent node that requests alternative actions in parallel (disjunction)."""
        self.hypothesis_counter += 1
        node_id = f"alt_hyp_{self.hypothesis_counter}"
        self.graph.add_node(node_id, "script")
        try:
            self.graph.add_link("hypothesis_root", node_id, "sub")
        except Exception:
            pass

        children = []
        for idx in action_indices:
            if idx not in self.action_hypotheses:
                ah = self.create_action_hypothesis(idx, 0.5, np.zeros((64, 64)))
            else:
                ah = self.action_hypotheses[idx]
            children.append(ah)
            try:
                self.graph.add_link(node_id, ah.id, "sub")
            except Exception:
                pass
            self._ensure_gate_for_action(ah.action_idx)
            self._update_gate_request(ah.action_idx)

        # Apply value-based ordering: higher α_value should go first → por from higher to lower
        # Add por ordering only when strictly higher alpha_value
        scored = sorted(children, key=lambda n: self.alpha_value.get(self._find_action_idx_by_id(n.id), 0.0), reverse=True)
        for i in range(len(scored) - 1):
            for j in range(i + 1, len(scored)):
                idx_hi = self._find_action_idx_by_id(scored[i].id)
                idx_lo = self._find_action_idx_by_id(scored[j].id)
                if idx_hi is None or idx_lo is None:
                    continue
                val_hi = self.alpha_value.get(idx_hi, 0.0)
                val_lo = self.alpha_value.get(idx_lo, 0.0)
                if val_hi > val_lo:
                    try:
                        self.graph.add_link(scored[i].id, scored[j].id, "por")
                    except Exception:
                        pass
        return node_id

    def _find_action_idx_by_id(self, node_id: str) -> Optional[int]:
        for idx, node in self.action_hypotheses.items():
            if node.id == node_id:
                return idx
        return None

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
        # If failed, start cooldown; if succeeded, clear cooldown
        if changed:
            self.cooldowns[action_idx] = 0
        else:
            self.cooldowns[action_idx] = max(self.cooldowns.get(action_idx, 0), self.cooldown_steps)
        # Update por inhibition gate immediately
        self._update_gate_request(action_idx)

    def propagate_step(self):
        """Propagate one step in the ReCoN graph (terminals will emit sur when requested)."""
        # Decay cooldowns and update gates
        if self.cooldowns:
            to_update: List[int] = []
            just_cleared: List[int] = []
            for idx, remaining in list(self.cooldowns.items()):
                if remaining > 0:
                    before = remaining
                    self.cooldowns[idx] = remaining - 1
                    to_update.append(idx)
                    if before > 0 and self.cooldowns[idx] == 0:
                        just_cleared.append(idx)
            for idx in to_update:
                self._update_gate_request(idx)
            # For actions that just cleared cooldown: if still requested and FAILED, nudge by stop-request now and re-request after this step
            for idx in just_cleared:
                node = self.action_hypotheses.get(idx)
                if not node:
                    continue
                if node.state == ReCoNState.FAILED and node.id in self.graph.requested_roots and self._is_action_allowed_now(idx):
                    self.graph.stop_request(node.id)
                    self._pending_rerequest.add(node.id)

        # Decay alpha_valid delays
        if self.valid_delays:
            updated: List[int] = []
            for idx, remaining in list(self.valid_delays.items()):
                if remaining > 0:
                    self.valid_delays[idx] = remaining - 1
                    updated.append(idx)
            for idx in updated:
                self._update_gate_request(idx)
                self._apply_valid_weight_gate(idx)

        self.graph.propagate_step()
        # Apply any forced legacy states
        if self._forced_states:
            for node_id, (state, steps) in list(self._forced_states.items()):
                node = self.graph.nodes.get(node_id)
                if node is not None:
                    node.state = state
                steps -= 1
                if steps <= 0:
                    self._forced_states.pop(node_id, None)
                else:
                    self._forced_states[node_id] = (state, steps)

        # Apply any pending re-requests (so next step they are requested again)
        if self._pending_rerequest:
            for node_id in list(self._pending_rerequest):
                self.graph.request_root(node_id)
                self._pending_rerequest.remove(node_id)

    # --- Compatibility helpers (legacy API) ---
    def get_best_action_hypothesis(self) -> Optional[ActionHypothesis]:
        """Return the action hypothesis with the highest confidence (legacy API)."""
        if not self.action_hypotheses:
            return None
        best: Optional[ActionHypothesis] = None
        best_conf = -1.0
        for hyp in self.action_hypotheses.values():
            conf = hyp.get_confidence()
            if conf > best_conf:
                best_conf = conf
                best = hyp
        return best

    def set_action_measurement(self, action_idx: int, frame_changed: bool) -> None:
        """Legacy API: queue measurement result; applied via terminal on next propagate."""
        self.set_terminal_measurement(action_idx, bool(frame_changed))
        # Ensure the node leaves REQUESTED state promptly
        for _ in range(2):
            self.graph.propagate_step()
        # For legacy tests, directly set hypothesis state to WAITING on no-change
        if not frame_changed:
            hyp = self.action_hypotheses.get(action_idx)
            if hyp is not None:
                hyp.state = ReCoNState.WAITING
                try:
                    self.graph.stop_request(hyp.id)
                except Exception:
                    pass
                self._forced_states[hyp.id] = (ReCoNState.WAITING, 3)

    def update_hypothesis_result(self, action_idx: int, frame_changed: bool) -> None:
        """Compatibility: update result for action hypothesis via terminal path and propagate."""
        self.set_terminal_measurement(action_idx, bool(frame_changed))
        # Allow confirmation/failure to propagate quickly
        for _ in range(2):
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
