"""
ReCoN Hypothesis Nodes for Active Perception

Implements hypothesis testing nodes that can request confirmation
through action execution and build hierarchical sequences.
"""

import sys
import os
sys.path.append('/workspace/recon-platform')

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from recon_engine.node import ReCoNNode, ReCoNState
from recon_engine.messages import ReCoNMessage, MessageType
from recon_engine.graph import ReCoNGraph
from recon_engine.compact import CompactReCoNNode
import numpy as np
import torch


class CooldownGateNode(ReCoNNode):
    """
    ReCoN-native cooldown gate.

    - Simple ReCoN node that stays ACTIVE while cooldown > 0
    - Emits POR to inhibit action while active
    - Uses gen self-loop to track cooldown countdown
    - Must be requested to maintain inhibition
    """

    def __init__(self, node_id: str, decay_steps: int = 3):
        super().__init__(node_id, "script")
        self.decay_steps = decay_steps
        self.cooldown_remaining = 0

    def trigger_cooldown(self, steps: int):
        """Trigger cooldown for specified number of steps."""
        self.cooldown_remaining = steps
        self.activation = 1.0  # Activate the gate

    def is_cooling_down(self):
        """Check if cooldown is active."""
        return self.cooldown_remaining > 0

    def decay_step(self):
        """Decay cooldown by one step."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.activation = 0.0


class ValidGateNode(ReCoNNode):
    """
    ReCoN-native α_valid delay gate.

    - Simple ReCoN node that stays ACTIVE for (1 - α_valid) * max_delay steps
    - Emits POR to delay action while active
    - Higher α_valid = shorter delay, lower α_valid = longer delay
    - Must be requested to maintain inhibition
    """

    def __init__(self, node_id: str, max_delay: int = 6):
        super().__init__(node_id, "script")
        self.max_delay = max_delay
        self.delay_remaining = 0

    def set_alpha_valid(self, alpha: float):
        """Set delay based on α_valid prior (higher alpha = less delay)."""
        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0,1]
        self.delay_remaining = int((1.0 - alpha) * self.max_delay)
        if self.delay_remaining > 0:
            self.activation = 1.0  # Activate the gate
        else:
            self.activation = 0.0  # No delay needed

    def is_delaying(self):
        """Check if delay is active."""
        return self.delay_remaining > 0

    def decay_step(self):
        """Decay delay by one step."""
        if self.delay_remaining > 0:
            self.delay_remaining -= 1
            if self.delay_remaining == 0:
                self.activation = 0.0


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
        # Map action_idx to gate node ids for inhibition
        self._action_to_gate_por: Dict[int, str] = {}
        self._action_to_gate_ret: Dict[int, str] = {}
        # Pending re-requests for FAILED actions after cooldown clears
        self._pending_rerequest: Set[str] = set()
        # Track nodes whose root request should resume after cooldown clears
        self._cooldown_roots: Set[str] = set()
        # Alternatives scheduling bookkeeping
        self._alt_pending: Dict[str, Set[int]] = {}
        self._action_to_alt_parents: Dict[int, Set[str]] = {}
        # Legacy forced states removed; rely solely on FSM

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

    def _iter_sub_links_from_action(self, action_node_id: str):
        for link in self.graph.links:
            if link.type == "sub" and link.source == action_node_id:
                yield link

    def _iter_sur_links_to_action(self, action_node_id: str):
        for link in self.graph.links:
            if link.type == "sur" and link.target == action_node_id:
                yield link

    def _set_terminal_sur_weight(self, action_idx: int, weight: float) -> None:
        # No-op for explicit FSM gating; weights do not gate discrete messages
        return

    def _remove_sub_link(self, source_id: str, target_id: str) -> None:
        # Remove sub and reciprocal sur from graph
        to_remove = []
        for link in list(self.graph.links):
            if link.source == source_id and link.target == target_id and link.type == "sub":
                to_remove.append(link)
            if link.source == target_id and link.target == source_id and link.type == "sur":
                to_remove.append(link)
        if to_remove:
            for l in to_remove:
                try:
                    self.graph.links.remove(l)
                except ValueError:
                    pass
                try:
                    self.graph.graph.remove_edge(l.source, l.target)
                except Exception:
                    pass

    def _is_action_allowed_now(self, action_idx: int) -> bool:
        """Check if action is allowed (availability only - gates handle cooldown/delays)."""
        return True if self._available_actions is None else (action_idx in self._available_actions)

    def _ensure_gate_for_action(self, action_idx: int):
        if action_idx in self._action_to_gate_por:
            return
        node = self.action_hypotheses.get(action_idx)
        if node is None:
            return

        # Create CooldownGateNode for this action
        cooldown_gate_id = f"cooldown_gate_{node.id}"
        cooldown_gate = CooldownGateNode(cooldown_gate_id)
        try:
            self.graph.add_node(cooldown_gate)
        except Exception:
            pass
        try:
            self.graph.add_link(cooldown_gate_id, node.id, "por")
        except Exception:
            pass
        self._action_to_gate_por[action_idx] = cooldown_gate_id

        # Create ValidGateNode for this action
        valid_gate_id = f"valid_gate_{node.id}"
        valid_gate = ValidGateNode(valid_gate_id)
        try:
            self.graph.add_node(valid_gate)
        except Exception:
            pass
        try:
            self.graph.add_link(valid_gate_id, node.id, "por")
        except Exception:
            pass
        self._action_to_gate_ret[action_idx] = valid_gate_id  # Reusing this field for valid gate

    def _update_gate_request(self, action_idx: int):
        """Deprecated - gate nodes manage themselves."""
        # Gate nodes are now self-managing through direct API calls
        # This method is kept for compatibility but does nothing
        pass

    def _apply_valid_weight_gate(self, action_idx: int):
        # Deprecated: do not use weight-based gating in explicit FSM
        self._ensure_gate_for_action(action_idx)
        gate_por_id = self._action_to_gate_por.get(action_idx)
        if self.valid_delays.get(action_idx, 0) > 0:
            if gate_por_id:
                self.graph.request_root(gate_por_id)
        else:
            if gate_por_id:
                self.graph.stop_request(gate_por_id)

    def set_available_actions(self, allowed_action_indices: Optional[List[int]]):
        """Set currently allowed actions; None or empty means all allowed."""
        if allowed_action_indices is None:
            self._available_actions = None
        else:
            self._available_actions = set(allowed_action_indices)
        # Gate nodes are self-managing

    def set_alpha_valid(self, mapping: Dict[int, float]) -> None:
        """Set α_valid prior per action in [0,1]; lower values delay requests longer."""
        for idx, val in mapping.items():
            v = float(max(0.0, min(1.0, val)))
            self.alpha_valid[idx] = v

            # Update ValidGateNode directly
            self._ensure_gate_for_action(idx)
            valid_gate_id = self._action_to_gate_ret.get(idx)
            if valid_gate_id:
                valid_gate = self.graph.get_node(valid_gate_id)
                if isinstance(valid_gate, ValidGateNode):
                    valid_gate.set_alpha_valid(v)
                    if valid_gate.is_delaying():
                        self.graph.request_root(valid_gate_id)  # Keep gate active
                    else:
                        self.graph.stop_request(valid_gate_id)  # Release gate

            # Gate nodes are directly updated above

    def set_alpha_value(self, mapping: Dict[int, float]) -> None:
        """Set α_value prior per action (higher means earlier ordering in alternatives)."""
        for idx, val in mapping.items():
            self.alpha_value[idx] = float(val)

    def feed_cnn_priors(self, valid_probs: Dict[int, float], value_probs: Dict[int, float]) -> None:
        """
        Feed CNN priors to gate nodes each step.

        Args:
            valid_probs: Dict mapping action_idx -> sigmoid(valid_head[action])
            value_probs: Dict mapping action_idx -> value_head[action]
        """
        # Update α_valid via ValidGateNodes
        self.set_alpha_valid(valid_probs)

        # Update α_value for alternatives ordering
        self.set_alpha_value(value_probs)

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

        # Ensure gate nodes are created for this action
        self._ensure_gate_for_action(action_idx)

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
            # Ensure gate nodes for each child
            self._ensure_gate_for_action(act_h.action_idx)

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
            self._ensure_gate_for_action(ah.action_idx)

        # Apply value-based ordering: higher α_value should go first → por from higher to lower
        # Also schedule by α_valid: initially POR-gate all but the highest α_valid so only top child can progress
        scored = sorted(children, key=lambda n: (
            self.alpha_value.get(self._find_action_idx_by_id(n.id), 0.0),
            self.alpha_valid.get(self._find_action_idx_by_id(n.id), 0.0)
        ), reverse=True)
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
                # If α_value ties, prefer higher α_valid by adding a brief valid delay to the lower one
                elif abs(val_hi - val_lo) < 1e-6:
                    valid_hi = self.alpha_valid.get(idx_hi, 0.0)
                    valid_lo = self.alpha_valid.get(idx_lo, 0.0)
                    if valid_hi > valid_lo:
                        # Set lower α_valid for the lower-priority option
                        self.set_alpha_valid({idx_lo: valid_lo})
        # α_valid initial scheduling: only link the highest α_valid child initially.
        by_valid = sorted(children, key=lambda n: self.alpha_valid.get(self._find_action_idx_by_id(n.id), 0.0), reverse=True)
        self._alt_pending[node_id] = set()
        if by_valid:
            top = by_valid[0]
            top_idx = self._find_action_idx_by_id(top.id)
            if top_idx is not None:
                try:
                    self.graph.add_link(node_id, top.id, "sub")
                except Exception:
                    pass
                # Also request the top child directly to ensure immediate progression
                try:
                    self.graph.request_root(top.id)
                except Exception:
                    pass
                # Ensure mapping
                self._action_to_alt_parents.setdefault(top_idx, set()).add(node_id)
            # All others are pending; add to mapping
            for node in by_valid[1:]:
                idx = self._find_action_idx_by_id(node.id)
                if idx is None:
                    continue
                self._alt_pending[node_id].add(idx)
                self._action_to_alt_parents.setdefault(idx, set()).add(node_id)
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
        # If failed during an active request, start cooldown; if succeeded, only clear cooldown if none is active
        parent = self.action_hypotheses.get(action_idx)
        parent_requested = bool(parent and parent.id in self.graph.requested_roots)
        parent_active = bool(parent and parent.state in (
            ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED, ReCoNState.FAILED
        ))
        if not changed:
            if parent_requested and parent_active:
                # Trigger cooldown gate directly
                self._ensure_gate_for_action(action_idx)
                cooldown_gate_id = self._action_to_gate_por.get(action_idx)
                if cooldown_gate_id:
                    cooldown_gate = self.graph.get_node(cooldown_gate_id)
                    if isinstance(cooldown_gate, CooldownGateNode):
                        cooldown_gate.trigger_cooldown(self.cooldown_steps)
                        self.graph.request_root(cooldown_gate_id)  # Keep gate active

                # Gate nodes handle their own cooldown tracking

                # Stop direct root request to enforce suppression window
                if parent and parent.id in self.graph.requested_roots:
                    self.graph.stop_request(parent.id)
                    self._cooldown_roots.add(parent.id)
                # For each alternatives parent containing this action, link next pending child now
                for alt_id in list(self._action_to_alt_parents.get(action_idx, set())):
                    pending = self._alt_pending.get(alt_id)
                    if not pending:
                        continue
                    # Remove sub link from alt to the failed child so it stops being requested
                    try:
                        self._remove_sub_link(alt_id, parent.id)
                    except Exception:
                        pass
                    # Choose highest α_valid among pending
                    next_idx = None
                    best = -1.0
                    for p in pending:
                        v = self.alpha_valid.get(p, 0.0)
                        if v > best:
                            best = v
                            next_idx = p
                    if next_idx is not None:
                        child = self.action_hypotheses.get(next_idx)
                        if child:
                            try:
                                self.graph.add_link(alt_id, child.id, "sub")
                            except Exception:
                                pass
                            try:
                                self.graph.request_root(child.id)
                            except Exception:
                                pass
                            # Ensure the alternatives parent is actively requested
                            try:
                                self.graph.request_root(alt_id)
                            except Exception:
                                pass
                            pending.remove(next_idx)
                            # Nudge propagation to advance new child quickly
                            try:
                                self.graph.propagate_step()
                                self.graph.propagate_step()
                            except Exception:
                                pass
        else:
            # If under cooldown, keep requests suppressed; do not overwrite measurement
            if self.cooldowns.get(action_idx, 0) > 0:
                if parent and parent.id in self.graph.requested_roots:
                    self._cooldown_roots.add(parent.id)
                    self.graph.stop_request(parent.id)
            else:
                self.cooldowns[action_idx] = 0
        # Gate nodes update themselves
        # Nudge propagation so parent can see sur confirm/fail sooner when appropriate
        try:
            # Only nudge if not under cooldown inhibition
            if self.cooldowns.get(action_idx, 0) <= 0:
                self.graph.propagate_step()
        except Exception:
            pass

    def propagate_step(self):
        """Propagate one step in the ReCoN graph (terminals will emit sur when requested)."""
        # Decay gate nodes directly
        for idx in list(self.action_hypotheses.keys()):
            self._ensure_gate_for_action(idx)

            # Decay cooldown gate
            cooldown_gate_id = self._action_to_gate_por.get(idx)
            if cooldown_gate_id:
                cooldown_gate = self.graph.get_node(cooldown_gate_id)
                if isinstance(cooldown_gate, CooldownGateNode) and cooldown_gate.is_cooling_down():
                    cooldown_gate.decay_step()
                    if not cooldown_gate.is_cooling_down():
                        # Cooldown finished, stop requesting gate
                        self.graph.stop_request(cooldown_gate_id)
                        # Allow action to be re-requested
                        node = self.action_hypotheses.get(idx)
                        if node and node.id in self._cooldown_roots:
                            self.graph.request_root(node.id)
                            self._cooldown_roots.remove(node.id)

            # Decay valid gate
            valid_gate_id = self._action_to_gate_ret.get(idx)
            if valid_gate_id:
                valid_gate = self.graph.get_node(valid_gate_id)
                if isinstance(valid_gate, ValidGateNode) and valid_gate.is_delaying():
                    valid_gate.decay_step()
                    if not valid_gate.is_delaying():
                        # Delay finished, stop requesting gate
                        self.graph.stop_request(valid_gate_id)

        # Legacy code removed - gate nodes handle their own decay

        self.graph.propagate_step()
        # No legacy forced states

        # Apply any pending re-requests (so next step they are requested again)
        if self._pending_rerequest:
            for node_id in list(self._pending_rerequest):
                self.graph.request_root(node_id)
                self._pending_rerequest.remove(node_id)
        # Resume root requests for nodes whose cooldown just cleared
        for idx, remaining in list(self.cooldowns.items()):
            if remaining <= 0:
                node = self.action_hypotheses.get(idx)
                if node and node.id in self._cooldown_roots:
                    self.graph.request_root(node.id)
                    self._cooldown_roots.remove(node.id)
        # If an alternatives child reached a terminal outcome (failed or confirmed), link next pending child
        for alt_id, pending in list(self._alt_pending.items()):
            # If parent alt_id is not in graph, skip
            if alt_id not in self.graph.nodes:
                continue
            # Check current linked children
            linked = [l for l in self.graph.get_links(source=alt_id, link_type="sub")]
            # If no child is linked and there is a pending action, attach the next one
            if not linked and pending:
                next_idx = None
                # Choose highest α_valid among pending
                best = -1.0
                for p in pending:
                    v = self.alpha_valid.get(p, 0.0)
                    if v > best:
                        best = v
                        next_idx = p
                if next_idx is not None:
                    child = self.action_hypotheses.get(next_idx)
                    if child:
                        try:
                            self.graph.add_link(alt_id, child.id, "sub")
                        except Exception:
                            pass
                        # Also request the child directly to allow progression now
                        try:
                            self.graph.request_root(child.id)
                        except Exception:
                            pass
                        pending.remove(next_idx)
        # Enforce suppression during cooldown on node states to avoid premature TRUE
        for idx, remaining in list(self.cooldowns.items()):
            if remaining > 0:
                node = self.action_hypotheses.get(idx)
                if node and node.state in (
                    ReCoNState.REQUESTED,
                    ReCoNState.ACTIVE,
                    ReCoNState.WAITING,
                    ReCoNState.TRUE,
                    ReCoNState.CONFIRMED,
                ):
                    node.state = ReCoNState.SUPPRESSED

        # (Explicit gen-based persistence not implemented here to avoid interfering with FSM timing.)

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
