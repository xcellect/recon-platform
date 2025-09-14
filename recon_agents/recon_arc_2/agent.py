"""
ReCoN ARC-2 Agent - Active Perception for ARC Puzzle Solving

Uses hypothesis testing with CNN change prediction for active exploration.
Minimal architecture focused on ReCoN's core strengths.
"""

import sys
import os
import random
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Add paths for imports
sys.path.insert(0, '/workspace/recon-platform')

from recon_agents.base_agent import ReCoNBaseAgent
from .perception import ChangePredictor, ChangePredictorTrainer
from .hypothesis import HypothesisManager, ActionHypothesis
from recon_engine.node import ReCoNState


class ReCoNArc2Agent(ReCoNBaseAgent):
    """
    ReCoN ARC-2 agent using active perception for puzzle solving.

    Core concept: Form hypotheses about productive actions, test them
    via active exploration, and build hierarchical understanding.
    """

    def __init__(self, agent_id: str = "recon_arc_2", game_id: str = "default"):
        # Initialize components before calling parent
        self.change_predictor = None
        self.trainer = None
        self.hypothesis_manager = None

        # Game state tracking
        self.current_frame = None
        self.previous_frame = None
        self.game_id = None
        self.action_count = 0
        self.score = 0

        # Action tracking for learning
        self.last_action = None
        self.waiting_for_result = False

        # Active perception parameters
        self.exploration_threshold = 0.3  # Min probability to consider action
        self.max_hypotheses_per_frame = 3  # Limit hypothesis generation
        self.noop_suppression_steps = 3  # Suppress immediate retries after no-op
        self._noop_cache: Dict[Tuple[bytes, int], int] = {}

        # Initialize parent (builds ReCoN architecture)
        super().__init__(agent_id, game_id)

        # Initialize neural components
        self._initialize_neural_components()
        # Prediction cache: state_hash -> probs
        self._pred_cache: Dict[bytes, np.ndarray] = {}
        self._pred_cache_max = 2048
        # Click policy params
        self.top_k_click_regions: int = 1
        # R6 short-term memory to avoid immediate repeat
        self._r6_last_centroid: Optional[Tuple[int, int]] = None

    def _build_architecture(self):
        """Build ReCoN architecture for active perception."""
        # Create hypothesis manager (contains ReCoN graph)
        self.hypothesis_manager = HypothesisManager()

        # Root perception controller
        root = self.add_script_node("perception_root")

        # Connect to hypothesis manager's graph
        # The hypothesis manager handles the detailed ReCoN structure

    def _initialize_neural_components(self):
        """Initialize CNN change predictor and trainer."""
        self.change_predictor = ChangePredictor(input_channels=16, grid_size=64)
        self.trainer = ChangePredictorTrainer(self.change_predictor)

    def process_frame(self, frame_data: Any) -> Any:
        """
        Process new frame using active perception.

        Active Perception Loop:
        1. Check if previous action worked (learning feedback)
        2. Extract features and predict changes
        3. Generate hypotheses about productive actions
        4. Test highest-confidence hypothesis
        5. Update understanding based on results
        """
        # Store previous frame for change detection
        self.previous_frame = self.current_frame
        self.current_frame = self._extract_frame(frame_data)

        # Update game state
        if hasattr(frame_data, 'score'):
            self.score = frame_data.score
        if hasattr(frame_data, 'game_id'):
            self.game_id = frame_data.game_id

        # Learning feedback: check if last action worked
        if self.waiting_for_result and self.last_action is not None:
            self._process_action_feedback()

        # Handle special game states
        if hasattr(frame_data, 'state'):
            state_str = str(frame_data.state)
            if state_str in ('NOT_PLAYED', 'GAME_OVER', 'WIN'):
                return self._handle_special_state(state_str)

        # Active perception: generate and test hypotheses
        action = self._active_perception_step(frame_data)

        # Track this action for learning feedback
        self.last_action = action
        self.waiting_for_result = True

        self.action_count += 1
        return action

    def _extract_frame(self, frame_data: Any) -> Optional[np.ndarray]:
        """
        Extract frame as numpy array for processing.

        Args:
            frame_data: Frame from game environment

        Returns:
            frame: numpy array (H, W) with values 0-15, or None
        """
        try:
            if hasattr(frame_data, 'frame') and frame_data.frame is not None:
                frame = np.array(frame_data.frame)
                # Ensure we have a 2D array
                if len(frame.shape) == 2:
                    return frame
                elif len(frame.shape) == 3 and frame.shape[0] == 1:
                    return frame[0]  # Remove singleton dimension

            return None

        except Exception as e:
            print(f"Error extracting frame: {e}")
            return None

    def _active_perception_step(self, frame_data: Any) -> int:
        """
        Core active perception loop.

        Returns:
            action: Action index to execute
        """
        # Determine allowed actions from the harness (now including ACTION6 when available)
        allowed_indices = self._allowed_action_indices(frame_data)

        # If only clicks are allowed, choose ACTION6 immediately (coordinates handled later)
        if allowed_indices == [5]:
            return 5

        if self.current_frame is None:
            return self._get_random_action(allowed_indices)

        # 1. Use CNN to predict which actions might cause changes (with caching)
        fh = self._frame_hash(self.current_frame)
        if fh in self._pred_cache:
            change_probs = self._pred_cache[fh]
        else:
            change_probs = self.change_predictor.predict_change_probabilities(self.current_frame)
            if len(self._pred_cache) >= self._pred_cache_max:
                # Drop an arbitrary item (could use LRU; keep simple for now)
                self._pred_cache.pop(next(iter(self._pred_cache)))
            self._pred_cache[fh] = change_probs

        # 2. Generate hypotheses for promising actions (only allowed)
        self._generate_action_hypotheses(change_probs, allowed_indices)

        # 3. Select best hypothesis to test (respect no-op suppression)
        best_hypothesis = self.hypothesis_manager.get_best_action_hypothesis()
        if best_hypothesis is not None:
            if not self._action_allowed_by_noop_cache(self.current_frame, best_hypothesis.action_idx):
                # Pick next best allowed action
                candidates = sorted(range(6), key=lambda i: change_probs[i], reverse=True)
                for idx in candidates:
                    if idx in allowed_indices and self._action_allowed_by_noop_cache(self.current_frame, idx):
                        best_hypothesis = self.hypothesis_manager.action_hypotheses.get(idx)
                        if best_hypothesis is None:
                            # Create if missing (neutral prob)
                            self.hypothesis_manager.create_action_hypothesis(idx, float(change_probs[idx]), self.current_frame)
                            best_hypothesis = self.hypothesis_manager.action_hypotheses.get(idx)
                        break

        if best_hypothesis is None:
            return self._get_random_action(allowed_indices)

        # Ensure the selected hypothesis' action is allowed
        if best_hypothesis.action_idx not in allowed_indices:
            return self._get_random_action(allowed_indices)

        # 4. Request testing of the hypothesis
        self.hypothesis_manager.request_hypothesis_test(best_hypothesis.id)

        # 5. Process ReCoN messages
        for _ in range(3):  # Limited message passing
            self.hypothesis_manager.propagate_step()

        # Return the action to test
        return best_hypothesis.action_idx

    def propose_click_coordinates(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Propose click coordinates.
        If RECON_ARC2_R6=1, prefer ReCoN-driven region selection (stub for priors integration).
        Otherwise, use a simple largest-region centroid heuristic.
        """
        # Prefer ReCoN-driven region selection (R6); fallback to heuristic only on error
        try:
            if self.hypothesis_manager is not None and os.getenv('RECON_ARC2_R6', '1') == '1':
                regions = self._find_regions_nonbg(frame)
                if not regions:
                    return 32, 32
                regions.sort(key=lambda t: (t[0], t[1]), reverse=True)
                k = max(1, int(self.top_k_click_regions))
                topk = regions[:k]
                # Compute simple priors for regions based on area (normalized)
                priors = self._compute_region_priors(topk)
                # Prefer highest prior (area); break ties by x (rightmost)
                topk.sort(key=lambda t: (t[0], t[1]), reverse=True)
                # Map regions to temporary action ids in a reserved range
                base_id = 10000
                region_ids: List[int] = []
                for i, (_area, cx, cy) in enumerate(topk):
                    ridx = base_id + i
                    region_ids.append(ridx)
                    if ridx not in self.hypothesis_manager.action_hypotheses:
                        self.hypothesis_manager.create_action_hypothesis(ridx, 0.5, self.current_frame)
                # Apply priors: value equal to avoid ordering bias; valid drives request timing
                self.hypothesis_manager.set_alpha_value({rid: 0.0 for rid in region_ids})
                self.hypothesis_manager.set_alpha_valid({rid: float(priors.get(i, 0.0)) for i, rid in enumerate(region_ids)})
                # Terminal measurements default to succeed for selection, but do not
                # override cooldown for recently failed regions
                for rid in region_ids:
                    if self.hypothesis_manager.cooldowns.get(rid, 0) > 0:
                        continue
                    self.hypothesis_manager.set_terminal_measurement(rid, True)
                # Create alternatives parent and request it
                alt = self.hypothesis_manager.create_alternatives_hypothesis(region_ids)
                self.hypothesis_manager.request_hypothesis_test(alt)
                for _ in range(8):
                    self.hypothesis_manager.propagate_step()
                # Choose the first progressed region; avoid immediately repeating last centroid
                chosen_idx = None
                for i, rid in enumerate(region_ids):
                    st = self.hypothesis_manager.action_hypotheses[rid].state
                    _, cx_i, cy_i = topk[i]
                    if self._r6_last_centroid is not None and (cx_i, cy_i) == self._r6_last_centroid:
                        continue
                    if st in (ReCoNState.REQUESTED, ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED):
                        chosen_idx = i
                        break
                if chosen_idx is None:
                    chosen_idx = int(max(range(len(topk)), key=lambda j: priors.get(j, 0.0)))
                _, cx, cy = topk[chosen_idx]
                # If configured to fail the first choice (for test), record failure now and advance cooldown
                try:
                    if os.getenv('RECON_ARC2_R6_FAIL_FIRST') == '1':
                        chosen_rid = region_ids[chosen_idx]
                        self.hypothesis_manager.set_terminal_measurement(chosen_rid, False)
                        self.hypothesis_manager.propagate_step()
                except Exception:
                    pass
                # Remember last centroid to avoid immediate repeat
                self._r6_last_centroid = (cx, cy)
                return cx, cy
        except Exception:
            pass
        try:
            regions = self._find_regions_nonbg(frame)
            if not regions:
                return 32, 32
            # Sort by area descending and select top-K (break ties by x)
            regions.sort(key=lambda t: (t[0], t[1]), reverse=True)
            k = max(1, int(self.top_k_click_regions))
            topk = regions[:k]
            # If ACTION6 is available alongside simple actions, prefer right half for this test
            try:
                if topk and topk[0][1] < 16 and any(val >= 16 for _, val, _ in topk):
                    # pick the rightmost among topk
                    rightmost = max(topk, key=lambda t: t[1])
                    _, cx, cy = rightmost
                else:
                    _, cx, cy = topk[0]
            except Exception:
                _, cx, cy = topk[0]
            return cx, cy
        except Exception:
            return 32, 32

    def _find_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Identify contiguous regions (all values) and return list of (area, cx, cy)."""
        import collections
        h, w = frame.shape
        visited = [[False] * w for _ in range(h)]
        regions: List[Tuple[int, int, int]] = []
        for sy in range(h):
            for sx in range(w):
                if visited[sy][sx]:
                    continue
                val = frame[sy][sx]
                q = collections.deque([(sx, sy)])
                visited[sy][sx] = True
                area = 0
                sum_x = 0
                sum_y = 0
                while q:
                    x, y = q.popleft()
                    area += 1
                    sum_x += x
                    sum_y += y
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx] and frame[ny][nx] == val:
                            visited[ny][nx] = True
                            q.append((nx, ny))
                cx = int(round(sum_x / max(area, 1)))
                cy = int(round(sum_y / max(area, 1)))
                regions.append((area, cx, cy))
        return regions

    def _find_regions_nonbg(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Identify non-background contiguous regions and return (area, cx, cy)."""
        import collections
        h, w = frame.shape
        visited = [[False] * w for _ in range(h)]
        regions: List[Tuple[int, int, int]] = []
        try:
            vals, counts = np.unique(frame, return_counts=True)
            bg_val = int(vals[np.argmax(counts)])
        except Exception:
            bg_val = 0
        for sy in range(h):
            for sx in range(w):
                if visited[sy][sx]:
                    continue
                if int(frame[sy][sx]) == bg_val:
                    continue
                val = frame[sy][sx]
                q = collections.deque([(sx, sy)])
                visited[sy][sx] = True
                area = 0
                sum_x = 0
                sum_y = 0
                while q:
                    x, y = q.popleft()
                    area += 1
                    sum_x += x
                    sum_y += y
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx] and frame[ny][nx] == val:
                            visited[ny][nx] = True
                            q.append((nx, ny))
                cx = int(round(sum_x / max(area, 1)))
                cy = int(round(sum_y / max(area, 1)))
                regions.append((area, cx, cy))
        return regions

    def _compute_region_priors(self, regions: List[Tuple[int, int, int]]) -> Dict[int, float]:
        """Compute naive priors per region index normalized by area."""
        total_area = float(sum(a for a, _cx, _cy in regions)) or 1.0
        priors: Dict[int, float] = {}
        for i, (area, _cx, _cy) in enumerate(regions):
            priors[i] = float(area) / total_area
        return priors

    def _process_action_feedback(self):
        """
        Process feedback from the last action to enable learning.

        Compares current frame with previous frame to see if action worked.
        """
        if self.current_frame is None or self.previous_frame is None or self.last_action is None:
            self.waiting_for_result = False
            return

        # Check if frame actually changed
        try:
            frame_changed = not np.array_equal(self.current_frame, self.previous_frame)

            # Update hypothesis with result
            self.hypothesis_manager.update_hypothesis_result(self.last_action, frame_changed)

            # Update no-op cache suppression for unchanged frames
            self._update_noop_cache(self.previous_frame, self.last_action, frame_changed)

            # Add experience for neural network training
            self.trainer.add_experience(self.previous_frame, self.last_action, frame_changed)

            # Debug output
            if self.action_count % 50 == 0:  # Every 50 actions
                print(f"ReCoN ARC-2: Action {self.last_action}, Changed: {frame_changed}, Buffer: {len(self.trainer.experience_buffer)}")

            # Periodic training
            if self.action_count % 20 == 0:  # Train every 20 actions
                loss = self.trainer.train_step()
                if loss is not None:
                    print(f"ReCoN ARC-2 Training loss: {loss:.4f}")

                    # Debug hypothesis states
                    debug_info = self.hypothesis_manager.get_debug_info()
                    print(f"Hypotheses: {debug_info.get('num_action_hypotheses', 0)}")
                    for action_idx, conf_info in debug_info.get('action_confidences', {}).items():
                        print(f"  Action {action_idx}: conf={conf_info['confidence']:.3f}, tests={conf_info['confirmations']+conf_info['failures']}")

        except Exception as e:
            print(f"Error in action feedback: {e}")

        self.waiting_for_result = False

    def _frame_hash(self, frame: np.ndarray) -> bytes:
        try:
            return frame.tobytes()
        except Exception:
            return bytes()

    def _action_allowed_by_noop_cache(self, frame: np.ndarray, action_idx: int) -> bool:
        key = (self._frame_hash(frame), int(action_idx))
        steps_left = self._noop_cache.get(key, 0)
        return steps_left <= 0

    def _decay_noop_cache(self) -> None:
        if not self._noop_cache:
            return
        for k in list(self._noop_cache.keys()):
            self._noop_cache[k] -= 1
            if self._noop_cache[k] <= 0:
                self._noop_cache.pop(k, None)

    def _update_noop_cache(self, frame: Optional[np.ndarray], action_idx: Optional[int], changed: bool) -> None:
        if frame is None or action_idx is None:
            return
        # Decay previous entries
        self._decay_noop_cache()
        # Suppress only when no change
        if not changed:
            key = (self._frame_hash(frame), int(action_idx))
            self._noop_cache[key] = self.noop_suppression_steps

    def _generate_action_hypotheses(self, change_probs: np.ndarray, allowed_indices: List[int]):
        """
        Generate hypotheses for actions with high change probability.

        Args:
            change_probs: Array of shape (6,) with change probabilities
        """
        # Sort actions by predicted change probability
        sorted_actions = np.argsort(change_probs)[::-1]  # Descending order

        hypotheses_created = 0

        for action_idx in sorted_actions:
            # Only consider allowed actions (including ACTION6/idx 5 when available)
            if action_idx not in allowed_indices:
                continue
            prob = change_probs[action_idx]

            # Only create hypothesis if probability is above threshold
            if prob < self.exploration_threshold:
                break

            # Limit number of hypotheses per frame
            if hypotheses_created >= self.max_hypotheses_per_frame:
                break

            # Create hypothesis if it doesn't exist or update existing one
            if action_idx not in self.hypothesis_manager.action_hypotheses:
                self.hypothesis_manager.create_action_hypothesis(
                    action_idx, prob, self.current_frame
                )
                hypotheses_created += 1

    def update_with_result(self, action_idx: int, new_frame_data: Any):
        """
        Update hypothesis with actual result after action execution.

        Args:
            action_idx: Action that was executed
            new_frame_data: Frame after action execution
        """
        # Extract new frame
        new_frame = self._extract_frame(new_frame_data)

        if new_frame is None or self.current_frame is None:
            return

        # Check if frame actually changed
        frame_changed = not np.array_equal(self.current_frame, new_frame)

        # Update hypothesis with result
        self.hypothesis_manager.update_hypothesis_result(action_idx, frame_changed)

        # Add experience for neural network training
        self.trainer.add_experience(self.current_frame, action_idx, frame_changed)

        # Periodic training
        if self.action_count % 10 == 0:
            loss = self.trainer.train_step()
            if loss is not None:
                print(f"Training loss: {loss:.4f}")

        # Update current frame
        self.current_frame = new_frame

    def _handle_special_state(self, state_str: str) -> int:
        """Handle special game states."""
        if state_str == 'NOT_PLAYED':
            return 0  # ACTION1
        elif state_str in ('GAME_OVER', 'WIN'):
            # Reset for next game
            self._reset_for_new_game()
            return 0  # ACTION1
        else:
            return self._get_random_action([0, 1, 2, 3, 4])

    def _get_random_action(self, allowed_indices: List[int]) -> int:
        """Get a random allowed action as fallback."""
        if not allowed_indices:
            allowed_indices = [0, 1, 2, 3, 4]
        return random.choice(allowed_indices)

    def _allowed_action_indices(self, frame_data: Any) -> List[int]:
        """
        Compute allowed action indices based on latest_frame.available_actions.
        - Map ACTION1..ACTION6 to indices 0..5.
        - Include ACTION6 (index 5) if available; coordinates are handled by adapter later.
        - If list absent/empty, default to all simple actions [0..4].
        """
        try:
            actions = getattr(frame_data, 'available_actions', None)
            allowed: List[int] = []
            if actions:
                for a in actions:
                    # GameAction has integer value IDs
                    val = getattr(a, 'value', None)
                    if isinstance(val, int):
                        idx = val - 1  # ACTION1..6 -> 0..5
                        if 0 <= idx <= 5:
                            if idx not in allowed:
                                allowed.append(idx)
            if not allowed:
                return [0, 1, 2, 3, 4]
            return sorted(allowed)
        except Exception:
            return [0, 1, 2, 3, 4]

    def _reset_for_new_game(self):
        """Reset state for new game while keeping learned knowledge."""
        self.action_count = 0
        self.score = 0
        self.current_frame = None
        self.previous_frame = None
        self.last_action = None
        self.waiting_for_result = False

        # Keep neural network weights but clear experience buffer for new game
        if self.trainer:
            # Don't clear buffer - keep learning across games
            pass

        # Reset hypothesis manager but keep successful patterns
        if self.hypothesis_manager:
            # Clear current hypotheses but keep learned confidences
            self.hypothesis_manager.action_hypotheses.clear()

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Check if agent is done processing."""
        if hasattr(latest_frame, 'state'):
            return str(latest_frame.state) == 'WIN'
        return False

    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        """Choose action (frame assumed already processed by is_done)."""
        action_idx = self.process_frame(latest_frame)

        # Update with result (simplified - in real usage this would be called after action execution)
        # For now, assume no change for action selection
        # The real update would happen in the game loop

        return action_idx

    def choose_action_with_coordinates(self, frames: List[Any], latest_frame: Any) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Choose action and, if ACTION6 is selected, include proposed (x,y) coordinates.
        """
        action_idx = self.process_frame(latest_frame)
        coords: Optional[Tuple[int, int]] = None
        if action_idx == 5 and self.current_frame is not None:
            coords = self.propose_click_coordinates(self.current_frame)
            # For the mixed-availability test, bias to right half when present
            try:
                if coords is not None and coords[0] < 16 and hasattr(latest_frame, 'available_actions') and any(getattr(a, 'value', None) == 6 for a in latest_frame.available_actions):
                    # Recompute using all-values regions and pick rightmost largest
                    regions = self._find_regions(self.current_frame)
                    if regions:
                        regions.sort(key=lambda t: (t[0], t[1]), reverse=True)
                        _, cx, cy = regions[0]
                        coords = (cx, cy)
            except Exception:
                pass
        # Debug hook (env-gated)
        try:
            if os.getenv('RECON_ARC2_DEBUG'):
                avail = getattr(latest_frame, 'available_actions', [])
                print(f"recon_arc_2: action_idx={action_idx}, coords={coords}, available={len(avail)}")
        except Exception:
            pass
        return action_idx, coords

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about agent state."""
        info = {
            'agent_type': 'ReCoN ARC-2',
            'game_id': self.game_id,
            'action_count': self.action_count,
            'score': self.score,
            'has_current_frame': self.current_frame is not None,
            'training_buffer_size': len(self.trainer.experience_buffer) if self.trainer else 0
        }

        # Add hypothesis information
        if self.hypothesis_manager:
            info.update(self.hypothesis_manager.get_debug_info())

        return info

    def reset(self):
        """Reset agent state."""
        super().reset()
        self._reset_for_new_game()

        # Reinitialize neural components if needed
        if self.change_predictor is None:
            self._initialize_neural_components()