# ReCoN ARC-2 Agent Architecture Overview

## Executive Summary

The ReCoN ARC-2 agent implements a **thin orchestrator pattern** with **pure Request Confirmation Networks (ReCoN)** for active perception-based action selection in ARC puzzles. The agent delegates all hypothesis testing and decision-making logic to a pure ReCoN hypothesis manager, maintaining clean separation of concerns between harness integration, frame orchestration, and active perception.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARC-AGI-3-Agents Harness                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Harness Adapter                                │    │
│  │          (recon_arc_2.py)                                  │    │
│  │                                                             │    │
│  │  • choose_action() -> delegates to agent                   │    │
│  │  • is_done() -> delegates to agent                         │    │
│  │  • Thin pass-through wrapper                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ReCoN ARC-2 Thin Agent                        │
│                        (agent.py)                                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Thin Orchestrator                              │    │
│  │                                                             │    │
│  │  1. Extract frames from game data                          │    │
│  │  2. Get CNN predictions for change probabilities           │    │
│  │  3. Feed α_valid/α_value priors to hypothesis manager      │    │
│  │  4. Single root request to "hypothesis_root"               │    │
│  │  5. Propagate ReCoN until action emerges                   │    │
│  │  6. Convert to GameAction + attach coordinates             │    │
│  │                                                             │    │
│  │  • NO manual state control                                 │    │
│  │  • NO hypothesis selection                                 │    │
│  │  • NO cooldown tracking                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                  │                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │          Neural Components                                  │    │
│  │                                                             │    │
│  │  • ChangePredictor: CNN for action→change probabilities    │    │
│  │  • ChangePredictorTrainer: Experience-based CNN training   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Pure ReCoN Hypothesis Manager                     │
│                      (hypothesis.py)                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ReCoN Graph Architecture                       │    │
│  │                                                             │    │
│  │           hypothesis_root (single request point)           │    │
│  │                        │                                   │    │
│  │              ┌─────────┴─────────┐                         │    │
│  │              ▼                   ▼                         │    │
│  │     action_hyp_1          action_hyp_2                     │    │
│  │    (ACTION1 test)        (ACTION2 test)                    │    │
│  │          │                       │                         │    │
│  │          ▼                       ▼                         │    │
│  │  action_hyp_1_term      action_hyp_2_term                  │    │
│  │  (measurement)          (measurement)                      │    │
│  │                                                             │    │
│  │  • Por/ret links: Natural inhibition between alternatives  │    │
│  │  • Gen loops: Failed state persistence (cooldown)         │    │
│  │  • Sub/sur links: Parent-child + confidence feedback      │    │
│  │  • Link weights: CNN priors modulate message flow         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Pure ReCoN Semantics                           │    │
│  │                                                             │    │
│  │  • States emerge from Table 1 FSM (8-state)               │    │
│  │  • Single root request propagates automatically            │    │
│  │  • α_valid → sub link weights (request delays)             │    │
│  │  • α_value → sur link weights (confirmation flow)          │    │
│  │  • Terminal measurements trigger confirmations             │    │
│  │  • Action emerges via get_selected_action()                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Data Flow: Frame → Action

### Phase 1: Harness Integration
```
ARC Game Environment
        ↓
   FrameData object
   • frame: 64x64 grid
   • state: PLAYING/WIN/GAME_OVER
   • score: current score
   • available_actions: valid actions
        ↓
Harness Adapter.choose_action()
   • Minimal pass-through
   • Error handling + debug logging
        ↓
Agent.choose_action() → Agent.process_frame()
```

### Phase 2: Agent Orchestration
```python
def process_frame(self, frame_data: Any) -> GameAction:
    # 1. Extract frame and handle game state
    self.current_frame = self._extract_frame(frame_data)

    # Handle score changes (new level reset)
    if frame_data.score != self.score:
        self.hypothesis_manager.reset_for_new_level()

    # Handle special states
    if frame_data.state in ('NOT_PLAYED', 'GAME_OVER', 'WIN'):
        return self._convert_to_game_action(0)  # RESET

    # 2. Get CNN predictions
    change_probs = self.change_predictor.predict_change_probabilities(
        self.current_frame  # 64x64 → [prob_0, prob_1, ..., prob_5]
    )

    # 3. Feed priors to hypothesis manager
    priors = {i: float(change_probs[i]) for i in range(6)}
    self.hypothesis_manager.feed_cnn_priors(
        alpha_valid=priors,  # → sub link weights
        alpha_value=priors   # → sur link weights
    )

    # Set availability constraints
    self.hypothesis_manager.set_available_actions(allowed_actions)

    # 4. Single root request
    self.hypothesis_manager.request_hypothesis_test("hypothesis_root")

    # 5. Propagate until action emerges
    for _ in range(20):  # Max iterations
        self.hypothesis_manager.propagate_step()
        selected_action = self.hypothesis_manager.get_selected_action()
        if selected_action is not None:
            break

    # 6. Convert and return
    return self._convert_to_game_action(selected_action)
```

### Phase 3: Pure ReCoN Hypothesis Testing

#### 3.1 Graph Structure
```
hypothesis_root
├── action_hyp_1 (ACTION1: "Move cursor up")
│   ├── gen → action_hyp_1 (failed state persistence)
│   └── sub → action_hyp_1_term (measurement node)
├── action_hyp_2 (ACTION2: "Move cursor down")
│   ├── por ← action_hyp_1 (inhibition from higher priority)
│   ├── ret → action_hyp_1 (retro-inhibition)
│   ├── gen → action_hyp_2 (failed state persistence)
│   └── sub → action_hyp_2_term (measurement node)
└── ... (ACTION3, ACTION4, ACTION5, ACTION6)
```

#### 3.2 ReCoN Message Passing Flow
```
1. request_hypothesis_test("hypothesis_root")
   └→ REQUESTED state on hypothesis_root

2. propagate_step() × N iterations:

   Step 1: Root → Children
   hypothesis_root (REQUESTED)
   └─sub_links→ action_hyp_* (REQUESTED)

   Step 2: Priority Resolution
   • High α_valid → faster request propagation
   • High α_value → stronger confirmation feedback
   • Por links inhibit lower-priority alternatives

   Step 3: Terminal Measurements
   action_hyp_i → ACTIVE → measurement_node
   • If measurement set: CONFIRMED/FAILED
   • If no measurement: stays ACTIVE/WAITING

   Step 4: State Emergence
   • CONFIRMED: High confidence, ready for selection
   • TRUE: Measurement confirmed hypothesis
   • ACTIVE/WAITING: Still testing
   • FAILED: Measurement rejected hypothesis
   • SUPPRESSED: Inhibited by por links
```

#### 3.3 Action Selection Algorithm
```python
def get_selected_action(self) -> Optional[int]:
    # State priority ranking (higher = more advanced)
    priority = {
        CONFIRMED: 7,    # Best: confirmed by measurement
        TRUE: 6,         # Good: measurement supports hypothesis
        WAITING: 5,      # Decent: waiting for measurement
        ACTIVE: 4,       # OK: actively being tested
        REQUESTED: 3,    # Meh: just requested
        SUPPRESSED: 2,   # Poor: inhibited by alternatives
        FAILED: 1,       # Bad: measurement rejected
        INACTIVE: 0      # Worst: not engaged
    }

    # Find action with highest state priority
    # Tie-break by confidence (CNN prediction + testing history)
    # Respect availability constraints
    return best_action_idx
```

## Core ReCoN Architecture Details

### Node Types

#### 1. ActionHypothesis Nodes
```python
class ActionHypothesis(ReCoNNode):
    """
    Hypothesis: "Action X will be productive in this context"
    """
    def __init__(self, action_idx: int, predicted_prob: float):
        self.action_idx = action_idx  # 0-5 for ACTION1-ACTION6
        self.predicted_change_prob = predicted_prob  # CNN prediction
        self.tested = False
        self.confirmation_count = 0  # Successful tests
        self.failure_count = 0       # Failed tests

    def get_confidence(self) -> float:
        if not self.tested:
            return self.predicted_change_prob  # Pure CNN
        # Blend CNN + empirical evidence
        empirical = self.confirmation_count / (confirmation + failure)
        weight = min(total_tests / 10.0, 0.8)
        return weight * empirical + (1 - weight) * self.predicted_change_prob
```

#### 2. Terminal Measurement Nodes
```python
class TerminalMeasurementNode(ReCoNNode):
    """
    Performs measurements when action hypothesis is tested
    """
    def set_measurement(self, changed: bool):
        self._measurement = changed

    def measure(self) -> float:
        return 1.0 if self._measurement else 0.0
```

### Link Types and Semantics

#### Sub/Sur Links (Parent-Child)
```
hypothesis_root --sub--> action_hyp_1
action_hyp_1 --sur--> hypothesis_root

• Sub: Parent requests child testing
• Sur: Child reports back to parent
• Weight = α_valid (sub) / α_value (sur)
• Higher weights = faster/stronger flow
```

#### Por/Ret Links (Inhibition)
```
action_hyp_1 --por--> action_hyp_2
action_hyp_2 --ret--> action_hyp_1

• Por: Inhibits target until source completes
• Ret: Retro-inhibits source when target activates
• Creates natural priority ordering
```

#### Gen Links (Persistence)
```
action_hyp_1 --gen--> action_hyp_1

• Self-loop maintains FAILED states
• Provides natural "cooldown" mechanism
• No Python-side timer management
```

### CNN Prior Integration

#### α_valid (Request Delays)
```python
# CNN predicts action validity
alpha_valid = {
    0: 0.8,  # ACTION1: high validity → fast request
    1: 0.3,  # ACTION2: low validity → slow request
    ...
}
# Applied as sub link weights
sub_link.weight = alpha_valid[action_idx]
```

#### α_value (Confirmation Strength)
```python
# CNN predicts action value/reward
alpha_value = {
    0: 0.9,  # ACTION1: high value → strong confirmation
    1: 0.2,  # ACTION2: low value → weak confirmation
    ...
}
# Applied as sur link weights
sur_link.weight = alpha_value[action_idx]
```

## Learning and Adaptation

### Experience Collection
```python
def _update_learning(self, action_idx: int):
    # Check if frame changed after action
    frame_changed = not np.array_equal(current_frame, previous_frame)

    # Update hypothesis statistics
    self.hypothesis_manager.update_hypothesis_result(action_idx, frame_changed)

    # Add CNN training experience
    self.trainer.add_experience(previous_frame, action_idx, frame_changed)

    # Periodic CNN training (every 20 actions)
    if self.action_count % 20 == 0:
        self.trainer.train_step()
```

### Hypothesis Testing Feedback Loop
```
Action Execution → Frame Change Detection → Terminal Measurement
     ↑                                              ↓
GameAction ← ReCoN Selection ← State Update ← Confirmation/Failure
```

## Coordinate Handling (ACTION6)

When ACTION6 (click) is selected:

```python
def _convert_to_game_action(self, action_idx: int) -> GameAction:
    action = GameAction.from_id(action_idx)

    if action.value == 6:  # ACTION6
        x, y = self.propose_click_coordinates(self.current_frame)
        action.set_data({"x": int(x), "y": int(y)})

    return action

def propose_click_coordinates(self, frame: np.ndarray) -> Tuple[int, int]:
    # Delegate to hypothesis manager's region analysis
    if hasattr(self.hypothesis_manager, 'propose_click_coordinates'):
        return self.hypothesis_manager.propose_click_coordinates(frame)
    # Fallback: center of frame
    return 32, 32
```

## Key Design Principles

### 1. **Thin Orchestrator Pattern**
- Agent has minimal logic, delegates everything to components
- No manual state management or hypothesis selection
- Pure coordination between CNN and ReCoN

### 2. **Pure ReCoN Semantics**
- States emerge from message passing, not Python control
- Single root request drives all activity
- Link weights (not gates) control flow
- Natural inhibition via por/ret links

### 3. **Clean Separation of Concerns**
```
Harness Adapter: Type conversion + error handling
Agent: Frame orchestration + CNN integration
Hypothesis Manager: Pure ReCoN active perception
CNN: Change prediction + learning
```

### 4. **No Logic Duplication**
- Special state handling: Agent only
- Action conversion: Agent only
- Coordinate attachment: Agent only
- Hypothesis testing: ReCoN only

## Performance Characteristics

### Computational Complexity
- **Frame Processing**: O(1) - constant orchestration steps
- **CNN Inference**: O(64×64) - single forward pass
- **ReCoN Propagation**: O(N×L) - N nodes, L links per step
- **Action Selection**: O(N) - linear scan of hypotheses

### Memory Usage
- **Frame Storage**: 64×64×2 (current + previous)
- **CNN Model**: ~1MB parameters
- **ReCoN Graph**: ~10 nodes, ~30 links per level
- **Experience Buffer**: Configurable, default 1000 samples

### Typical Performance
- **Frame→Action**: ~10ms (CNN + ReCoN propagation)
- **Action Emergence**: 1-5 ReCoN propagation steps
- **Level Reset**: <1ms (hypothesis cleanup)

## Error Handling and Robustness

### Graceful Degradation
```python
# If no action emerges from ReCoN
if selected_action is None:
    selected_action = self._get_random_action(available_actions)

# If hypothesis manager fails
try:
    return self.recon_arc2_agent.choose_action(frames, latest_frame)
except Exception:
    return self._get_fallback_action(latest_frame)
```

### State Recovery
- New level detection resets all hypotheses
- Failed ReCoN propagation falls back to random action
- CNN prediction errors default to uniform priors

This architecture achieves the goal of pure ReCoN active perception while maintaining clean integration with the ARC-AGI-3 harness and providing robust real-time performance.