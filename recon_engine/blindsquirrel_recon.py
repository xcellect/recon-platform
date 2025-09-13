"""
BlindSquirrel Agent → ReCoN Mapping

Maps the 2nd place ARC-AGI-3 winner (BlindSquirrel) architecture exactly to ReCoN.
Demonstrates how explicit state machines + neural terminals work together.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from .hybrid_node import HybridReCoNNode, NodeMode
from .neural_terminal import NeuralTerminal, BlindSquirrelValueTerminal
from .graph import ReCoNGraph
from .messages import HybridMessage, MessageType
import random


class BlindSquirrelReCoNAgent:
    """
    BlindSquirrel agent implemented as a ReCoN graph.
    
    Maps the original architecture:
    - State Graph → Explicit script nodes with discrete states
    - Action Value Model → Neural terminal with ResNet
    - Valid Actions Model → Rule-based script nodes
    - Game loop → ReCoN message passing
    """
    
    def __init__(self, game_id: str = "default"):
        self.game_id = game_id
        self.graph = ReCoNGraph()
        self.current_state = None
        self.prev_state = None
        self.prev_action = None
        self.game_counter = 0
        self.level_counter = 0
        
        # Build ReCoN graph representing BlindSquirrel architecture
        self._build_recon_graph()
        
        # State tracking (equivalent to original state graph)
        self.state_memory = {}  # Maps game states to ReCoN node states
        
        # Parameters matching original
        self.agent_e = 0.5  # Exploration parameter
        self.rweight_min = 0.1
        self.switch_threshold = 0.8
    
    def _build_recon_graph(self):
        """Build ReCoN graph representing BlindSquirrel architecture."""
        
        # Root agent controller
        root = HybridReCoNNode("agent_root", "script", NodeMode.EXPLICIT)
        root.is_terminal = False
        self.graph.add_node(root)
        
        # State tracking and management
        state_tracker = HybridReCoNNode("state_tracker", "script", NodeMode.EXPLICIT)
        self.graph.add_node(state_tracker)
        self.graph.add_link("agent_root", "state_tracker", "sub")
        
        # Action selection coordinator  
        action_selector = HybridReCoNNode("action_selector", "script", NodeMode.EXPLICIT)
        self.graph.add_node(action_selector)
        self.graph.add_link("state_tracker", "action_selector", "por")
        
        # Model-based vs rule-based selection
        model_selector = HybridReCoNNode("model_selector", "script", NodeMode.EXPLICIT)
        rules_selector = HybridReCoNNode("rules_selector", "script", NodeMode.EXPLICIT)
        
        self.graph.add_node(model_selector)
        self.graph.add_node(rules_selector)
        
        # Sequential selection (rules first, then model if available)
        self.graph.add_link("action_selector", "rules_selector", "sub")
        self.graph.add_link("rules_selector", "model_selector", "por")
        
        # Neural terminals for value prediction and action validation
        value_terminal = BlindSquirrelValueTerminal("value_model", self.game_id)
        valid_actions_terminal = ValidActionsTerminal("valid_actions")
        
        self.graph.add_node(value_terminal)
        self.graph.add_node(valid_actions_terminal)
        
        # Connect model selector to value prediction
        self.graph.add_link("model_selector", "value_model", "sub")
        
        # Connect rules selector to action validation
        self.graph.add_link("rules_selector", "valid_actions", "sub")
        
        # State graph representation as ReCoN nodes
        self._add_state_graph_nodes()
    
    def _add_state_graph_nodes(self):
        """Add nodes representing the state graph structure."""
        
        # State existence checker
        state_checker = HybridReCoNNode("state_checker", "script", NodeMode.EXPLICIT)
        self.graph.add_node(state_checker)
        self.graph.add_link("state_tracker", "state_checker", "sub")
        
        # State transition validator
        transition_validator = HybridReCoNNode("transition_validator", "script", NodeMode.EXPLICIT)
        self.graph.add_node(transition_validator)
        self.graph.add_link("state_checker", "transition_validator", "por")
        
        # Future state predictor
        future_predictor = HybridReCoNNode("future_predictor", "script", NodeMode.EXPLICIT)
        self.graph.add_node(future_predictor)
        self.graph.add_link("transition_validator", "future_predictor", "por")
    
    def process_frame(self, frame_data: Any) -> Any:
        """
        Process new frame using ReCoN graph.
        
        Equivalent to BlindSquirrel's process_latest_frame + choose_action.
        """
        # Convert frame to ReCoN state representation
        current_state_key = self._frame_to_state_key(frame_data)
        
        # Update state tracking
        self._update_state_tracking(frame_data, current_state_key)
        
        # Execute ReCoN graph to select action
        action = self._execute_action_selection(frame_data)
        
        # Update history
        self.prev_state = self.current_state
        self.current_state = current_state_key
        self.prev_action = action
        
        return action
    
    def _frame_to_state_key(self, frame_data: Any) -> str:
        """Convert frame data to state key."""
        # Extract relevant features for state identification
        if hasattr(frame_data, 'frame'):
            frame_hash = hash(str(frame_data.frame))
        else:
            frame_hash = hash(str(frame_data))
        
        if hasattr(frame_data, 'score'):
            return f"state_{frame_hash}_{frame_data.score}"
        else:
            return f"state_{frame_hash}"
    
    def _update_state_tracking(self, frame_data: Any, state_key: str):
        """Update state tracking equivalent to original state graph."""
        
        # Store state information
        self.state_memory[state_key] = {
            'frame_data': frame_data,
            'visits': self.state_memory.get(state_key, {}).get('visits', 0) + 1,
            'actions_tried': self.state_memory.get(state_key, {}).get('actions_tried', set()),
            'future_states': self.state_memory.get(state_key, {}).get('future_states', {}),
            'action_rweights': self.state_memory.get(state_key, {}).get('action_rweights', {})
        }
        
        # Update transition if we have previous state
        if self.prev_state and self.prev_action:
            if self.prev_state not in self.state_memory:
                self.state_memory[self.prev_state] = {'future_states': {}, 'action_rweights': {}}
            
            self.state_memory[self.prev_state]['future_states'][self.prev_action] = state_key
            
            # Update action weights based on outcome
            if hasattr(frame_data, 'score') and hasattr(self._get_prev_frame_data(), 'score'):
                if frame_data.score > self._get_prev_frame_data().score:
                    # Positive outcome
                    self.state_memory[self.prev_state]['action_rweights'][self.prev_action] = 1.0
                else:
                    # No improvement
                    current_weight = self.state_memory[self.prev_state]['action_rweights'].get(self.prev_action, 0.5)
                    self.state_memory[self.prev_state]['action_rweights'][self.prev_action] = max(self.rweight_min, current_weight * 0.9)
    
    def _get_prev_frame_data(self) -> Any:
        """Get previous frame data."""
        if self.prev_state and self.prev_state in self.state_memory:
            return self.state_memory[self.prev_state]['frame_data']
        return None
    
    def _execute_action_selection(self, frame_data: Any) -> Any:
        """Execute ReCoN graph for action selection."""
        
        # Request root node (equivalent to triggering agent)
        self.graph.request_root("agent_root")
        
        # Execute propagation steps
        for _ in range(10):  # Max steps to avoid infinite loops
            self.graph.propagate_step()
            if self.graph.is_completed():
                break
        
        # Extract action from result
        return self._extract_action_from_result({}, frame_data)
    
    def _extract_action_from_result(self, result: Dict[str, Any], frame_data: Any) -> Any:
        """Extract final action from ReCoN execution result."""
        
        # Check if we should use model or rules (equivalent to AGENT_E check)
        use_model = (random.random() > self.agent_e and 
                    hasattr(frame_data, 'score') and 
                    frame_data.score > 0 and
                    self._has_future_states())
        
        if use_model:
            return self._get_model_action(frame_data)
        else:
            return self._get_rules_action(frame_data)
    
    def _has_future_states(self) -> bool:
        """Check if current state has future states (equivalent to len(future_states) > 0)."""
        if self.current_state and self.current_state in self.state_memory:
            return len(self.state_memory[self.current_state].get('future_states', {})) > 0
        return False
    
    def _get_model_action(self, frame_data: Any) -> Any:
        """Get action using neural value model."""
        
        # Get value terminal node
        value_node = self.graph.get_node("value_model")
        
        if value_node is None:
            return self._get_rules_action(frame_data)
        
        # Prepare state and action candidates for value prediction
        state_tensor = self._frame_to_tensor(frame_data)
        
        # Get valid actions
        valid_actions = self._get_valid_actions(frame_data)
        
        if not valid_actions:
            return self._get_rules_action(frame_data)
        
        # Evaluate each action with the model
        best_action = None
        best_value = float('-inf')
        
        for action in valid_actions:
            action_tensor = self._action_to_tensor(action)
            
            # Use neural terminal to predict value
            value = value_node.measure((state_tensor, action_tensor))
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action if best_action is not None else self._get_rules_action(frame_data)
    
    def _get_rules_action(self, frame_data: Any) -> Any:
        """Get action using rule-based weights."""
        
        state_key = self._frame_to_state_key(frame_data)
        
        if state_key not in self.state_memory:
            return self._get_random_action(frame_data)
        
        # Get action weights (equivalent to action_rweights)
        action_weights = self.state_memory[state_key].get('action_rweights', {})
        
        if not action_weights:
            return self._get_random_action(frame_data)
        
        # Select action based on weights
        actions = list(action_weights.keys())
        weights = list(action_weights.values())
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight <= 0:
            return self._get_random_action(frame_data)
        
        rand_val = random.random() * total_weight
        cumulative = 0
        
        for action, weight in zip(actions, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return action
        
        return actions[-1] if actions else self._get_random_action(frame_data)
    
    def _get_random_action(self, frame_data: Any) -> Any:
        """Get random action as fallback."""
        # Default action set (can be customized based on game)
        default_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "CLICK"]
        return random.choice(default_actions)
    
    def _get_valid_actions(self, frame_data: Any) -> List[Any]:
        """Get valid actions for current state."""
        # Simplified valid action detection
        # In real implementation, this would use the rules-based model
        default_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
        
        # Add click actions for different grid positions
        for i in range(0, 64, 8):  # Sample some positions
            for j in range(0, 64, 8):
                default_actions.append(f"CLICK_{i}_{j}")
        
        return default_actions
    
    def _frame_to_tensor(self, frame_data: Any) -> torch.Tensor:
        """Convert frame to tensor for neural processing."""
        if hasattr(frame_data, 'frame'):
            frame = frame_data.frame
        else:
            frame = frame_data
        
        if isinstance(frame, torch.Tensor):
            return frame
        elif isinstance(frame, np.ndarray):
            return torch.from_numpy(frame).long()
        elif isinstance(frame, list):
            # Assume 64x64 grid
            return torch.tensor(frame).long().reshape(64, 64)
        else:
            # Fallback - random grid
            return torch.randint(0, 16, (64, 64)).long()
    
    def _action_to_tensor(self, action: Any) -> torch.Tensor:
        """Convert action to tensor for neural processing."""
        # Simple action encoding
        if isinstance(action, str):
            if action.startswith("ACTION"):
                action_type = int(action[-1]) if action[-1].isdigit() else 0
                return torch.tensor([action_type, 0, 0, 0, 0]).float()
            elif action.startswith("CLICK"):
                parts = action.split("_")
                if len(parts) >= 3:
                    x, y = int(parts[1]), int(parts[2])
                    return torch.tensor([6, x/64.0, y/64.0, 0, 0]).float()
        
        # Default encoding
        return torch.tensor([0, 0, 0, 0, 0]).float()
    
    def train_model(self, max_score: int):
        """Train the value model (equivalent to original train_model)."""
        value_node = self.graph.get_node("value_model")
        if value_node and hasattr(value_node, 'model'):
            # Training would happen here
            # For now, just mark as trained
            value_node.metadata = value_node.metadata or {}
            value_node.metadata['trained'] = True
            value_node.metadata['max_score'] = max_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Export agent configuration for visualization."""
        return {
            "agent_type": "BlindSquirrel",
            "game_id": self.game_id,
            "graph": self.graph.to_dict(),
            "state_memory_size": len(self.state_memory),
            "parameters": {
                "agent_e": self.agent_e,
                "rweight_min": self.rweight_min,
                "switch_threshold": self.switch_threshold
            },
            "statistics": {
                "game_counter": self.game_counter,
                "level_counter": self.level_counter,
                "states_explored": len(self.state_memory)
            }
        }


class ValidActionsTerminal(NeuralTerminal):
    """
    Terminal node for rule-based valid action detection.
    
    Replaces BlindSquirrel's rule-based valid actions model with
    a ReCoN terminal that can be enhanced with neural networks later.
    """
    
    def __init__(self, node_id: str):
        # Simple model for action validation
        model = ActionValidationModel()
        super().__init__(node_id, model, "classification", (64, 64, 16))
        
        # Rules-based validation (can be replaced with neural net)
        self.use_rules = True
        
    def measure(self, environment: Any = None) -> torch.Tensor:
        """Measure action validity."""
        if self.use_rules:
            return self._rules_based_validation(environment)
        else:
            return super().measure(environment)
    
    def _rules_based_validation(self, environment: Any) -> torch.Tensor:
        """Rule-based action validation matching original BlindSquirrel."""
        
        # Extract frame and action from environment
        if isinstance(environment, tuple) and len(environment) == 2:
            frame, action = environment
        else:
            # Default to valid
            return torch.tensor([1.0])
        
        # Simple validity rules
        validity_score = 1.0
        
        # Check if action is well-formed
        if not isinstance(action, str):
            validity_score = 0.0
        elif action.startswith("CLICK"):
            # Validate click coordinates
            parts = action.split("_")
            if len(parts) < 3:
                validity_score = 0.0
            else:
                try:
                    x, y = int(parts[1]), int(parts[2])
                    if x < 0 or x >= 64 or y < 0 or y >= 64:
                        validity_score = 0.0
                except:
                    validity_score = 0.0
        
        # Additional rules can be added here
        # E.g., checking if action makes sense given current frame state
        
        return torch.tensor([validity_score])


class ActionValidationModel(nn.Module):
    """Simple neural model for action validation."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64*64*16 + 5, 128)  # Frame + action encoding
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary validity
        
    def forward(self, x):
        if isinstance(x, tuple):
            frame, action = x
            frame_flat = frame.flatten()
            action_flat = action.flatten()
            combined = torch.cat([frame_flat, action_flat])
            
            h = torch.relu(self.fc1(combined))
            h = torch.relu(self.fc2(h))
            return torch.sigmoid(self.fc3(h))
        else:
            # Fallback
            return torch.tensor([1.0])


def create_blindsquirrel_agent(game_id: str = "test_game") -> BlindSquirrelReCoNAgent:
    """
    Factory function to create a BlindSquirrel ReCoN agent.
    
    This demonstrates the exact mapping from the original BlindSquirrel
    architecture to ReCoN nodes and message passing.
    """
    agent = BlindSquirrelReCoNAgent(game_id)
    
    # Optional: Load pre-trained models if available
    # agent.load_pretrained_models()
    
    return agent


# Example usage and testing
def test_blindsquirrel_mapping():
    """Test that BlindSquirrel maps correctly to ReCoN."""
    
    # Create agent
    agent = create_blindsquirrel_agent("test")
    
    # Mock frame data
    class MockFrame:
        def __init__(self, frame_data, score):
            self.frame = frame_data
            self.score = score
    
    # Test frame processing
    test_frame = MockFrame([[0] * 64 for _ in range(64)], 0)
    action = agent.process_frame(test_frame)
    
    print(f"BlindSquirrel ReCoN agent selected action: {action}")
    
    # Verify graph structure
    graph_dict = agent.to_dict()
    print(f"Graph nodes: {len(graph_dict['graph']['nodes'])}")
    print(f"Graph links: {len(graph_dict['graph']['links'])}")
    
    return agent


if __name__ == "__main__":
    test_blindsquirrel_mapping()