"""
Base Agent Class for ReCoN Applications

Provides common functionality for agents built on the ReCoN platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np
from recon_engine import ReCoNGraph, ReCoNNode
from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
from recon_engine.neural_terminal import NeuralTerminal


class ReCoNBaseAgent(ABC):
    """
    Base class for agents that use ReCoN as their cognitive architecture.

    Provides common patterns for:
    - Graph construction and management
    - State tracking and transitions
    - Neural component integration
    - Message passing coordination
    """

    def __init__(self, agent_id: str, game_id: str = "default"):
        self.agent_id = agent_id
        self.game_id = game_id

        # Core ReCoN graph
        self.graph = ReCoNGraph()

        # State management
        self.current_state_key = None
        self.previous_state_key = None
        self.state_history = []

        # Agent lifecycle
        self.is_initialized = False
        self.step_count = 0

        # Build the agent's cognitive architecture
        self._build_architecture()

    @abstractmethod
    def _build_architecture(self):
        """Build the ReCoN graph representing the agent's cognitive architecture."""
        pass

    @abstractmethod
    def process_frame(self, frame_data: Any) -> Any:
        """Process a new frame and return an action."""
        pass

    def add_script_node(self, node_id: str, mode: NodeMode = NodeMode.EXPLICIT) -> HybridReCoNNode:
        """Add a script node to the agent's graph."""
        node = HybridReCoNNode(node_id, "script", mode)
        self.graph.add_node(node)
        return node

    def add_neural_terminal(self, node_id: str, model: torch.nn.Module,
                          output_mode: str = "value") -> NeuralTerminal:
        """Add a neural terminal to the agent's graph."""
        terminal = NeuralTerminal(node_id, model, output_mode)
        self.graph.add_node(terminal)
        return terminal

    def connect_nodes(self, parent_id: str, child_id: str, link_type: str):
        """Connect two nodes in the graph."""
        self.graph.add_link(parent_id, child_id, link_type)

    def execute_graph(self, root_node: str, max_steps: int = 10) -> Dict[str, Any]:
        """Execute the ReCoN graph starting from root node."""
        # Request the root node
        self.graph.request_root(root_node)

        # Execute propagation steps
        for step in range(max_steps):
            self.graph.propagate_step()
            if self.graph.is_completed():
                break

        # Extract results
        return self._extract_execution_results()

    def _extract_execution_results(self) -> Dict[str, Any]:
        """Extract results from graph execution."""
        results = {}

        # Get states of all nodes
        for node_id, node in self.graph.nodes.items():
            results[node_id] = {
                'state': str(node.state),
                'activation': getattr(node, 'activation', None)
            }

        return results

    def update_state_tracking(self, state_key: str, metadata: Optional[Dict] = None):
        """Update state tracking for the agent."""
        self.previous_state_key = self.current_state_key
        self.current_state_key = state_key

        # Record state transition
        if self.previous_state_key and self.current_state_key:
            transition = {
                'from': self.previous_state_key,
                'to': self.current_state_key,
                'step': self.step_count,
                'metadata': metadata or {}
            }
            self.state_history.append(transition)

        self.step_count += 1

    def get_node(self, node_id: str) -> Optional[ReCoNNode]:
        """Get a node from the graph."""
        return self.graph.get_node(node_id)

    def reset(self):
        """Reset the agent to initial state."""
        self.current_state_key = None
        self.previous_state_key = None
        self.state_history.clear()
        self.step_count = 0

        # Reset graph state
        self.graph.reset()

    def to_dict(self) -> Dict[str, Any]:
        """Export agent state for visualization/debugging."""
        return {
            'agent_id': self.agent_id,
            'game_id': self.game_id,
            'step_count': self.step_count,
            'current_state': self.current_state_key,
            'graph': self.graph.to_dict(),
            'state_history_length': len(self.state_history)
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the agent."""
        return {
            'total_steps': self.step_count,
            'unique_states_visited': len(set(s['to'] for s in self.state_history)),
            'graph_nodes': len(self.graph.nodes),
            'graph_links': len(self.graph.links),
        }


class StateTracker:
    """Helper class for tracking game states and transitions."""

    def __init__(self):
        self.states = {}  # state_key -> state_info
        self.transitions = {}  # (from_state, action) -> to_state
        self.milestones = {}  # (game_id, score) -> state_key

    def add_state(self, state_key: str, frame_data: Any, metadata: Optional[Dict] = None):
        """Add a new state to tracking."""
        self.states[state_key] = {
            'frame_data': frame_data,
            'visits': self.states.get(state_key, {}).get('visits', 0) + 1,
            'metadata': metadata or {},
            'actions_tried': set(),
            'future_states': {}
        }

    def add_transition(self, from_state: str, action: Any, to_state: str):
        """Record a state transition."""
        self.transitions[(from_state, action)] = to_state

        # Update forward references
        if from_state in self.states:
            self.states[from_state]['future_states'][action] = to_state
            self.states[from_state]['actions_tried'].add(action)

    def get_state_info(self, state_key: str) -> Optional[Dict]:
        """Get information about a state."""
        return self.states.get(state_key)

    def has_state(self, state_key: str) -> bool:
        """Check if state exists."""
        return state_key in self.states

    def get_future_states(self, state_key: str) -> Dict:
        """Get future states reachable from given state."""
        return self.states.get(state_key, {}).get('future_states', {})

    def add_milestone(self, game_id: str, score: int, state_key: str):
        """Mark a state as a milestone (level completion)."""
        self.milestones[(game_id, score)] = state_key

    def get_milestone(self, game_id: str, score: int) -> Optional[str]:
        """Get milestone state for given game and score."""
        return self.milestones.get((game_id, score))