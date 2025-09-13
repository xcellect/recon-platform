"""
ReCoN ARC-1 Agent - Harness Adapter

Adapter that connects the ReCoN ARC-1 agent to the ARC-AGI-3-Agents harness.
Maintains EXACT same flow as BlindSquirrel implementation.
"""

import sys
import os
import random
from typing import Any, List, Optional

# Add recon-platform to path
sys.path.insert(0, '/workspace/recon-platform')

from .agent import Agent
from .structs import FrameData, GameAction, GameState

# Lazy import to avoid import errors during agent registration
ReCoNArc1Agent = None


class ReCoNArc1(Agent):
    """
    ReCoN ARC-1 agent adapter for ARC-AGI-3-Agents harness.

    Maintains EXACT flow as BlindSquirrel:
    1. is_done() processes frame and checks WIN
    2. choose_action() assumes frame already processed by is_done()
    """

    # Match BlindSquirrel MAX_ACTIONS
    MAX_ACTIONS: int = 50000

    def __init__(self, card_id: str, game_id: str, agent_name: str, ROOT_URL: str, record: bool, *args, **kwargs):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, *args, **kwargs)
        self.recon_arc1_agent = None

    def _ensure_agent(self):
        """Lazy initialization of the ReCoN ARC-1 agent."""
        if self.recon_arc1_agent is None:
            try:
                global ReCoNArc1Agent
                if ReCoNArc1Agent is None:
                    from recon_agents.recon_arc_1.agent import ReCoNArc1Agent

                self.recon_arc1_agent = ReCoNArc1Agent("recon_arc_1", self.game_id)

            except Exception as e:
                print(f"Error initializing ReCoN ARC-1 agent: {e}")
                import traceback
                traceback.print_exc()
                self.recon_arc1_agent = None

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """
        Check if agent is done - EXACT BlindSquirrel flow.

        Process frame first then check WIN condition.
        """
        self._ensure_agent()

        try:
            if self.recon_arc1_agent:
                # Process frame first (like BlindSquirrel)
                self.recon_arc1_agent.process_latest_frame(latest_frame)

                # Check WIN condition
                if latest_frame.state is GameState.WIN:
                    return True
                return False
            else:
                return latest_frame.state == GameState.WIN
        except Exception as e:
            print(f"Error in ReCoN ARC-1 is_done: {e}")
            import traceback
            traceback.print_exc()
            return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Choose action - EXACT BlindSquirrel flow.

        Assumes frame already processed by is_done().
        """
        self._ensure_agent()

        # Handle special cases (same as BlindSquirrel)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        try:
            if self.recon_arc1_agent:
                # Get current state (should be already processed)
                current_state = self.recon_arc1_agent.current_state
                if not current_state:
                    return self._get_fallback_action(latest_frame)

                # Use epsilon-greedy selection (identical to BlindSquirrel)
                AGENT_E = getattr(self.recon_arc1_agent, 'EPSILON', 0.5)
                use_model = (
                    AGENT_E < random.random() and
                    latest_frame.score > 0 and
                    len(current_state.future_states) > 0 and
                    hasattr(self.recon_arc1_agent.state_graph, 'action_model') and
                    self.recon_arc1_agent.state_graph.action_model is not None
                )

                if use_model:
                    action_idx = self.recon_arc1_agent._get_model_action()
                else:
                    action_idx = self.recon_arc1_agent._get_rweights_action()

                # Get action data from ReCoN implementation
                action_data = current_state.get_action_obj(action_idx)

                # Update state for next iteration
                self.recon_arc1_agent.prev_state = current_state
                self.recon_arc1_agent.prev_action = action_idx

                # Convert action data to GameAction enum for harness
                return self._convert_action_data(action_data)
            else:
                return self._get_fallback_action(latest_frame)

        except Exception as e:
            print(f"Error choosing action in ReCoN ARC-1: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_action(latest_frame)

    def _convert_action_data(self, action_data: Any) -> GameAction:
        """Convert ReCoN action data to GameAction enum for harness."""
        if isinstance(action_data, dict):
            if action_data.get('type') == 'basic':
                action_id = action_data.get('action_id', 0)
                if action_id == 0:
                    return GameAction.ACTION1
                elif action_id == 1:
                    return GameAction.ACTION2
                elif action_id == 2:
                    return GameAction.ACTION3
                elif action_id == 3:
                    return GameAction.ACTION4
                elif action_id == 4:
                    return GameAction.ACTION5
                else:
                    return GameAction.ACTION1
            elif action_data.get('type') == 'click':
                # For click actions, create ACTION6 with position data
                click_action = GameAction.ACTION6
                if 'x' in action_data and 'y' in action_data:
                    click_action.set_data({"x": action_data['x'], "y": action_data['y']})
                return click_action
            else:
                return GameAction.ACTION1
        elif isinstance(action_data, int):
            # Direct action index mapping
            if action_data == 0:
                return GameAction.ACTION1
            elif action_data == 1:
                return GameAction.ACTION2
            elif action_data == 2:
                return GameAction.ACTION3
            elif action_data == 3:
                return GameAction.ACTION4
            elif action_data == 4:
                return GameAction.ACTION5
            else:
                return GameAction.ACTION6  # Click actions
        else:
            return GameAction.ACTION1

    def _get_fallback_action(self, latest_frame: FrameData) -> GameAction:
        """Get fallback action when ReCoN ARC-1 agent fails."""
        # Simple fallback strategy
        if GameAction.ACTION1 in latest_frame.available_actions:
            return GameAction.ACTION1
        elif latest_frame.available_actions:
            return latest_frame.available_actions[0]
        else:
            return GameAction.RESET

    def get_debug_info(self) -> dict:
        """Get debug information from the agent."""
        if self.recon_arc1_agent and hasattr(self.recon_arc1_agent, 'get_debug_info'):
            return self.recon_arc1_agent.get_debug_info()
        else:
            return {'agent_type': 'ReCoN ARC-1', 'status': 'not_initialized'}

    def reset(self):
        """Reset the agent."""
        if self.recon_arc1_agent:
            self.recon_arc1_agent.reset()