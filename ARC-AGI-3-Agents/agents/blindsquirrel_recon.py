"""
BlindSquirrel ReCoN Agent - Harness Adapter

Adapter that connects the BlindSquirrel ReCoN agent to the ARC-AGI-3-Agents harness.
Maintains EXACT same flow as original BlindSquirrel implementation.
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
BlindSquirrelAgent = None


class BlindSquirrelReCoN(Agent):
    """
    BlindSquirrel agent adapter for ARC-AGI-3-Agents harness.
    
    Maintains EXACT flow as original:
    1. is_done() processes frame and checks WIN
    2. choose_action() assumes frame already processed by is_done()
    """
    
    # Match original BlindSquirrel MAX_ACTIONS (50000 instead of default 80)
    MAX_ACTIONS: int = 50000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blindsquirrel_agent = None

    def _ensure_agent(self):
        """Lazy initialization of the BlindSquirrel agent."""
        if self.blindsquirrel_agent is None:
            try:
                global BlindSquirrelAgent
                if BlindSquirrelAgent is None:
                    from recon_agents.blindsquirrel.agent import BlindSquirrelAgent

                self.blindsquirrel_agent = BlindSquirrelAgent("blindsquirrel", self.game_id)

            except Exception as e:
                print(f"Error initializing BlindSquirrel ReCoN agent: {e}")
                self.blindsquirrel_agent = None

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """
        Check if agent is done - EXACT original flow.
        
        Original: is_done() calls process_latest_frame() then checks WIN
        """
        self._ensure_agent()

        try:
            if self.blindsquirrel_agent:
                # Process frame first (like original)
                self.blindsquirrel_agent.process_latest_frame(latest_frame)
                
                # Check WIN condition (like original)
                if latest_frame.state is GameState.WIN:
                    return True
                return False
            else:
                return latest_frame.state == GameState.WIN
        except Exception as e:
            print(f"Error in is_done: {e}")
            # Fallback to simple WIN check
            return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Choose action - EXACT original flow.
        
        Original: choose_action() assumes frame already processed by is_done()
        """
        self._ensure_agent()

        # Handle special cases (same as original)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        try:
            if self.blindsquirrel_agent:
                # EXACT original logic: use epsilon-greedy selection
                current_state = self.blindsquirrel_agent.current_state
                if not current_state:
                    return self._get_fallback_action(latest_frame)

                # Original logic: check epsilon, score, and future states
                AGENT_E = getattr(self.blindsquirrel_agent, 'EPSILON', 0.5)
                use_model = (
                    AGENT_E < random.random() and 
                    latest_frame.score > 0 and 
                    len(current_state.future_states) > 0 and
                    hasattr(self.blindsquirrel_agent.state_graph, 'action_model') and
                    self.blindsquirrel_agent.state_graph.action_model is not None
                )

                if use_model:
                    action_idx = self.blindsquirrel_agent._get_model_action()
                else:
                    action_idx = self.blindsquirrel_agent._get_rweights_action()

                # Get action data from ReCoN implementation
                action_data = current_state.get_action_obj(action_idx)
                
                # Update state for next iteration (like original)
                self.blindsquirrel_agent.prev_state = current_state
                self.blindsquirrel_agent.prev_action = action_idx
                
                # Convert action data to GameAction enum for harness
                return self._convert_action_data(action_data)
            else:
                return self._get_fallback_action(latest_frame)

        except Exception as e:
            print(f"Error choosing action in BlindSquirrel ReCoN: {e}")
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
        """Get fallback action when BlindSquirrel agent fails."""
        # Simple fallback strategy
        if GameAction.ACTION1 in latest_frame.available_actions:
            return GameAction.ACTION1
        elif latest_frame.available_actions:
            return latest_frame.available_actions[0]
        else:
            return GameAction.RESET