"""
ReCoN ARC-2 Agent - Harness Adapter

Adapter that connects the ReCoN ARC-2 agent to the ARC-AGI-3-Agents harness.
Uses active perception with hypothesis testing for puzzle solving.
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
ReCoNArc2Agent = None


class ReCoNArc2(Agent):
    """
    ReCoN ARC-2 agent adapter for ARC-AGI-3-Agents harness.

    Uses active perception with hypothesis testing to solve ARC puzzles.
    """

    MAX_ACTIONS: int = 50000

    def __init__(self, card_id: str, game_id: str, agent_name: str, ROOT_URL: str, record: bool, *args, **kwargs):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, *args, **kwargs)
        self.recon_arc2_agent = None

    def _ensure_agent(self):
        """Lazy initialization of the ReCoN ARC-2 agent."""
        if self.recon_arc2_agent is None:
            try:
                global ReCoNArc2Agent
                if ReCoNArc2Agent is None:
                    from recon_agents.recon_arc_2.agent import ReCoNArc2Agent

                self.recon_arc2_agent = ReCoNArc2Agent("recon_arc_2", self.game_id)

            except Exception as e:
                print(f"Error initializing ReCoN ARC-2 agent: {e}")
                import traceback
                traceback.print_exc()
                self.recon_arc2_agent = None

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """
        Check if agent is done - EXACT working pattern.
        
        Process frame first (sets self.guid), then check WIN condition.
        """
        self._ensure_agent()

        try:
            if self.recon_arc2_agent:
                # Process frame first (like BlindSquirrel)
                self.recon_arc2_agent.process_latest_frame(latest_frame)
                
                # Check WIN condition
                if latest_frame.state is GameState.WIN:
                    return True
                return False
            else:
                return latest_frame.state == GameState.WIN
        except Exception as e:
            print(f"Error in ReCoN ARC-2 is_done: {e}")
            import traceback
            traceback.print_exc()
            return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Choose action - EXACT working pattern.
        
        Assumes frame already processed by is_done().
        """
        self._ensure_agent()

        # Handle special cases (same as BlindSquirrel)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        try:
            if self.recon_arc2_agent:
                # Get action from agent (frame already processed in is_done)
                action_result = self.recon_arc2_agent.get_action_from_processed_frame()
                
                # Convert result to GameAction
                action = self._convert_action_result(action_result, latest_frame)

                # Env-gated debug hook
                try:
                    if os.getenv('RECON_ARC2_DEBUG'):
                        avail_len = len(getattr(latest_frame, 'available_actions', []) or [])
                        action_data = getattr(action, 'action_data', None)
                        coords_info = ""
                        if action_data and hasattr(action_data, 'x') and hasattr(action_data, 'y'):
                            coords_info = f" x={action_data.x} y={action_data.y}"
                        print(f"recon_arc_2(adapter): action={action.name}, coords=game_id='{self.game_id}'{coords_info}, available={avail_len}")
                except Exception:
                    pass

                return action
            else:
                return self._get_fallback_action(latest_frame)

        except Exception as e:
            print(f"Error choosing action in ReCoN ARC-2: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_action(latest_frame)

    def _convert_action_result(self, action_result: Any, latest_frame: FrameData) -> GameAction:
        """Convert ReCoN action result to GameAction enum for harness."""
        if isinstance(action_result, dict):
            if action_result.get('type') == 'basic':
                action_id = action_result.get('action_id', 0)
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
            elif action_result.get('type') == 'click':
                # For click actions, create ACTION6 with position data
                click_action = GameAction.ACTION6
                if 'x' in action_result and 'y' in action_result:
                    click_action.set_data({"x": action_result['x'], "y": action_result['y']})
                return click_action
            else:
                return GameAction.ACTION1
        elif hasattr(action_result, 'value'):
            # Already a GameAction
            return action_result
        elif isinstance(action_result, int):
            # Direct action index mapping (hypothesis index to GameAction)
            if action_result == 0:
                return GameAction.ACTION1
            elif action_result == 1:
                return GameAction.ACTION2
            elif action_result == 2:
                return GameAction.ACTION3
            elif action_result == 3:
                return GameAction.ACTION4
            elif action_result == 4:
                return GameAction.ACTION5
            elif action_result == 5:
                # ACTION6 - need coordinates from agent
                click_action = GameAction.ACTION6
                if self.recon_arc2_agent and hasattr(self.recon_arc2_agent, 'current_frame') and self.recon_arc2_agent.current_frame is not None:
                    x, y = self.recon_arc2_agent.propose_click_coordinates(self.recon_arc2_agent.current_frame)
                    click_action.set_data({"x": int(x), "y": int(y)})
                else:
                    click_action.set_data({"x": 32, "y": 32})  # Default center
                return click_action
            else:
                return GameAction.ACTION1
        else:
            return GameAction.ACTION1

    def _get_fallback_action(self, latest_frame: FrameData) -> GameAction:
        """Get fallback action when ReCoN ARC-2 agent fails."""
        # Simple fallback strategy
        if GameAction.ACTION1 in latest_frame.available_actions:
            return GameAction.ACTION1
        elif latest_frame.available_actions:
            return latest_frame.available_actions[0]
        else:
            return GameAction.RESET

    def get_debug_info(self) -> dict:
        """Get debug information from the agent."""
        if self.recon_arc2_agent and hasattr(self.recon_arc2_agent, 'get_debug_info'):
            return self.recon_arc2_agent.get_debug_info()
        else:
            return {'agent_type': 'ReCoN ARC-2', 'status': 'not_initialized'}

    def reset(self):
        """Reset the agent."""
        if self.recon_arc2_agent:
            self.recon_arc2_agent.reset()