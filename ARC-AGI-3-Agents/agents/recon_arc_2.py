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
        Check if agent is done.

        Process frame and check WIN condition.
        """
        self._ensure_agent()

        try:
            if self.recon_arc2_agent:
                return self.recon_arc2_agent.is_done(frames, latest_frame)
            else:
                return latest_frame.state == GameState.WIN
        except Exception as e:
            print(f"Error in ReCoN ARC-2 is_done: {e}")
            import traceback
            traceback.print_exc()
            return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Choose action using active perception.

        Uses hypothesis testing to select productive actions.
        """
        self._ensure_agent()

        # Handle special cases
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        try:
            if self.recon_arc2_agent:
                # Use active perception to choose action (with coordinates if ACTION6)
                action_idx, coords = self.recon_arc2_agent.choose_action_with_coordinates(frames, latest_frame)

                # Convert to GameAction enum
                action = self._convert_action_idx(action_idx)

                # Attach coordinates for ACTION6 if provided
                if action == GameAction.ACTION6 and coords is not None:
                    x, y = coords
                    action.set_data({"x": int(x), "y": int(y)})

                # Env-gated debug hook
                try:
                    if os.getenv('RECON_ARC2_DEBUG'):
                        avail_len = len(getattr(latest_frame, 'available_actions', []) or [])
                        print(f"recon_arc_2(adapter): action={action.name}, coords={getattr(action, 'action_data', None)}, available={avail_len}")
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

    def _convert_action_idx(self, action_idx: int) -> GameAction:
        """Convert action index to GameAction enum."""
        action_map = {
            0: GameAction.ACTION1,
            1: GameAction.ACTION2,
            2: GameAction.ACTION3,
            3: GameAction.ACTION4,
            4: GameAction.ACTION5,
            5: GameAction.ACTION6
        }
        return action_map.get(action_idx, GameAction.ACTION1)

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