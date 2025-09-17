"""
ReCoN ARC Angel Agent - Harness Adapter

Ultra-thin adapter that connects the ReCoN ARC Angel agent to the ARC-AGI-3-Agents harness.
The underlying agent already returns proper GameAction objects, so this adapter just proxies calls.
"""

import sys
import os
from typing import Any, List

# Add recon-platform to path
sys.path.insert(0, '/workspace/recon-platform')

from .agent import Agent
from .structs import FrameData, GameAction, GameState

# Lazy import to avoid import errors during agent registration
ReCoNArcAngelAgent = None


class ReCoNArcAngel(Agent):
    """
    ReCoN ARC Angel agent adapter for ARC-AGI-3-Agents harness.
    
    Ultra-thin proxy adapter - the underlying agent already returns GameAction objects,
    so we just need to proxy the calls and handle lazy initialization.
    """
    
    # Match other ReCoN agents MAX_ACTIONS (50000 instead of default 80)
    MAX_ACTIONS: int = 50000

    def __init__(self, card_id: str, game_id: str, agent_name: str, ROOT_URL: str, record: bool, *args, **kwargs):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, *args, **kwargs)
        self.recon_arc_angel_agent = None

    def _ensure_agent(self):
        """Lazy initialization of the ReCoN ARC Angel agent."""
        if self.recon_arc_angel_agent is None:
            try:
                global ReCoNArcAngelAgent
                if ReCoNArcAngelAgent is None:
                    from recon_agents.recon_arc_angel.production_agent import ProductionReCoNArcAngel as ReCoNArcAngelAgent

                self.recon_arc_angel_agent = ReCoNArcAngelAgent(self.game_id)

            except Exception as e:
                print(f"Error initializing ReCoN ARC Angel agent: {e}")
                import traceback
                traceback.print_exc()
                self.recon_arc_angel_agent = None

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """
        Check if agent is done - proxy to underlying agent.
        """
        self._ensure_agent()

        try:
            if self.recon_arc_angel_agent:
                # Proxy directly to underlying agent
                return self.recon_arc_angel_agent.is_done(frames, latest_frame)
            else:
                # Fallback to simple WIN check
                return latest_frame.state == GameState.WIN
        except Exception as e:
            print(f"Error in ReCoN ARC Angel is_done: {e}")
            import traceback
            traceback.print_exc()
            return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Choose action - proxy to underlying agent.
        
        The underlying agent already returns proper GameAction objects.
        """
        self._ensure_agent()

        # Handle special cases first
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        try:
            if self.recon_arc_angel_agent:
                # Proxy directly to underlying agent - it already returns GameAction
                return self.recon_arc_angel_agent.choose_action(frames, latest_frame)
            else:
                return self._get_fallback_action(latest_frame)

        except Exception as e:
            print(f"Error choosing action in ReCoN ARC Angel: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_action(latest_frame)

    def _get_fallback_action(self, latest_frame: FrameData) -> GameAction:
        """Get fallback action when ReCoN ARC Angel agent fails."""
        # Simple fallback strategy
        if GameAction.ACTION1 in latest_frame.available_actions:
            return GameAction.ACTION1
        elif latest_frame.available_actions:
            return latest_frame.available_actions[0]
        else:
            return GameAction.RESET

    def get_debug_info(self) -> dict:
        """Get debug information from the agent."""
        if self.recon_arc_angel_agent and hasattr(self.recon_arc_angel_agent, 'get_stats'):
            return {
                'agent_type': 'ReCoN ARC Angel',
                'status': 'initialized',
                'stats': self.recon_arc_angel_agent.get_stats()
            }
        else:
            return {'agent_type': 'ReCoN ARC Angel', 'status': 'not_initialized'}

    def reset(self):
        """Reset the agent."""
        if self.recon_arc_angel_agent and hasattr(self.recon_arc_angel_agent, 'reset'):
            self.recon_arc_angel_agent.reset()
