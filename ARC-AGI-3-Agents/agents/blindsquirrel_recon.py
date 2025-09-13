"""
BlindSquirrel ReCoN Agent - Harness Adapter

Adapter that connects the BlindSquirrel ReCoN agent to the ARC-AGI-3-Agents harness.
"""

import sys
import os
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

    Bridges the BlindSquirrel ReCoN implementation to the competition harness.
    """

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

    def process_latest_frame(self, latest_frame: FrameData):
        """Process the latest frame using BlindSquirrel logic."""
        self._ensure_agent()

        try:
            if self.blindsquirrel_agent:
                self.blindsquirrel_agent.process_latest_frame(latest_frame)
        except Exception as e:
            print(f"Error processing frame in BlindSquirrel ReCoN: {e}")
            # Continue with fallback behavior

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Check if the agent is done."""
        self._ensure_agent()

        # Use BlindSquirrel's is_done logic (it processes frame internally)
        try:
            if self.blindsquirrel_agent:
                return self.blindsquirrel_agent.is_done(frames, latest_frame)
            else:
                return latest_frame.state == GameState.WIN
        except Exception as e:
            print(f"Error in is_done: {e}")
            # Fallback to simple WIN check
            return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose an action based on the current state."""
        self._ensure_agent()

        # Handle special cases
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # Use BlindSquirrel ReCoN agent for action selection
        try:
            if self.blindsquirrel_agent:
                # Use choose_action method to separate from frame processing in is_done
                action_data = self.blindsquirrel_agent._choose_action(latest_frame)

                # Convert action data to GameAction
                if action_data is not None:
                    return self._convert_action_data(action_data)
                else:
                    return self._get_fallback_action(latest_frame)
            else:
                return self._get_fallback_action(latest_frame)

        except Exception as e:
            print(f"Error choosing action in BlindSquirrel ReCoN: {e}")
            return self._get_fallback_action(latest_frame)

    def _convert_action_data(self, action_data: Any) -> GameAction:
        """Convert BlindSquirrel action data to GameAction enum."""
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
                # For click actions, return ACTION6 (click action)
                return GameAction.ACTION6
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