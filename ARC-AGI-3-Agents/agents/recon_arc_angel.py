"""
ReCoN ARC Angel Agent - Harness Adapter

Ultra-thin adapter that connects the ReCoN ARC Angel agent to the ARC-AGI-3-Agents harness.
The underlying agent already returns proper GameAction objects, so this adapter just proxies calls.

EXTENSIONS ENABLED:
- Extension 1A: Compact ReCoN with gen loops
- Extension 1B: Link weight learning from action outcomes
- Extension 1C: Discrete timing mode
- Extension 3B: Global Workspace Theory broadcasting

REPRODUCIBILITY: For deterministic results, run with:
    export PATH="$HOME/.local/bin:$PATH" && PYTHONHASHSEED=0 uv run python main.py -a reconarcangel

Change seed in _ensure_agent() (line ~76): seed = 42 (or any integer)
"""

import sys
import os
from typing import Any, List

# Add recon-platform to path (repo-update directory)
sys.path.insert(0, '/workspace/repo-update/recon-platform')

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
    MAX_ACTIONS: int = 100

    def __init__(self, card_id: str, game_id: str, agent_name: str, ROOT_URL: str, record: bool, *args, **kwargs):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, *args, **kwargs)
        self.recon_arc_angel_agent = None
        self.seed = None  # For reproducibility tracking

    def _set_random_seeds(self, seed):
        """Set random seeds for reproducibility.

        IMPORTANT: For full reproducibility, also set PYTHONHASHSEED=0 before running:
            PYTHONHASHSEED=0 uv run python main.py -a reconarcangel
        """
        if seed is None:
            return  # Non-deterministic mode (fastest)

        import os
        import random
        import numpy as np
        import torch

        # Check PYTHONHASHSEED (critical for reproducibility)
        pythonhashseed = os.environ.get('PYTHONHASHSEED')
        if pythonhashseed != '0':
            print(f"⚠️  WARNING: PYTHONHASHSEED not set to 0 (currently: {pythonhashseed})")
            print(f"   For fully deterministic results, run with:")
            print(f"   PYTHONHASHSEED=0 uv run python main.py -a reconarcangel")
            print(f"   (seed={seed} is set, but hash randomization may still cause variation)\n")

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # CUDA deterministic operations (if using GPU)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _ensure_agent(self):
        """Lazy initialization of the ReCoN ARC Angel agent."""
        if self.recon_arc_angel_agent is None:
            try:
                # Set seed for reproducibility (BEFORE creating neural networks)
                seed = 150  # Change this: 42=default, None=non-deterministic, any int=custom
                self._set_random_seeds(seed)
                self.seed = seed

                global ReCoNArcAngelAgent
                if ReCoNArcAngelAgent is None:
                    # Force fresh import by removing cached modules (enables live modifications)
                    import sys

                    # Delete cached modules to force Python to reimport from source
                    modules_to_clear = [
                        'recon_agents.recon_arc_angel.improved_production_agent',
                        'recon_agents.recon_arc_angel.improved_hierarchy_manager',
                        'recon_agents.recon_arc_angel.link_weight_learner',
                        'recon_agents.recon_arc_angel.global_workspace',  # Extension 3B
                    ]
                    for mod_name in modules_to_clear:
                        if mod_name in sys.modules:
                            del sys.modules[mod_name]

                    # Now import fresh from source (not from cache)
                    from recon_agents.recon_arc_angel.improved_production_agent import ImprovedProductionReCoNArcAngel as ReCoNArcAngelAgent

                # Basic version (baseline)
                # self.recon_arc_angel_agent = ReCoNArcAngelAgent(self.game_id)

                # Extended version with Extensions 1A, 1B, 1C, 3B
                self.recon_arc_angel_agent = ReCoNArcAngelAgent(
                    self.game_id,
                    use_compact=True,              # Extension 1A: Compact ReCoN with gen loops (Section 3.2 style)
                    timing_mode="discrete",        # Extension 1C: "discrete", "activation", or "hybrid"
                    enable_link_learning=False,     # Extension 1B: Learn link weights from action outcomes
                    enable_global_workspace=False,  # Extension 3B: Global Workspace Theory broadcasting
                    cnn_threshold=0.1,             # CNN confidence threshold
                    max_objects=50                 # Max objects to track (BlindSquirrel limit)
                )

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
                'seed': self.seed,  # Show seed for reproducibility tracking
                'stats': self.recon_arc_angel_agent.get_stats()
            }
        else:
            return {
                'agent_type': 'ReCoN ARC Angel',
                'status': 'not_initialized',
                'seed': self.seed
            }

    def reset(self):
        """Reset the agent."""
        if self.recon_arc_angel_agent and hasattr(self.recon_arc_angel_agent, 'reset'):
            self.recon_arc_angel_agent.reset()
