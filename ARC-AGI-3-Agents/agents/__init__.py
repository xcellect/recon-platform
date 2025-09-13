from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.langgraph_thinking import LangGraphThinking
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
from .blindsquirrel_recon import BlindSquirrelReCoN
from .recon_arc_1 import ReCoNArc1
from .recon_arc_2 import ReCoNArc2

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent
AVAILABLE_AGENTS["recon_arc_1"] = ReCoNArc1
AVAILABLE_AGENTS["recon_arc_2"] = ReCoNArc2

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "BlindSquirrelReCoN",
    "ReCoNArc1",
    "ReCoNArc2",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
