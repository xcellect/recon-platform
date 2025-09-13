"""
API Schemas for ReCoN Platform

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class NodeType(str, Enum):
    """Node type options."""
    SCRIPT = "script"
    TERMINAL = "terminal"


class LinkType(str, Enum):
    """Link type options.""" 
    POR = "por"
    RET = "ret"
    SUB = "sub"
    SUR = "sur"
    GEN = "gen"


class NodeCreateRequest(BaseModel):
    """Request to create a new node."""
    node_id: str = Field(..., description="Unique identifier for the node")
    node_type: NodeType = Field(NodeType.SCRIPT, description="Type of node")
    

class NodeResponse(BaseModel):
    """Response containing node information."""
    node_id: str
    node_type: str
    state: str
    activation: Union[float, List[float]]


class LinkCreateRequest(BaseModel):
    """Request to create a new link."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID") 
    link_type: LinkType = Field(..., description="Type of link")
    weight: Union[float, List[float]] = Field(1.0, description="Link weight")


class LinkResponse(BaseModel):
    """Response containing link information."""
    source: str
    target: str
    link_type: str
    weight: Union[float, List[float]]


class NetworkCreateRequest(BaseModel):
    """Request to create a new network."""
    network_id: Optional[str] = Field(None, description="Optional network ID")
    

class NetworkResponse(BaseModel):
    """Response containing network information."""
    network_id: str
    nodes: List[NodeResponse]
    links: List[LinkResponse]
    step_count: int


class ExecuteRequest(BaseModel):
    """Request to execute a script.""" 
    root_node: str = Field(..., description="Root node to execute")
    max_steps: int = Field(100, description="Maximum execution steps")


class ExecuteResponse(BaseModel):
    """Response from script execution."""
    network_id: str
    root_node: str
    result: str  # "confirmed", "failed", "timeout"
    final_states: Dict[str, str]
    steps_taken: int


class StateSnapshot(BaseModel):
    """Current state of network for visualization."""
    nodes: Dict[str, Dict[str, Any]]
    step: int
    requested_roots: List[str]
    messages: int


class VisualizationResponse(BaseModel):
    """Response for visualization data."""
    network_id: str
    snapshot: StateSnapshot
    

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    details: Optional[str] = None