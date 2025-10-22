"""
FastAPI Application for ReCon ARC Demo

REST API for creating and executing ReCoN networks.
Foundation for Phase 3 implementation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uuid
import os
import json
from pathlib import Path
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from recon_engine import ReCoNGraph, ReCoNNode, ReCoNState
from recon_engine.compact import CompactReCoNGraph
from schemas import (
    NodeCreateRequest, NodeResponse, LinkCreateRequest, LinkResponse,
    NetworkCreateRequest, NetworkResponse, ExecuteRequest, ExecuteResponse,
    StateSnapshot, VisualizationResponse, ErrorResponse, ExecutionHistoryResponse,
    ExecutionStep
)


class TerminalConfig(BaseModel):
    """Configuration for a terminal node."""
    node_id: str = Field(..., description="Terminal node ID")
    measurement_value: float = Field(0.5, description="Measurement value (0-1)")

class DirectExecuteRequest(BaseModel):
    """Request to execute a network directly from data."""
    network_data: dict = Field(..., description="Network data to execute")
    root_node: str = Field(..., description="Root node to execute")
    max_steps: int = Field(100, description="Maximum execution steps")
    terminal_configs: Optional[List[TerminalConfig]] = Field([], description="Terminal configurations")


# Global storage for networks (replace with database in production)
networks: Dict[str, ReCoNGraph] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("ReCon ARC Demo API starting up...")
    
    # Create demo network - corrected structure
    demo_id = "demo"
    demo_graph = ReCoNGraph()
    demo_graph.add_node("Root", "script")
    demo_graph.add_node("A", "script")
    demo_graph.add_node("B", "script")
    demo_graph.add_node("TA", "terminal")  # Terminal for A
    demo_graph.add_node("TB", "terminal")  # Terminal for B

    # Terminals start with default behavior (auto-confirm), can be overridden via API
    # Users can manually control via /networks/{id}/nodes/{node_id}/confirm endpoint

    # Proper sequence structure: Root requests both A and B, A inhibits B until complete
    demo_graph.add_link("Root", "A", "sub")    # Root requests A
    demo_graph.add_link("Root", "B", "sub")    # Root requests B
    demo_graph.add_link("A", "B", "por")       # A inhibits B until A completes
    demo_graph.add_link("A", "TA", "sub")      # A validates via TA
    demo_graph.add_link("B", "TB", "sub")      # B validates via TB
    
    networks[demo_id] = demo_graph
    print(f"Created demo network with ID: {demo_id}")
    
    yield
    
    # Shutdown
    print("ReCon ARC Demo API shutting down...")
    networks.clear()


app = FastAPI(
    title="ReCon ARC Demo API",
    description="REST API for Request Confirmation Networks - ARC Demo",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8888",  # Local development
        "http://localhost:3000",  # Alternative local dev
        "https://melodic-chaja-8295bd.netlify.app",  # Netlify production
        "*"  # Fallback for other origins
    ],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "ReCon ARC Demo API", "version": "0.1.0", "networks": len(networks)}


@app.post("/networks", response_model=NetworkResponse)
async def create_network(request: NetworkCreateRequest):
    """Create a new ReCoN network."""
    network_id = request.network_id or str(uuid.uuid4())
    
    if network_id in networks:
        raise HTTPException(status_code=400, detail=f"Network {network_id} already exists")
    
    networks[network_id] = ReCoNGraph()
    
    return NetworkResponse(
        network_id=network_id,
        nodes=[],
        links=[], 
        step_count=0
    )


@app.get("/networks", response_model=List[str])
async def list_networks():
    """List all network IDs."""
    return list(networks.keys())


@app.get("/networks/{network_id}", response_model=NetworkResponse)
async def get_network(network_id: str):
    """Get network information."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    graph = networks[network_id]
    
    nodes = [
        NodeResponse(
            node_id=node.id,
            node_type=node.type,
            state=node.state.value,
            activation=node.activation.tolist() if hasattr(node.activation, 'tolist') else node.activation
        )
        for node in graph.nodes.values()
    ]
    
    links = [
        LinkResponse(
            source=link.source,
            target=link.target,
            link_type=link.type,
            weight=link.weight.tolist() if hasattr(link.weight, 'tolist') else link.weight
        )
        for link in graph.links
    ]
    
    return NetworkResponse(
        network_id=network_id,
        nodes=nodes,
        links=links,
        step_count=graph.step_count
    )


@app.delete("/networks/{network_id}")
async def delete_network(network_id: str):
    """Delete a network."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    del networks[network_id]
    return {"message": f"Network {network_id} deleted"}


@app.post("/networks/{network_id}/nodes", response_model=NodeResponse)
async def create_node(network_id: str, request: NodeCreateRequest):
    """Add a node to the network."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    graph = networks[network_id]
    
    try:
        node = graph.add_node(request.node_id, request.node_type.value)
        
        return NodeResponse(
            node_id=node.id,
            node_type=node.type,
            state=node.state.value,
            activation=node.activation.tolist() if hasattr(node.activation, 'tolist') else node.activation
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/networks/{network_id}/links", response_model=LinkResponse) 
async def create_link(network_id: str, request: LinkCreateRequest):
    """Add a link to the network."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    graph = networks[network_id]
    
    try:
        link = graph.add_link(request.source, request.target, request.link_type.value, request.weight)
        
        return LinkResponse(
            source=link.source,
            target=link.target,
            link_type=link.type,
            weight=link.weight.tolist() if hasattr(link.weight, 'tolist') else link.weight
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/networks/{network_id}/execute", response_model=ExecuteResponse)
async def execute_script(network_id: str, request: ExecuteRequest):
    """Execute a script in the network."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    graph = networks[network_id]

    try:
        # Reset network before execution
        graph.reset()

        # Execute script
        result = graph.execute_script(request.root_node, request.max_steps)

        # Get final states
        final_states = {node_id: node.state.value for node_id, node in graph.nodes.items()}

        return ExecuteResponse(
            network_id=network_id,
            root_node=request.root_node,
            result=result,
            final_states=final_states,
            steps_taken=graph.step_count
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/execute-network", response_model=ExecutionHistoryResponse)
async def execute_network_direct(request: DirectExecuteRequest):
    """Execute a network directly from provided data (no storage)."""
    try:
        # Create graph from provided data
        graph = ReCoNGraph.from_dict(request.network_data)
        
        # Configure terminals with user-defined measurement values
        print(f"Configuring {len(request.terminal_configs or [])} terminals")
        for terminal_config in request.terminal_configs or []:
            print(f"Configuring terminal {terminal_config.node_id}: measurement={terminal_config.measurement_value}")
            if terminal_config.node_id in graph.nodes:
                terminal_node = graph.get_node(terminal_config.node_id)
                if terminal_node.type == "terminal":
                    # Set measurement function to return the user-specified value
                    measurement_val = float(terminal_config.measurement_value)
                    terminal_node.measurement_fn = lambda env, val=measurement_val: val
                    print(f"Set {terminal_config.node_id} measurement to {measurement_val}")
                    
                    # Test the measurement function
                    test_measurement = terminal_node.measure()
                    print(f"{terminal_config.node_id} measurement test: {test_measurement}")
        
        # Reset and execute
        graph.reset()
        execution_result = graph.execute_script_with_history(request.root_node, request.max_steps)

        # Convert steps to schema format
        steps = [
            ExecutionStep(
                step=step_data["step"],
                states=step_data["states"],
                messages=step_data["messages"]
            )
            for step_data in execution_result["steps"]
        ]

        return ExecutionHistoryResponse(
            network_id="direct",
            root_node=request.root_node,
            result=execution_result["result"],
            steps=steps,
            final_state=execution_result["final_state"],
            total_steps=execution_result["total_steps"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Execution failed: {str(e)}")


@app.post("/networks/{network_id}/execute-with-history", response_model=ExecutionHistoryResponse)
async def execute_script_with_history(network_id: str, request: ExecuteRequest):
    """Execute a script in the network and return full execution history."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    graph = networks[network_id]

    try:
        # Reset network before execution
        graph.reset()

        # Execute script with history
        execution_result = graph.execute_script_with_history(request.root_node, request.max_steps)

        # Convert steps to schema format
        steps = [
            ExecutionStep(
                step=step_data["step"],
                states=step_data["states"],
                messages=step_data["messages"]
            )
            for step_data in execution_result["steps"]
        ]

        return ExecutionHistoryResponse(
            network_id=network_id,
            root_node=request.root_node,
            result=execution_result["result"],
            steps=steps,
            final_state=execution_result["final_state"],
            total_steps=execution_result["total_steps"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/networks/{network_id}/visualize", response_model=VisualizationResponse)
async def get_visualization_data(network_id: str):
    """Get current network state for visualization.""" 
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    graph = networks[network_id]
    snapshot_data = graph.get_state_snapshot()
    
    snapshot = StateSnapshot(
        nodes=snapshot_data["nodes"],
        step=snapshot_data["step"],
        requested_roots=snapshot_data["requested_roots"],
        messages=snapshot_data["messages"]
    )
    
    return VisualizationResponse(
        network_id=network_id,
        snapshot=snapshot
    )


@app.post("/networks/{network_id}/step")
async def propagate_step(network_id: str):
    """Perform one propagation step."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    graph = networks[network_id]
    graph.propagate_step()
    
    return {"step": graph.step_count, "message": "Propagation step completed"}


@app.post("/networks/{network_id}/request/{node_id}")
async def request_node(network_id: str, node_id: str):
    """Request a node (start script execution)."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    graph = networks[network_id]
    
    try:
        graph.request_root(node_id)
        return {"message": f"Requested node {node_id}", "requested_roots": list(graph.requested_roots)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/networks/{network_id}/reset")
async def reset_network(network_id: str):
    """Reset network to initial state."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    graph = networks[network_id]
    graph.reset()

    return {"message": "Network reset", "step": graph.step_count}


@app.post("/networks/{network_id}/nodes/{node_id}/confirm")
async def confirm_node(network_id: str, node_id: str):
    """Manually confirm a terminal node."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    graph = networks[network_id]

    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    node = graph.get_node(node_id)
    if node.type != "terminal":
        raise HTTPException(status_code=400, detail=f"Node {node_id} is not a terminal")

    # Set terminal to return successful measurement
    node.measurement_fn = lambda env: 1.0  # Above threshold, will confirm

    return {"message": f"Terminal {node_id} will confirm on next propagation step"}


@app.post("/networks/{network_id}/nodes/{node_id}/fail")
async def fail_node(network_id: str, node_id: str):
    """Manually fail a terminal node."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    graph = networks[network_id]

    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    node = graph.get_node(node_id)
    if node.type != "terminal":
        raise HTTPException(status_code=400, detail=f"Node {node_id} is not a terminal")

    # Set terminal to return failed measurement
    node.measurement_fn = lambda env: 0.0  # Below threshold, will fail

    return {"message": f"Terminal {node_id} will fail on next propagation step"}


# Export network data endpoints for future phases
@app.get("/networks/{network_id}/export") 
async def export_network(network_id: str, format: str = "json"):
    """Export network in specified format."""
    if network_id not in networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")
    
    graph = networks[network_id]
    
    if format == "json":
        return graph.to_dict()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@app.post("/networks/import")
async def import_network(data: dict, network_id: Optional[str] = None):
    """Import network from data."""
    try:
        graph = ReCoNGraph.from_dict(data)
        
        final_network_id = network_id or str(uuid.uuid4())
        networks[final_network_id] = graph
        
        return {"network_id": final_network_id, "message": "Network imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")


# Parsed Networks endpoints - serve static parsed network files
@app.get("/parsed-networks")
async def list_parsed_networks():
    """List available parsed networks from static files."""
    parsed_networks_dir = Path("ui/parsed_networks")
    
    if not parsed_networks_dir.exists():
        return []
    
    networks = []
    for json_file in parsed_networks_dir.glob("*.json"):
        network_id = json_file.stem
        networks.append({
            "id": network_id,
            "name": network_id.replace("_", " ").title(),
            "filename": json_file.name
        })
    
    return networks


@app.get("/parsed-networks/{network_id}")
async def get_parsed_network(network_id: str):
    """Get a specific parsed network."""
    parsed_networks_dir = Path("ui/parsed_networks")
    network_file = parsed_networks_dir / f"{network_id}.json"
    
    if not network_file.exists():
        raise HTTPException(status_code=404, detail=f"Parsed network {network_id} not found")
    
    try:
        with open(network_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load network: {str(e)}")


def parse_execution_history(log_data: dict, network_id: str):
    """Parse execution history from different log formats."""
    # ARCON format: "history": { "steps": [...] }
    if "history" in log_data:
        history = log_data["history"]
        steps = history.get("steps", [])
        return {
            "network_id": network_id,
            "root_node": "score_increase_hypothesis",  # ARCON uses this root
            "result": history.get("result", "unknown"),
            "steps": steps,
            "final_state": steps[-1]["states"] if steps else {},
            "total_steps": len(steps)
        }
    
    # ReCoN ARC Angel format: "recon_steps": [...]
    elif "recon_steps" in log_data:
        recon_steps = log_data["recon_steps"]
        
        # Convert ReCoN format to unified format
        unified_steps = []
        for i, recon_step in enumerate(recon_steps):
            # Extract node states from the detailed node structure
            states = {}
            nodes = recon_step.get("nodes", {})
            for node_id, node_data in nodes.items():
                states[node_id] = node_data.get("state", "inactive")
            
            # Create unified step format
            unified_step = {
                "step": i,
                "states": states,
                "messages": []  # ReCoN ARC Angel format doesn't have messages in same structure
            }
            unified_steps.append(unified_step)
        
        return {
            "network_id": network_id,
            "root_node": "frame_change_hypothesis",  # ReCoN ARC Angel uses this root
            "result": "unknown",  # Would need to determine from final states
            "steps": unified_steps,
            "final_state": unified_steps[-1]["states"] if unified_steps else {},
            "total_steps": len(unified_steps)
        }
    
    # Fallback for unknown format
    return {
        "network_id": network_id,
        "root_node": "unknown",
        "result": "unknown", 
        "steps": [],
        "final_state": {},
        "total_steps": 0
    }


@app.get("/parsed-networks/{network_id}/execution-history")
async def get_parsed_network_execution_history(network_id: str):
    """Get execution history for a parsed network from actual logs."""
    try:
        # Map network IDs to their log directories
        log_mapping = {
            "arcon_as66": "arcon_20250917T192437Z/game_as66-821a4dcad9c2/level_0",
            "recon_arc_angel_vc33": "recon_arc_angel_20250917T193110Z/game_vc33-6ae7bf49eea5/level_0"
        }
        
        if network_id not in log_mapping:
            # Return empty history for unknown networks
            return parse_execution_history({}, network_id)
        
        log_dir = Path("recon_log") / log_mapping[network_id]
        
        if not log_dir.exists():
            raise HTTPException(status_code=404, detail=f"Log directory not found for {network_id}")
        
        # Find the first execution log file (they contain full execution history)
        json_files = sorted(list(log_dir.glob("*.json")))
        if not json_files:
            raise HTTPException(status_code=404, detail=f"No execution logs found for {network_id}")
        
        # Load the first log file (contains complete execution)
        log_file = json_files[0]
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        return parse_execution_history(log_data, network_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load execution history: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)