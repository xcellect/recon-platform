"""
ReCoN Graph Implementation

ReCoN network with nodes and typed links.
Handles message propagation and script execution.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import networkx as nx
import torch
from .node import ReCoNNode, ReCoNState
from .messages import ReCoNMessage, MessageType
from .hybrid_node import NodeMode


class ReCoNLink:
    """A typed link between ReCoN nodes."""
    
    def __init__(self, source: str, target: str, link_type: str, weight: Union[float, torch.Tensor] = 1.0):
        self.source = source
        self.target = target
        self.type = link_type  # "por", "ret", "sub", "sur"
        self.weight = weight
        
        # Validate link type
        valid_types = {"por", "ret", "sub", "sur", "gen"}
        if link_type not in valid_types:
            raise ValueError(f"Invalid link type: {link_type}. Must be one of {valid_types}")
    
    def __repr__(self):
        return f"ReCoNLink({self.source}->{self.target} via {self.type}, w={self.weight})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight.tolist() if isinstance(self.weight, torch.Tensor) else self.weight
        }
    
    @classmethod  
    def from_dict(cls, data: Dict[str, Any]) -> 'ReCoNLink':
        """Deserialize from dictionary."""
        weight = data["weight"]
        if isinstance(weight, list):
            weight = torch.tensor(weight)
        return cls(data["source"], data["target"], data["type"], weight)


class ReCoNGraph:
    """
    ReCoN network graph with enhanced export capabilities.
    
    Handles message propagation, script execution, and provides
    comprehensive export functionality for visualization.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ReCoNNode] = {}
        self.links: List[ReCoNLink] = []
        
        # Execution state
        self.requested_roots: Set[str] = set()
        self.message_queue: List[ReCoNMessage] = []
        self.step_count = 0
        
    def add_node(self, node: Union[str, ReCoNNode], node_type: str = "script") -> ReCoNNode:
        """Add a node to the network."""
        if isinstance(node, str):
            # Create new node from ID
            node_id = node
            if node_id in self.nodes:
                raise ValueError(f"Node {node_id} already exists")
            node = ReCoNNode(node_id, node_type)
        else:
            # Use existing node object
            node_id = node.id
            if node_id in self.nodes:
                raise ValueError(f"Node {node_id} already exists")
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, node_obj=node)
        
        # Initially set link flags to False
        node._has_children = False
        node._has_por_successors = False
        node._has_sequence_children = False
        
        return node
    
    def add_link(self, source: str, target: str, link_type: str, weight: Union[float, torch.Tensor] = 1.0) -> ReCoNLink:
        """
        Add a typed link between nodes.
        
        Automatically creates reciprocal links for por/ret and sub/sur pairs.
        """
        if source not in self.nodes:
            raise ValueError(f"Source node {source} does not exist")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} does not exist")
            
        # Check constraints from paper
        source_node = self.nodes[source]
        target_node = self.nodes[target]
        
        # Terminal nodes can only be targeted by sub, source of sur
        if target_node.type == "terminal" and link_type not in ["sub"]:
            raise ValueError(f"Terminal nodes can only be targeted by 'sub' links")
        if source_node.type == "terminal" and link_type not in ["sur"]:
            raise ValueError(f"Terminal nodes can only source 'sur' links")
        
        # Each pair of nodes can have exactly one pair of por/ret or sub/sur links
        existing_links = [(l.source, l.target, l.type) for l in self.links]
        
        if link_type in ["por", "ret"]:
            # Check for existing por/ret pair
            if any((source, target, "por") == (s, t, lt) or (target, source, "ret") == (s, t, lt) 
                   for s, t, lt in existing_links):
                raise ValueError(f"por/ret link pair already exists between {source} and {target}")
            # Enforce paper rule: do not allow both por/ret and sub/sur between same node pair
            if any(((source, target, "sub") == (s, t, lt)) or ((target, source, "sur") == (s, t, lt)) or
                   ((target, source, "sub") == (s, t, lt)) or ((source, target, "sur") == (s, t, lt))
                   for s, t, lt in existing_links):
                raise ValueError(
                    f"Cannot add {link_type} between {source} and {target}: sub/sur pair already exists for this node pair"
                )
                
        if link_type in ["sub", "sur"]:
            # Check for existing sub/sur pair  
            if any((source, target, "sub") == (s, t, lt) or (target, source, "sur") == (s, t, lt)
                   for s, t, lt in existing_links):
                raise ValueError(f"sub/sur link pair already exists between {source} and {target}")
            # Enforce paper rule: do not allow both sub/sur and por/ret between same node pair
            if any(((source, target, "por") == (s, t, lt)) or ((target, source, "ret") == (s, t, lt)) or
                   ((target, source, "por") == (s, t, lt)) or ((source, target, "ret") == (s, t, lt))
                   for s, t, lt in existing_links):
                raise ValueError(
                    f"Cannot add {link_type} between {source} and {target}: por/ret pair already exists for this node pair"
                )
        
        # Create primary link
        link = ReCoNLink(source, target, link_type, weight)
        self.links.append(link)
        self.graph.add_edge(source, target, link_obj=link, link_type=link_type)
        
        # Update link flags
        if link_type == "sub":
            self.nodes[source]._has_children = True
            # Check if the child has por links (is part of sequence)
            if hasattr(self.nodes[target], '_has_por_successors') and self.nodes[target]._has_por_successors:
                self.nodes[source]._has_sequence_children = True
        elif link_type == "por":
            self.nodes[source]._has_por_successors = True
            # Update parent's sequence children flag
            for parent_link in self.get_links(target=source, link_type="sub"):
                self.nodes[parent_link.source]._has_sequence_children = True
        
        # Create reciprocal link automatically
        reciprocal_type = None
        if link_type == "por":
            reciprocal_type = "ret"
        elif link_type == "ret":
            reciprocal_type = "por" 
        elif link_type == "sub":
            reciprocal_type = "sur"
        elif link_type == "sur":
            reciprocal_type = "sub"
            
        if reciprocal_type:
            reciprocal_link = ReCoNLink(target, source, reciprocal_type, weight)
            self.links.append(reciprocal_link)
            self.graph.add_edge(target, source, link_obj=reciprocal_link, link_type=reciprocal_type)
        
        return link
    
    def has_link(self, source: str, target: str, link_type: str) -> bool:
        """Check if a specific link exists."""
        return any(l.source == source and l.target == target and l.type == link_type 
                  for l in self.links)
    
    def get_node(self, node_id: str) -> ReCoNNode:
        """Get node by ID."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        return self.nodes[node_id]
    
    def get_links(self, source: Optional[str] = None, target: Optional[str] = None, 
                  link_type: Optional[str] = None) -> List[ReCoNLink]:
        """Get links matching criteria."""
        result = []
        for link in self.links:
            if source and link.source != source:
                continue
            if target and link.target != target:
                continue  
            if link_type and link.type != link_type:
                continue
            result.append(link)
        return result
    
    def request_root(self, node_id: str):
        """Start script execution by requesting root node."""
        if node_id not in self.nodes:
            raise ValueError(f"Root node {node_id} does not exist")
            
        self.requested_roots.add(node_id)
        
        # Send initial request message and immediately process it
        root_node = self.nodes[node_id]
        message = ReCoNMessage(
            MessageType.REQUEST,
            "system",
            node_id, 
            "sub",
            1.0
        )
        root_node.add_incoming_message(message)
        
        # Immediately update the root node state from INACTIVE to REQUESTED
        if root_node.state == ReCoNState.INACTIVE:
            # Process the request to move to REQUESTED state
            inputs = {"sub": 1.0, "por": 0.0, "ret": 0.0, "sur": 0.0}
            root_node.update_state(inputs)
    
    def stop_request(self, node_id: str):
        """Stop script execution by removing request from root node."""
        if node_id in self.requested_roots:
            self.requested_roots.remove(node_id)
            
        # This will cause nodes to reset to inactive on next propagation step
    
    def propagate_step(self):
        """
        Perform one step of message propagation.
        
        Two phases: propagation (collect messages) and calculation (update states).
        """
        self.step_count += 1
        
        # Phase 1: Clear old messages and collect new ones
        for node in self.nodes.values():
            for link_type in node.incoming_messages:
                node.incoming_messages[link_type].clear()
        
        # Add root requests
        for root_id in self.requested_roots:
            root_node = self.nodes[root_id] 
            message = ReCoNMessage(MessageType.REQUEST, "system", root_id, "sub", 1.0)
            root_node.add_incoming_message(message)
        
        # Propagate messages from previous step
        for message in self.message_queue:
            if message.target in self.nodes:
                self.nodes[message.target].add_incoming_message(message)
        
        # Phase 2: Update all nodes and generate new messages
        self.message_queue.clear()
        
        for node in self.nodes.values():
            # Update node state
            outgoing_signals = node.update_state()
            
            # Convert signals to messages for next step
            for link_type, signal in outgoing_signals.items():
                if signal is None:
                    continue
                    
                # Find links of this type from this node
                outgoing_links = self.get_links(source=node.id, link_type=link_type)
                
                for link in outgoing_links:
                    message_type = MessageType(signal) if isinstance(signal, str) else MessageType.REQUEST
                    activation = link.weight if signal in ["request", "confirm"] else 0.0
                    
                    if signal == "inhibit_request":
                        message_type = MessageType.INHIBIT_REQUEST
                        activation = -1.0
                    elif signal == "inhibit_confirm":
                        message_type = MessageType.INHIBIT_CONFIRM 
                        activation = -1.0
                    elif signal == "wait":
                        message_type = MessageType.WAIT
                        activation = 0.01
                    elif signal == "confirm":
                        message_type = MessageType.CONFIRM
                        activation = 1.0
                    elif signal == "request":
                        message_type = MessageType.REQUEST
                        activation = 1.0
                        
                    message = ReCoNMessage(
                        message_type,
                        node.id,
                        link.target,
                        link_type,
                        activation
                    )
                    self.message_queue.append(message)
        
        # Terminal nodes handle measurement in their update_state method
        # No need for separate terminal handling here
        
        # Handle sequence chain propagation for backward compatibility
        # If a node becomes TRUE and has por successors but no sub children,
        # automatically request the next node in the sequence
        self._handle_sequence_chain_propagation()
    
    def _handle_sequence_chain_propagation(self):
        """
        Handle backward compatibility for sequence chains.
        
        When a node becomes TRUE and has por successors but no sub children,
        automatically request the next node in the sequence chain.
        """
        # Disabled for strict paper compliance (no automatic sequence requests)
        return
    
    def is_completed(self) -> bool:
        """Check if all requested scripts have completed (confirmed or failed)."""
        for root_id in self.requested_roots:
            root_state = self.nodes[root_id].state
            if root_state not in [ReCoNState.CONFIRMED, ReCoNState.FAILED]:
                return False
        return True
    
    def get_results(self) -> Dict[str, str]:
        """Get final results for all requested roots."""
        results = {}
        for root_id in self.requested_roots:
            results[root_id] = self.nodes[root_id].state.value
        return results
    
    def reset(self):
        """Reset all nodes and execution state."""
        for node in self.nodes.values():
            node.reset()
        self.requested_roots.clear()
        self.message_queue.clear()
        self.step_count = 0
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state of all nodes for visualization."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "step": self.step_count,
            "requested_roots": list(self.requested_roots),
            "messages": len(self.message_queue)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire graph to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "links": [link.to_dict() for link in self.links],
            "requested_roots": list(self.requested_roots),
            "step_count": self.step_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReCoNGraph':
        """Deserialize graph from dictionary.""" 
        graph = cls()
        
        # Add nodes
        for node_data in data["nodes"]:
            node = ReCoNNode.from_dict(node_data)
            graph.nodes[node.id] = node
            graph.graph.add_node(node.id, node_obj=node)
        
        # Add links (skip auto-created reciprocals)
        added_pairs = set()
        for link_data in data["links"]:
            link = ReCoNLink.from_dict(link_data)
            
            # Check if we already added this pair
            pair_key = tuple(sorted([(link.source, link.target, link.type)]))
            reciprocal_key = tuple(sorted([(link.target, link.source, 
                                          "ret" if link.type == "por" else 
                                          "por" if link.type == "ret" else
                                          "sur" if link.type == "sub" else
                                          "sub" if link.type == "sur" else link.type)]))
            
            if pair_key not in added_pairs and reciprocal_key not in added_pairs:
                # Add this link, which will auto-create reciprocal
                graph.add_link(link.source, link.target, link.type, link.weight)
                added_pairs.add(pair_key)
        
        graph.requested_roots = set(data.get("requested_roots", []))
        graph.step_count = data.get("step_count", 0)
        
        return graph
    
    def execute_script(self, root_id: str, max_steps: int = 100) -> str:
        """
        Execute a script to completion.
        Returns final state: 'confirmed', 'failed', or 'timeout'.
        """
        self.request_root(root_id)

        for step in range(max_steps):
            self.propagate_step()

            if self.is_completed():
                result = self.get_results()[root_id]
                return result

        return "timeout"

    def execute_script_with_history(self, root_id: str, max_steps: int = 100) -> Dict[str, Any]:
        """
        Execute a script to completion, capturing full execution history.
        Returns final state and all intermediate steps.
        """
        history = []

        # Initial state (step 0)
        initial_states = {node_id: node.state.value for node_id, node in self.nodes.items()}
        history.append({
            "step": 0,
            "states": initial_states,
            "messages": []
        })

        self.request_root(root_id)

        for step in range(1, max_steps + 1):
            # Capture current messages before propagation
            current_messages = []
            for message in self.message_queue:
                current_messages.append({
                    "type": message.type.value,
                    "from": message.source,
                    "to": message.target,
                    "link": message.link_type
                })

            # Add root request messages
            if root_id in self.requested_roots:
                current_messages.append({
                    "type": "request",
                    "from": "user",
                    "to": root_id,
                    "link": "sub"
                })

            self.propagate_step()

            # Capture state after propagation
            step_states = {node_id: node.state.value for node_id, node in self.nodes.items()}

            history.append({
                "step": step,
                "states": step_states,
                "messages": current_messages
            })

            if self.is_completed():
                result = self.get_results()[root_id]
                return {
                    "result": result,
                    "steps": history,
                    "final_state": result,
                    "total_steps": step
                }

        return {
            "result": "timeout",
            "steps": history,
            "final_state": "timeout",
            "total_steps": max_steps
        }
    
    def to_react_flow_format(self) -> Dict[str, Any]:
        """
        Export graph in React Flow format for visualization.
        
        Returns nodes and edges in the format expected by React Flow.
        """
        react_nodes = []
        react_edges = []
        
        # Convert nodes
        for node_id, node in self.nodes.items():
            # Determine node appearance based on type and mode
            node_type = "default"
            node_color = "#ffffff"
            
            if hasattr(node, 'mode'):
                if node.mode.value == "explicit":
                    node_color = "#e1f5fe"  # Light blue for explicit
                elif node.mode.value == "implicit":
                    node_color = "#f3e5f5"  # Light purple for implicit
                elif node.mode.value == "neural":
                    node_color = "#e8f5e8"  # Light green for neural
            
            # State-based coloring
            if hasattr(node, 'state'):
                if node.state == ReCoNState.CONFIRMED:
                    node_color = "#c8e6c9"  # Green
                elif node.state == ReCoNState.FAILED:
                    node_color = "#ffcdd2"  # Red
                elif node.state == ReCoNState.ACTIVE:
                    node_color = "#fff3e0"  # Orange
                elif node.state == ReCoNState.WAITING:
                    node_color = "#f0f4c3"  # Yellow
            
            # Extract node metadata
            node_data = {
                "id": node_id,
                "type": node.type,
                "mode": getattr(node, 'mode', NodeMode.EXPLICIT).value if hasattr(node, 'mode') else "explicit",
                "state": node.state.value if hasattr(node, 'state') else "inactive",
                "activation": float(node.activation) if isinstance(node.activation, (int, float)) else 0.0,
                "is_neural": hasattr(node, 'model') and node.model is not None,
                "is_terminal": node.type == "terminal"
            }
            
            # Add neural model info if applicable
            if hasattr(node, 'model_info'):
                node_data["model_info"] = node.model_info
            
            react_node = {
                "id": node_id,
                "type": node_type,
                "position": {"x": 0, "y": 0},  # Will be auto-layouted
                "data": {
                    "label": node_id,
                    "nodeData": node_data
                },
                "style": {
                    "backgroundColor": node_color,
                    "border": "2px solid #333",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "minWidth": "120px"
                }
            }
            
            react_nodes.append(react_node)
        
        # Convert edges (links)
        for link in self.links:
            # Edge styling based on link type
            edge_color = "#333"
            edge_style = "solid"
            edge_width = 2
            
            if link.type == "sub":
                edge_color = "#1976d2"  # Blue for sub
                edge_width = 3
            elif link.type == "sur":
                edge_color = "#1976d2"  # Blue for sur
                edge_style = "dashed"
            elif link.type == "por":
                edge_color = "#388e3c"  # Green for por
                edge_width = 3
            elif link.type == "ret":
                edge_color = "#388e3c"  # Green for ret
                edge_style = "dashed"
            elif link.type == "gen":
                edge_color = "#f57c00"  # Orange for gen
                edge_style = "dotted"
            
            react_edge = {
                "id": f"{link.source}-{link.target}-{link.type}",
                "source": link.source,
                "target": link.target,
                "type": "smoothstep",
                "animated": hasattr(link, 'active') and link.active,
                "label": link.type,
                "labelStyle": {"fontSize": "12px", "fontWeight": "bold"},
                "style": {
                    "stroke": edge_color,
                    "strokeWidth": edge_width,
                    "strokeDasharray": "5,5" if edge_style == "dashed" else "2,2" if edge_style == "dotted" else None
                },
                "data": {
                    "linkType": link.type,
                    "weight": float(link.weight) if isinstance(link.weight, (int, float)) else 1.0
                }
            }
            
            react_edges.append(react_edge)
        
        return {
            "nodes": react_nodes,
            "edges": react_edges,
            "metadata": {
                "step_count": self.step_count,
                "requested_roots": list(self.requested_roots),
                "total_nodes": len(self.nodes),
                "total_links": len(self.links),
                "node_types": {
                    "script": len([n for n in self.nodes.values() if n.type == "script"]),
                    "terminal": len([n for n in self.nodes.values() if n.type == "terminal"])
                }
            }
        }
    
    def export_for_visualization(self, format_type: str = "react_flow") -> Dict[str, Any]:
        """
        Export graph in various formats for visualization.
        
        Args:
            format_type: "react_flow", "cytoscape", "d3", or "graphviz"
        """
        if format_type == "react_flow":
            return self.to_react_flow_format()
        elif format_type == "cytoscape":
            return self.to_cytoscape_format()
        elif format_type == "d3":
            return self.to_d3_format()
        elif format_type == "graphviz":
            return self.to_graphviz_format()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def to_cytoscape_format(self) -> Dict[str, Any]:
        """Export for Cytoscape.js visualization."""
        elements = []
        
        # Add nodes
        for node_id, node in self.nodes.items():
            elements.append({
                "data": {
                    "id": node_id,
                    "label": node_id,
                    "type": node.type,
                    "state": node.state.value if hasattr(node, 'state') else "inactive",
                    "mode": getattr(node, 'mode', "explicit"),
                    "activation": float(node.activation) if isinstance(node.activation, (int, float)) else 0.0
                }
            })
        
        # Add edges
        for link in self.links:
            elements.append({
                "data": {
                    "id": f"{link.source}-{link.target}-{link.type}",
                    "source": link.source,
                    "target": link.target,
                    "label": link.type,
                    "type": link.type,
                    "weight": float(link.weight) if isinstance(link.weight, (int, float)) else 1.0
                }
            })
        
        return {"elements": elements}
    
    def to_d3_format(self) -> Dict[str, Any]:
        """Export for D3.js visualization."""
        nodes = []
        links = []
        
        # Add nodes
        for node_id, node in self.nodes.items():
            nodes.append({
                "id": node_id,
                "group": 1 if node.type == "script" else 2,
                "type": node.type,
                "state": node.state.value if hasattr(node, 'state') else "inactive",
                "mode": getattr(node, 'mode', "explicit"),
                "activation": float(node.activation) if isinstance(node.activation, (int, float)) else 0.0
            })
        
        # Add links
        for link in self.links:
            links.append({
                "source": link.source,
                "target": link.target,
                "type": link.type,
                "weight": float(link.weight) if isinstance(link.weight, (int, float)) else 1.0,
                "value": 1  # For D3 force layout
            })
        
        return {"nodes": nodes, "links": links}
    
    def to_graphviz_format(self) -> str:
        """Export as Graphviz DOT format."""
        dot_lines = ["digraph ReCoN {"]
        dot_lines.append('  node [shape=box, style=rounded];')
        
        # Add nodes
        for node_id, node in self.nodes.items():
            style = "filled"
            color = "lightblue" if node.type == "script" else "lightgreen"
            
            if hasattr(node, 'state'):
                if node.state == ReCoNState.CONFIRMED:
                    color = "lightgreen"
                elif node.state == ReCoNState.FAILED:
                    color = "lightcoral"
                elif node.state == ReCoNState.ACTIVE:
                    color = "lightyellow"
            
            label = f"{node_id}\\n{node.type}"
            if hasattr(node, 'mode'):
                label += f"\\n{node.mode.value}"
            
            dot_lines.append(f'  "{node_id}" [label="{label}", fillcolor={color}, style={style}];')
        
        # Add edges
        for link in self.links:
            color = "blue" if link.type in ["sub", "sur"] else "green"
            style = "dashed" if link.type in ["sur", "ret"] else "solid"
            
            dot_lines.append(f'  "{link.source}" -> "{link.target}" [label="{link.type}", color={color}, style={style}];')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)
    
    def auto_layout_positions(self, width: int = 800, height: int = 600) -> Dict[str, Dict[str, float]]:
        """
        Generate automatic layout positions for nodes.
        
        Uses networkx layout algorithms to position nodes.
        """
        try:
            # Try spring layout first
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        except:
            # Fallback to circular layout
            pos = nx.circular_layout(self.graph)
        
        # Scale to desired dimensions
        positions = {}
        for node_id, (x, y) in pos.items():
            positions[node_id] = {
                "x": x * width,
                "y": y * height
            }
        
        return positions
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """
        Get trace of execution steps for debugging/visualization.
        
        Returns list of step snapshots showing state changes.
        """
        # This would be populated during execution
        # For now, return current state
        return [self.get_state_snapshot()]
    
    def __repr__(self):
        return f"ReCoNGraph(nodes={len(self.nodes)}, links={len(self.links)}, step={self.step_count})"