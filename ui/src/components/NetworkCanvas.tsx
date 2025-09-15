// Main React Flow canvas component for network visualization

import React, { useCallback, useEffect, useMemo } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  ConnectionMode,
  Controls,
  MiniMap,
  Background,
  NodeTypes,
  EdgeTypes,
} from '@xyflow/react';

import { ScriptNode, TerminalNode, HybridNode } from '../nodes';
import { useNetworkStore } from '../stores/networkStore';
import { ReCoNNode, ReCoNLink, ReCoNLinkType } from '../types/recon';
import { autoLayout } from '../utils/layout';

// Custom edge component for link types
const CustomEdge = ({ id, sourceX, sourceY, targetX, targetY, style, data }: any) => {
  const edgePath = `M${sourceX},${sourceY} L${targetX},${targetY}`;

  const getEdgeStyle = (linkType: ReCoNLinkType) => {
    const baseStyle = { strokeWidth: 2, ...style };

    switch (linkType) {
      case 'sub':
        return { ...baseStyle, stroke: '#1976d2', strokeWidth: 3 };
      case 'sur':
        return { ...baseStyle, stroke: '#1976d2', strokeDasharray: '5,5' };
      case 'por':
        return { ...baseStyle, stroke: '#388e3c', strokeWidth: 3 };
      case 'ret':
        return { ...baseStyle, stroke: '#388e3c', strokeDasharray: '5,5' };
      case 'gen':
        return { ...baseStyle, stroke: '#f57c00', strokeDasharray: '2,2' };
      default:
        return baseStyle;
    }
  };

  return (
    <g>
      <path
        id={id}
        style={getEdgeStyle(data?.linkType)}
        className="react-flow__edge-path"
        d={edgePath}
      />
      {/* Link type label */}
      <text>
        <textPath href={`#${id}`} style={{ fontSize: '12px', fill: '#666' }} startOffset="50%" textAnchor="middle">
          {data?.linkType || ''}
        </textPath>
      </text>
    </g>
  );
};

const nodeTypes: NodeTypes = {
  script: ScriptNode,
  terminal: TerminalNode,
  hybrid: HybridNode,
};

const edgeTypes: EdgeTypes = {
  custom: CustomEdge,
};

interface NetworkCanvasProps {
  onNodeSelect?: (nodeId: string | null) => void;
  onEdgeSelect?: (edgeId: string | null) => void;
}

export default function NetworkCanvas({ onNodeSelect, onEdgeSelect }: NetworkCanvasProps) {
  const {
    currentNetwork,
    selectedNode,
    selectedLink,
    selectNode,
    selectLink,
    addNode,
    addLink,
    updateNode,
    clearSelection,
  } = useNetworkStore();

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Convert ReCoN network to React Flow format
  const convertToReactFlowFormat = useCallback((network: any) => {
    if (!network) return { nodes: [], edges: [] };

    // Apply auto-layout if nodes don't have positions
    const layoutedNodes = autoLayout(network.nodes, network.links);

    const reactFlowNodes: Node[] = layoutedNodes.map((node: ReCoNNode) => ({
      id: node.id,
      type: node.type,
      position: node.position,
      data: {
        label: node.id,
        nodeData: node,
      },
      selected: selectedNode?.id === node.id,
    }));

    const reactFlowEdges: Edge[] = network.links.map((link: ReCoNLink) => ({
      id: link.id,
      source: link.source,
      target: link.target,
      type: 'custom',
      animated: false,
      data: {
        linkType: link.type,
        weight: link.weight,
      },
      selected: selectedLink?.id === link.id,
    }));

    return { nodes: reactFlowNodes, edges: reactFlowEdges };
  }, [selectedNode, selectedLink]);

  // Update React Flow when network changes
  useEffect(() => {
    if (currentNetwork) {
      const { nodes: reactFlowNodes, edges: reactFlowEdges } = convertToReactFlowFormat(currentNetwork);
      setNodes(reactFlowNodes);
      setEdges(reactFlowEdges);
    } else {
      setNodes([]);
      setEdges([]);
    }
  }, [currentNetwork, convertToReactFlowFormat, setNodes, setEdges]);

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    const reconNode = currentNetwork?.nodes.find(n => n.id === node.id);
    if (reconNode) {
      selectNode(reconNode);
      onNodeSelect?.(node.id);
    }
  }, [currentNetwork, selectNode, onNodeSelect]);

  // Handle edge selection
  const onEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    const reconLink = currentNetwork?.links.find(l => l.id === edge.id);
    if (reconLink) {
      selectLink(reconLink);
      onEdgeSelect?.(edge.id);
    }
  }, [currentNetwork, selectLink, onEdgeSelect]);

  // Handle new connections
  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target) return;

    // Default to 'sub' link type - this should be configurable in the UI
    const linkType: ReCoNLinkType = 'sub';

    addLink({
      source: connection.source,
      target: connection.target,
      type: linkType,
      weight: 1.0,
    });
  }, [addLink]);

  // Handle canvas click (clear selection)
  const onPaneClick = useCallback(() => {
    clearSelection();
    onNodeSelect?.(null);
    onEdgeSelect?.(null);
  }, [clearSelection, onNodeSelect, onEdgeSelect]);

  // Handle node position changes
  const onNodeDragStop = useCallback((event: React.MouseEvent, node: Node) => {
    updateNode(node.id, {
      position: node.position,
    });
  }, [updateNode]);

  // Handle adding new nodes via double-click
  const onPaneDoubleClick = useCallback((event: React.MouseEvent) => {
    const rect = (event.target as Element).getBoundingClientRect();
    const position = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };

    // Generate unique node ID
    const nodeId = `node_${Date.now()}`;

    addNode({
      id: nodeId,
      type: 'script', // Default type
      state: 'inactive',
      activation: 0,
      position,
    });
  }, [addNode]);

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        onPaneClick={onPaneClick}
        onNodeDragStop={onNodeDragStop}
        onPaneDoubleClick={onPaneDoubleClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionMode={ConnectionMode.Loose}
        fitView
        fitViewOptions={{ padding: 0.2 }}
      >
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            switch (node.type) {
              case 'script': return '#e3f2fd';
              case 'terminal': return '#e8f5e8';
              case 'hybrid': return '#f3e5f5';
              default: return '#f5f5f5';
            }
          }}
        />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}