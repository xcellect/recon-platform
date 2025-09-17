// Main React Flow canvas component for network visualization

import React, { useCallback, useEffect, useMemo, useRef } from 'react';
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
        return { ...baseStyle, stroke: '#dc2626', strokeWidth: 2 };
      case 'sur':
        return { ...baseStyle, stroke: '#dc2626', strokeDasharray: '5,5', strokeWidth: 2 };
      case 'por':
        return { ...baseStyle, stroke: '#dc2626', strokeWidth: 2 };
      case 'ret':
        return { ...baseStyle, stroke: '#dc2626', strokeDasharray: '5,5', strokeWidth: 2 };
      case 'gen':
        return { ...baseStyle, stroke: '#ef4444', strokeDasharray: '2,2', strokeWidth: 2 };
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
      {/* Link type label with background rect */}
      <rect 
        x={sourceX + (targetX - sourceX) / 2 - 15}
        y={sourceY + (targetY - sourceY) / 2 - 8}
        width="30"
        height="16"
        rx="4"
        fill="#1f2937"
        stroke="#374151"
        strokeWidth="1"
      />
      <text
        x={sourceX + (targetX - sourceX) / 2}
        y={sourceY + (targetY - sourceY) / 2 + 3}
        textAnchor="middle"
        style={{ 
          fontSize: '10px', 
          fill: '#e5e7eb',
          fontWeight: '500',
          fontFamily: 'system-ui, -apple-system, sans-serif'
        }}
      >
        {data?.linkType || ''}
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
  
  // Track if layout has been applied to prevent multiple layout passes
  const layoutAppliedRef = useRef<string | null>(null);
  const isLayoutingRef = useRef(false);

  // Memoize React Flow nodes (separated from edges to reduce re-renders)
  const reactFlowNodes = useMemo(() => {
    if (!currentNetwork) return [];

    return currentNetwork.nodes.map((node: ReCoNNode) => ({
      id: node.id,
      type: node.type,
      position: node.position,
      data: {
        label: node.id,
        nodeData: node,
      },
      selected: selectedNode?.id === node.id,
    }));
  }, [currentNetwork?.nodes, selectedNode?.id]);

  // Memoize React Flow edges (separated from nodes to reduce re-renders)
  const reactFlowEdges = useMemo(() => {
    if (!currentNetwork) return [];

    return currentNetwork.links.map((link: ReCoNLink) => ({
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
  }, [currentNetwork?.links, selectedLink?.id]);

  // Apply layout only once when network changes and nodes need positioning
  useEffect(() => {
    if (!currentNetwork || isLayoutingRef.current) return;
    
    const networkId = currentNetwork.id;
    const hasNodes = currentNetwork.nodes.length > 0;
    const hasLinks = currentNetwork.links.length > 0;
    
    // Check if layout has already been applied for this network
    if (layoutAppliedRef.current === networkId) return;
    
    // Check if nodes need positioning (all at origin)
    const needsLayout = hasNodes && currentNetwork.nodes.every(node => 
      node.position.x === 0 && node.position.y === 0
    );
    
    if (needsLayout && hasLinks) {
      isLayoutingRef.current = true;
      
      try {
        const layoutedNodes = autoLayout(currentNetwork.nodes, currentNetwork.links);
        
        // Update positions in batches to avoid multiple re-renders
        layoutedNodes.forEach(node => {
          updateNode(node.id, { position: node.position });
        });
        
        layoutAppliedRef.current = networkId;
      } finally {
        isLayoutingRef.current = false;
      }
    } else if (hasNodes && !hasLinks) {
      // For networks without links, just mark as layout applied
      layoutAppliedRef.current = networkId;
    }
  }, [currentNetwork?.id, currentNetwork?.nodes.length, currentNetwork?.links.length, updateNode]);

  // Update React Flow nodes when network nodes change
  useEffect(() => {
    setNodes(reactFlowNodes);
  }, [reactFlowNodes, setNodes]);

  // Update React Flow edges when network links change
  useEffect(() => {
    setEdges(reactFlowEdges);
  }, [reactFlowEdges, setEdges]);

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
    <div className="h-full w-full bg-gray-900">
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
              case 'script': return '#dc2626';
              case 'terminal': return '#991b1b';
              case 'hybrid': return '#ef4444';
              default: return '#374151';
            }
          }}
          maskColor="rgba(0, 0, 0, 0.6)"
        />
        <Background variant="dots" gap={12} size={1} color="#6b7280" />
      </ReactFlow>
    </div>
  );
}