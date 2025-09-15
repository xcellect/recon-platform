// Simplified Network Canvas that actually works
import React, { useCallback, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  Controls,
  MiniMap,
  Background,
} from 'reactflow';
import { useNetworkStore } from '../stores/networkStore';

interface SimpleNetworkCanvasProps {
  onNodeSelect?: (nodeId: string | null) => void;
  onEdgeSelect?: (edgeId: string | null) => void;
  executionHistory?: any[];
  currentStep?: number;
}

export default function SimpleNetworkCanvas({
  onNodeSelect,
  onEdgeSelect,
  executionHistory = [],
  currentStep = 0
}: SimpleNetworkCanvasProps) {
  const { currentNetwork, addLink } = useNetworkStore();

  // Get current step data for state override
  const currentStepData = executionHistory[currentStep];

  // Convert ReCoN nodes to React Flow nodes
  const reactFlowNodes: Node[] = currentNetwork?.nodes.map(node => {
    // Use execution history state if available, otherwise use base state
    const currentState = currentStepData?.states?.[node.id] || node.state;

    return {
      id: node.id,
      type: 'default',
      position: node.position,
      data: {
        label: `${node.id}\n(${node.type})\n${currentState}`,
        state: currentState
      },
      style: {
        backgroundColor: getNodeColor(node.type, currentState),
        border: `2px solid ${getBorderColor(node.type)}`,
        borderRadius: '8px',
        padding: '10px',
        minWidth: '100px',
        textAlign: 'center'
      }
    };
  }) || [];

  // Convert ReCoN links to React Flow edges
  const reactFlowEdges: Edge[] = currentNetwork?.links.map(link => ({
    id: link.id,
    source: link.source,
    target: link.target,
    label: link.type,
    style: {
      stroke: getLinkColor(link.type),
      strokeWidth: 2,
      strokeDasharray: link.type === 'por' || link.type === 'ret' ? '5,5' : undefined
    }
  })) || [];

  const [nodes, setNodes, onNodesChange] = useNodesState(reactFlowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(reactFlowEdges);

  // Update React Flow nodes when store changes or execution state changes
  useEffect(() => {
    setNodes(reactFlowNodes);
  }, [currentNetwork?.nodes, executionHistory, currentStep, setNodes]);

  // Update React Flow edges when store changes
  useEffect(() => {
    setEdges(reactFlowEdges);
  }, [currentNetwork?.links, setEdges]);

  const onConnect = useCallback(
    (params: Connection) => {
      if (params.source && params.target) {
        addLink({
          source: params.source,
          target: params.target,
          type: 'sub', // Default link type
          weight: 1.0,
        });
      }
    },
    [addLink],
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    onNodeSelect?.(node.id);
  }, [onNodeSelect]);

  const onEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    onEdgeSelect?.(edge.id);
  }, [onEdgeSelect]);

  const onPaneClick = useCallback(() => {
    onNodeSelect?.(null);
    onEdgeSelect?.(null);
  }, [onNodeSelect, onEdgeSelect]);

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
        fitView
        fitViewOptions={{ padding: 0.2 }}
      >
        <Controls />
        <MiniMap
          nodeColor={() => '#f0f0f0'}
          nodeStrokeColor={() => '#333'}
          nodeStrokeWidth={2}
        />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

// Helper functions for node styling - using exact colors from ReCoN draft
function getNodeColor(nodeType: string, state: string): string {
  const stateColors = {
    inactive: '#94a3b8',     // slate-400
    requested: '#0ea5e9',    // sky-500
    active: '#3b82f6',       // blue-500
    suppressed: '#71717a',   // zinc-400
    waiting: '#f59e0b',      // amber-500
    true: '#10b981',         // emerald-500
    confirmed: '#16a34a',    // green-600
    failed: '#dc2626'        // rose-600
  };

  return stateColors[state as keyof typeof stateColors] || '#f5f5f5';
}

function getBorderColor(nodeType: string): string {
  const colors = {
    script: '#1976d2',
    terminal: '#388e3c',
    hybrid: '#7b1fa2'
  };
  return colors[nodeType as keyof typeof colors] || '#666';
}

function getLinkColor(linkType: string): string {
  const colors = {
    sub: '#1976d2',
    sur: '#1976d2',
    por: '#388e3c',
    ret: '#388e3c',
    gen: '#f57c00'
  };
  return colors[linkType as keyof typeof colors] || '#666';
}