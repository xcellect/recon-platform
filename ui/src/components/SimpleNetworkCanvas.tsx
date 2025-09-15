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
import { hierarchicalLayout } from '../utils/layout';
import ReCoNNode from './ReCoNNode';

const nodeTypes = {
  reconNode: ReCoNNode,
};

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
  const { currentNetwork, addLink, updateNode } = useNetworkStore();

  // Get current step data for state override
  const currentStepData = executionHistory[currentStep];

  // Convert ReCoN nodes to React Flow nodes
  const reactFlowNodes: Node[] = currentNetwork?.nodes.map(node => {
    // Use execution history state if available, otherwise use base state
    const currentState = currentStepData?.states?.[node.id] || node.state;

    return {
      id: node.id,
      type: 'reconNode',
      position: node.position,
      data: {
        label: `${node.id}\n(${node.type})\n${currentState}`,
        state: currentState,
        nodeType: node.type
      }
    };
  }) || [];

  // Filter and convert ReCoN links to bidirectional React Flow edges
  const reactFlowEdges: Edge[] = (() => {
    if (!currentNetwork?.links) return [];

    // Group links by their bidirectional pairs
    const linkPairs = new Map<string, { primary: any; secondary?: any }>();

    currentNetwork.links.forEach(link => {
      const key = [link.source, link.target].sort().join('-');
      const reverseKey = [link.target, link.source].sort().join('-');

      if (!linkPairs.has(key)) {
        linkPairs.set(key, { primary: link });
      } else {
        const existing = linkPairs.get(key);
        if (existing) {
          existing.secondary = link;
        }
      }
    });

    // Convert pairs to bidirectional edges
    return Array.from(linkPairs.values()).map(({ primary, secondary }) => {
      const isHierarchical = primary.type === 'sub' || primary.type === 'sur' ||
                             (secondary && (secondary.type === 'sub' || secondary.type === 'sur'));

      const isSequential = primary.type === 'por' || primary.type === 'ret' ||
                           (secondary && (secondary.type === 'por' || secondary.type === 'ret'));

      let label = '';
      let sourceNode = primary.source;
      let targetNode = primary.target;

      if (isHierarchical) {
        // For hierarchical relationships, show from parent to child
        if (primary.type === 'sub') {
          label = 'sub/sur';
          sourceNode = primary.source;
          targetNode = primary.target;
        } else if (primary.type === 'sur') {
          label = 'sub/sur';
          sourceNode = primary.target;
          targetNode = primary.source;
        }
      } else if (isSequential) {
        // For sequential relationships, show from predecessor to successor
        if (primary.type === 'por') {
          label = 'por/ret';
          sourceNode = primary.source;
          targetNode = primary.target;
        } else if (primary.type === 'ret') {
          label = 'por/ret';
          sourceNode = primary.target;
          targetNode = primary.source;
        }
      } else {
        // Fallback for single links
        label = primary.type;
        sourceNode = primary.source;
        targetNode = primary.target;
      }

      // Determine source and target handles based on link type
      let sourceHandle = '';
      let targetHandle = '';

      if (isHierarchical) {
        // sub/sur links use top/bottom handles
        sourceHandle = 'bottom-source';  // Parent connects from bottom
        targetHandle = 'top';            // Child connects to top
      } else if (isSequential) {
        // por/ret links use left/right handles
        sourceHandle = 'right-source';   // Predecessor connects from right
        targetHandle = 'left';           // Successor connects to left
      }

      return {
        id: `${sourceNode}-${targetNode}-${label}`,
        source: sourceNode,
        target: targetNode,
        sourceHandle,
        targetHandle,
        label,
        type: 'smoothstep',
        style: {
          stroke: isHierarchical ? '#1976d2' : '#f57c00',
          strokeWidth: 2,
          strokeDasharray: isSequential ? '5,5' : undefined
        },
        markerEnd: {
          type: 'arrowclosed',
          color: isHierarchical ? '#1976d2' : '#f57c00'
        },
        markerStart: {
          type: 'arrowclosed',
          color: isHierarchical ? '#1976d2' : '#f57c00'
        }
      };
    });
  })();

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

  // Apply hierarchical layout only once when network first loads
  useEffect(() => {
    if (currentNetwork && currentNetwork.nodes.length > 0 && currentNetwork.links.length > 0) {
      // Check if ALL nodes are at origin (0,0) - indicating fresh load
      const allNodesAtOrigin = currentNetwork.nodes.every(node =>
        node.position.x === 0 && node.position.y === 0
      );

      if (allNodesAtOrigin) {
        const layoutedNodes = hierarchicalLayout(currentNetwork.nodes, currentNetwork.links);
        layoutedNodes.forEach(node => {
          updateNode(node.id, { position: node.position });
        });
      }
    }
  }, [currentNetwork?.id, currentNetwork?.nodes.length]); // Only run when network ID or node count changes

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
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.1, includeHiddenNodes: false, minZoom: 0.5, maxZoom: 1.5 }}
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

// Styling functions moved to ReCoNNode component