// Simplified Network Canvas that actually works
import React, { useCallback, useEffect, useMemo, useRef } from 'react';
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
  
  // Track if layout has been applied to prevent multiple layout passes
  const layoutAppliedRef = useRef<string | null>(null);
  const isLayoutingRef = useRef(false);

  // Get current step data for state override
  const currentStepData = executionHistory[currentStep];

  // Memoize React Flow nodes to prevent unnecessary re-renders
  const reactFlowNodes = useMemo(() => {
    if (!currentNetwork?.nodes) return [];

    return currentNetwork.nodes.map(node => {
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
    });
  }, [currentNetwork?.nodes, currentStepData]);

  // Memoize React Flow edges to prevent unnecessary re-renders
  const reactFlowEdges = useMemo(() => {
    if (!currentNetwork?.links) return [];

    // Group links by their bidirectional pairs
    const linkPairs = new Map<string, { primary: any; secondary?: any }>();

    currentNetwork.links.forEach(link => {
      const key = [link.source, link.target].sort().join('-');

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
  }, [currentNetwork?.links]);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Apply hierarchical layout only once when network changes and nodes need positioning
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
        const layoutedNodes = hierarchicalLayout(currentNetwork.nodes, currentNetwork.links);
        
        // Update positions in batch to avoid multiple re-renders
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

  // Update React Flow nodes when memoized nodes change
  useEffect(() => {
    setNodes(reactFlowNodes);
  }, [reactFlowNodes, setNodes]);

  // Update React Flow edges when memoized edges change
  useEffect(() => {
    setEdges(reactFlowEdges);
  }, [reactFlowEdges, setEdges]);

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

  // Handle node position changes
  const onNodeDragStop = useCallback((event: React.MouseEvent, node: Node) => {
    updateNode(node.id, {
      position: node.position,
    });
  }, [updateNode]);

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