// Simplified Network Canvas that actually works
import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import ReactFlow, {
  type Node,
  type Edge,
  useNodesState,
  useEdgesState,
  type Connection,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
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

    // Group links by undirected pair to create a single edge per pair
    const pairs = new Map<string, { types: Set<string>; sub?: any; sur?: any; por?: any; ret?: any }>();

    currentNetwork.links.forEach(link => {
      const key = [link.source, link.target].sort().join('::');
      if (!pairs.has(key)) pairs.set(key, { types: new Set() });
      const entry = pairs.get(key)!;
      entry.types.add(link.type);
      (entry as any)[link.type] = link;
    });

    const edges: Edge[] = [];

    pairs.forEach(entry => {
      const hasHier = entry.types.has('sub') || entry.types.has('sur');
      const hasSeq = entry.types.has('por') || entry.types.has('ret');

      if (hasHier) {
        // Orient from parent (sub source) to child (sub target) when available
        const parent = entry.sub ? entry.sub.source : entry.sur.target;
        const child = entry.sub ? entry.sub.target : entry.sur.source;
        edges.push({
          id: `${parent}-${child}-sub/sur`,
          source: parent,
          target: child,
          sourceHandle: 'bottom-source',
          targetHandle: 'top',
          label: 'sub/sur',
          type: 'smoothstep',
          style: { stroke: '#dc2626', strokeWidth: 2 },
          markerStart: { type: 'arrowclosed', color: '#dc2626' },
          markerEnd: { type: 'arrowclosed', color: '#dc2626' },
        } as Edge);
      }

      if (hasSeq) {
        // Orient from predecessor (por source) to successor (por target)
        const pred = entry.por ? entry.por.source : entry.ret.target;
        const succ = entry.por ? entry.por.target : entry.ret.source;
        edges.push({
          id: `${pred}-${succ}-por/ret`,
          source: pred,
          target: succ,
          sourceHandle: 'right-source',
          targetHandle: 'left',
          label: 'por/ret',
          type: 'smoothstep',
          style: { stroke: '#dc2626', strokeWidth: 2, strokeDasharray: '5,5' },
          markerStart: { type: 'arrowclosed', color: '#dc2626' },
          markerEnd: { type: 'arrowclosed', color: '#dc2626' },
        } as Edge);
      }
    });

    // gen or other single-direction links (rare)
    currentNetwork.links.forEach(link => {
      if (link.type !== 'gen') return;
      edges.push({
        id: `${link.source}-${link.target}-gen`,
        source: link.source,
        target: link.target,
        label: 'gen',
        type: 'smoothstep',
        style: { stroke: '#ef4444', strokeWidth: 2, strokeDasharray: '2,2' },
        markerEnd: { type: 'arrowclosed', color: '#ef4444' },
      } as Edge);
    });

    return edges;
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
        // Infer link type from the source handle position
        let inferred: 'sub' | 'sur' | 'por' | 'ret' | 'gen' = 'sub';
        switch (params.sourceHandle) {
          case 'bottom-source': inferred = 'sub'; break;
          case 'top-source': inferred = 'sur'; break;
          case 'right-source': inferred = 'por'; break;
          case 'left-source': inferred = 'ret'; break;
          default: inferred = 'sub';
        }
        addLink({
          source: params.source,
          target: params.target,
          type: inferred,
          weight: 1.0,
        });
      }
    },
    [addLink],
  );

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    onNodeSelect?.(node.id);
  }, [onNodeSelect]);

  const onEdgeClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    onEdgeSelect?.(edge.id);
  }, [onEdgeSelect]);

  const onPaneClick = useCallback(() => {
    onNodeSelect?.(null);
    onEdgeSelect?.(null);
  }, [onNodeSelect, onEdgeSelect]);

  // Handle node position changes
  const onNodeDragStop = useCallback((_event: React.MouseEvent, node: Node) => {
    updateNode(node.id, {
      position: node.position,
    });
  }, [updateNode]);

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
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.1, includeHiddenNodes: false, minZoom: 0.5, maxZoom: 1.5 }}
        onEdgesDelete={(eds) => {
          const store = useNetworkStore.getState();
          eds.forEach(e => {
            const id = e.id || '';
            const lastDash = id.lastIndexOf('-');
            if (lastDash === -1) return;
            const srcTgt = id.substring(0, lastDash); // source-target
            const label = id.substring(lastDash + 1); // 'sub/sur' or 'por/ret' or 'gen'
            const sep = srcTgt.indexOf('-');
            if (sep === -1) return;
            const source = srcTgt.substring(0, sep);
            const target = srcTgt.substring(sep + 1);
            if (label === 'sub/sur') {
              store.deleteLink(`${source}-${target}-sub`);
              store.deleteLink(`${target}-${source}-sur`);
            } else if (label === 'por/ret') {
              store.deleteLink(`${source}-${target}-por`);
              store.deleteLink(`${target}-${source}-ret`);
            } else {
              store.deleteLink(id);
            }
          });
        }}
        onNodesDelete={(nds) => {
          const store = useNetworkStore.getState();
          nds.forEach(n => store.deleteNode(n.id));
        }}
      >
        <Controls />
        <MiniMap
          nodeColor={() => '#dc2626'}
          nodeStrokeColor={() => '#991b1b'}
          nodeStrokeWidth={2}
          maskColor="rgba(0, 0, 0, 0.6)"
        />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} color="#6b7280" />
      </ReactFlow>
    </div>
  );
}

// Styling functions moved to ReCoNNode component