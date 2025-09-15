// Main ReCoN Network Builder Application
import React, { useState, useCallback, useEffect } from 'react';
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

// ReCoN Network Types
type ReCoNNodeType = 'script' | 'terminal' | 'hybrid';
type ReCoNState = 'inactive' | 'requested' | 'active' | 'confirmed' | 'failed';

interface ReCoNNodeData {
  label: string;
  nodeType: ReCoNNodeType;
  state: ReCoNState;
  activation: number;
}

// Initial demo network
const initialNodes: Node<ReCoNNodeData>[] = [
  {
    id: 'root',
    type: 'default',
    position: { x: 250, y: 25 },
    data: {
      label: 'Root Script',
      nodeType: 'script',
      state: 'inactive',
      activation: 0
    },
  },
  {
    id: 'child1',
    type: 'default',
    position: { x: 100, y: 125 },
    data: {
      label: 'Child A',
      nodeType: 'script',
      state: 'inactive',
      activation: 0
    },
  },
  {
    id: 'child2',
    type: 'default',
    position: { x: 400, y: 125 },
    data: {
      label: 'Child B',
      nodeType: 'script',
      state: 'inactive',
      activation: 0
    },
  },
  {
    id: 'terminal1',
    type: 'default',
    position: { x: 100, y: 225 },
    data: {
      label: 'Terminal 1',
      nodeType: 'terminal',
      state: 'inactive',
      activation: 0
    },
  },
  {
    id: 'terminal2',
    type: 'default',
    position: { x: 400, y: 225 },
    data: {
      label: 'Terminal 2',
      nodeType: 'terminal',
      state: 'inactive',
      activation: 0
    },
  },
];

const initialEdges: Edge[] = [
  { id: 'root-child1', source: 'root', target: 'child1', label: 'sub' },
  { id: 'root-child2', source: 'root', target: 'child2', label: 'sub' },
  { id: 'child1-terminal1', source: 'child1', target: 'terminal1', label: 'sub' },
  { id: 'child2-terminal2', source: 'child2', target: 'terminal2', label: 'sub' },
  { id: 'child1-child2', source: 'child1', target: 'child2', label: 'por' },
];

export default function ReCoNApp() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionStep, setExecutionStep] = useState(0);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  // Simulate ReCoN execution
  const executeNetwork = useCallback(async () => {
    if (isExecuting) return;

    setIsExecuting(true);
    setExecutionStep(0);

    // Reset all nodes to inactive
    setNodes(nodes => nodes.map(node => ({
      ...node,
      data: { ...node.data, state: 'inactive' as ReCoNState, activation: 0 }
    })));

    // Step 1: Request root
    setTimeout(() => {
      setNodes(nodes => nodes.map(node =>
        node.id === 'root'
          ? { ...node, data: { ...node.data, state: 'requested' as ReCoNState, activation: 1.0 } }
          : node
      ));
      setExecutionStep(1);
    }, 500);

    // Step 2: Root becomes active, requests children
    setTimeout(() => {
      setNodes(nodes => nodes.map(node => {
        if (node.id === 'root') {
          return { ...node, data: { ...node.data, state: 'active' as ReCoNState } };
        }
        if (node.id === 'child1' || node.id === 'child2') {
          return { ...node, data: { ...node.data, state: 'requested' as ReCoNState, activation: 1.0 } };
        }
        return node;
      }));
      setExecutionStep(2);
    }, 1500);

    // Step 3: Children become active, request terminals
    setTimeout(() => {
      setNodes(nodes => nodes.map(node => {
        if (node.id === 'child1' || node.id === 'child2') {
          return { ...node, data: { ...node.data, state: 'active' as ReCoNState } };
        }
        if (node.id === 'terminal1' || node.id === 'terminal2') {
          return { ...node, data: { ...node.data, state: 'requested' as ReCoNState, activation: 1.0 } };
        }
        return node;
      }));
      setExecutionStep(3);
    }, 2500);

    // Step 4: Terminals confirm (simulate measurement)
    setTimeout(() => {
      setNodes(nodes => nodes.map(node => {
        if (node.id === 'terminal1' || node.id === 'terminal2') {
          const confirmed = Math.random() > 0.3; // 70% success rate
          return {
            ...node,
            data: {
              ...node.data,
              state: confirmed ? 'confirmed' as ReCoNState : 'failed' as ReCoNState,
              activation: confirmed ? 1.0 : 0.0
            }
          };
        }
        return node;
      }));
      setExecutionStep(4);
    }, 3500);

    // Step 5: Propagate confirmations back up
    setTimeout(() => {
      setNodes(nodes => {
        const terminal1Confirmed = nodes.find(n => n.id === 'terminal1')?.data.state === 'confirmed';
        const terminal2Confirmed = nodes.find(n => n.id === 'terminal2')?.data.state === 'confirmed';

        return nodes.map(node => {
          if (node.id === 'child1') {
            return { ...node, data: { ...node.data, state: terminal1Confirmed ? 'confirmed' as ReCoNState : 'failed' as ReCoNState } };
          }
          if (node.id === 'child2') {
            return { ...node, data: { ...node.data, state: terminal2Confirmed ? 'confirmed' as ReCoNState : 'failed' as ReCoNState } };
          }
          if (node.id === 'root') {
            const allConfirmed = terminal1Confirmed && terminal2Confirmed;
            return { ...node, data: { ...node.data, state: allConfirmed ? 'confirmed' as ReCoNState : 'failed' as ReCoNState } };
          }
          return node;
        });
      });
      setExecutionStep(5);
      setIsExecuting(false);
    }, 4500);
  }, [isExecuting, setNodes]);

  const resetNetwork = useCallback(() => {
    setNodes(nodes => nodes.map(node => ({
      ...node,
      data: { ...node.data, state: 'inactive' as ReCoNState, activation: 0 }
    })));
    setExecutionStep(0);
    setIsExecuting(false);
  }, [setNodes]);

  // Custom node styling based on state
  useEffect(() => {
    setNodes(nodes => nodes.map(node => {
      const { state, nodeType } = node.data;
      let backgroundColor = '#ffffff';
      let borderColor = '#1a192b';

      // State-based coloring
      switch (state) {
        case 'inactive':
          backgroundColor = '#f7fafc';
          borderColor = '#cbd5e0';
          break;
        case 'requested':
          backgroundColor = '#bee3f8';
          borderColor = '#3182ce';
          break;
        case 'active':
          backgroundColor = '#faf089';
          borderColor = '#d69e2e';
          break;
        case 'confirmed':
          backgroundColor = '#c6f6d5';
          borderColor = '#38a169';
          break;
        case 'failed':
          backgroundColor = '#fed7d7';
          borderColor = '#e53e3e';
          break;
      }

      // Node type indicators
      if (nodeType === 'terminal') {
        borderColor = '#805ad5';
      }

      return {
        ...node,
        style: {
          ...node.style,
          backgroundColor,
          borderColor,
          borderWidth: '2px',
          borderRadius: '8px',
          padding: '10px',
        }
      };
    }));
  }, [nodes, setNodes]);

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200 p-4">
        <h1 className="text-2xl font-bold text-gray-900">ReCoN Network Builder</h1>
        <p className="text-sm text-gray-600">
          Visual interface for Request Confirmation Networks - Demo showing hierarchical script execution
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex items-center gap-4">
          <button
            onClick={executeNetwork}
            disabled={isExecuting}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-300"
          >
            {isExecuting ? 'Executing...' : 'Execute Network'}
          </button>

          <button
            onClick={resetNetwork}
            disabled={isExecuting}
            className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 disabled:bg-gray-300"
          >
            Reset
          </button>

          <div className="text-sm text-gray-600">
            Execution Step: {executionStep}/5
          </div>

          <div className="ml-auto text-xs text-gray-500">
            <div className="flex gap-4">
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 bg-gray-200 border border-gray-400 rounded"></div>
                Inactive
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 bg-blue-200 border border-blue-400 rounded"></div>
                Requested
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 bg-yellow-200 border border-yellow-600 rounded"></div>
                Active
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 bg-green-200 border border-green-600 rounded"></div>
                Confirmed
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 bg-red-200 border border-red-600 rounded"></div>
                Failed
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Network Canvas */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
        >
          <Controls />
          <MiniMap />
          <Background variant="dots" gap={12} size={1} />
        </ReactFlow>
      </div>

      {/* Footer */}
      <div className="bg-white border-t border-gray-200 px-4 py-2">
        <div className="text-xs text-gray-500 text-center">
          ReCoN Platform - Request Confirmation Networks for Neuro-Symbolic Script Execution
        </div>
      </div>
    </div>
  );
}