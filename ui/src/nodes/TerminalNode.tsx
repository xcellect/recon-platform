// Custom Terminal Node component for React Flow

import { Handle, Position, NodeProps } from '@xyflow/react';
import { ReCoNNode } from '../types/recon';

interface TerminalNodeData {
  label: string;
  nodeData: ReCoNNode;
}

export default function TerminalNode({ data, selected }: NodeProps<TerminalNodeData>) {
  const { nodeData } = data;

  const getStateColor = (state: string) => {
    switch (state) {
      case 'inactive': return 'bg-gray-100 border-gray-300';
      case 'requested': return 'bg-blue-100 border-blue-300';
      case 'active': return 'bg-yellow-100 border-yellow-300';
      case 'true': return 'bg-green-100 border-green-300';
      case 'confirmed': return 'bg-green-200 border-green-500';
      case 'failed': return 'bg-red-200 border-red-500';
      default: return 'bg-gray-100 border-gray-300';
    }
  };

  return (
    <div className={`relative px-3 py-2 shadow-md rounded-lg border-2 min-w-[100px] ${getStateColor(nodeData.state)} ${
      selected ? 'ring-2 ring-blue-500' : ''
    }`}>
      {/* Input handle (only receives requests) */}
      <Handle type="target" position={Position.Top} className="w-3 h-3" />

      {/* Node content */}
      <div className="text-center">
        {/* Terminal indicator */}
        <div className="flex items-center justify-center mb-1">
          <div className="w-2 h-2 bg-blue-500 rounded-full mr-1" />
          <div className="font-semibold text-sm">{data.label}</div>
        </div>

        <div className="text-xs text-gray-600">Terminal</div>
        <div className="text-xs text-gray-500 capitalize">{nodeData.state}</div>

        {nodeData.activation !== 0 && (
          <div className="text-xs font-mono">
            Measure: {nodeData.activation.toFixed(2)}
          </div>
        )}

        {/* Show measurement indicator */}
        {nodeData.state === 'true' && (
          <div className="text-xs text-green-600 font-semibold">✓ Measured</div>
        )}
        {nodeData.state === 'failed' && (
          <div className="text-xs text-red-600 font-semibold">✗ Failed</div>
        )}
      </div>

      {/* Output handle (only sends confirmations) */}
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
}