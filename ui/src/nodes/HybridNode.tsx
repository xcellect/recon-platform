// Custom Hybrid Node component for React Flow

import { Handle, Position, NodeProps } from '@xyflow/react';
import { ReCoNNode } from '../types/recon';

interface HybridNodeData {
  label: string;
  nodeData: ReCoNNode;
}

export default function HybridNode({ data, selected }: NodeProps<HybridNodeData>) {
  const { nodeData } = data;

  const getStateColor = (state: string) => {
    switch (state) {
      case 'inactive': return 'bg-purple-100 border-purple-300';
      case 'requested': return 'bg-blue-100 border-blue-400';
      case 'active': return 'bg-yellow-100 border-yellow-400';
      case 'suppressed': return 'bg-red-100 border-red-400';
      case 'waiting': return 'bg-orange-100 border-orange-400';
      case 'true': return 'bg-green-100 border-green-400';
      case 'confirmed': return 'bg-green-200 border-green-500';
      case 'failed': return 'bg-red-200 border-red-600';
      default: return 'bg-purple-100 border-purple-300';
    }
  };

  const getModeColor = (mode?: string) => {
    switch (mode) {
      case 'neural': return 'text-green-600 bg-green-100';
      case 'implicit': return 'text-purple-600 bg-purple-100';
      case 'explicit': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getModeIcon = (mode?: string) => {
    switch (mode) {
      case 'neural': return '🧠';
      case 'implicit': return '∞';
      case 'explicit': return '◯';
      default: return '◯';
    }
  };

  return (
    <div className={`relative px-4 py-3 shadow-lg rounded-lg border-2 min-w-[140px] ${getStateColor(nodeData.state)} ${
      selected ? 'ring-2 ring-purple-500' : ''
    }`}>
      {/* Input handles */}
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      <Handle type="target" position={Position.Left} className="w-3 h-3" />

      {/* Node content */}
      <div className="text-center">
        {/* Hybrid indicator with mode */}
        <div className="flex items-center justify-center mb-1">
          <span className="text-lg mr-1">{getModeIcon(nodeData.mode)}</span>
          <div className="font-semibold text-sm">{data.label}</div>
        </div>

        <div className="text-xs font-semibold text-purple-700">Hybrid</div>

        {/* Mode badge */}
        <div className={`inline-block px-2 py-1 rounded-full text-xs font-medium mt-1 ${getModeColor(nodeData.mode)}`}>
          {nodeData.mode || 'explicit'}
        </div>

        <div className="text-xs text-gray-500 capitalize mt-1">{nodeData.state}</div>

        {nodeData.activation !== 0 && (
          <div className="text-xs font-mono mt-1">
            Act: {nodeData.activation.toFixed(2)}
          </div>
        )}

        {/* Neural model indicator */}
        {nodeData.mode === 'neural' && (
          <div className="text-xs text-green-600 font-semibold mt-1">
            🔗 Neural Model
          </div>
        )}
      </div>

      {/* Mode indicator badge */}
      <div className={`absolute -top-2 -right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold border-2 border-white ${getModeColor(nodeData.mode)}`}>
        {getModeIcon(nodeData.mode)}
      </div>

      {/* Output handles */}
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}