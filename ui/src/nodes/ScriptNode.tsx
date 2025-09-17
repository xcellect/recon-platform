// Custom Script Node component for React Flow

import { Handle, Position, NodeProps } from '@xyflow/react';
import { ReCoNNode } from '../types/recon';

interface ScriptNodeData {
  label: string;
  nodeData: ReCoNNode;
}

export default function ScriptNode({ data, selected }: NodeProps<ScriptNodeData>) {
  const { nodeData } = data;

  const getStateColor = (state: string) => {
    switch (state) {
      case 'inactive': return 'bg-gray-200 border-gray-400';
      case 'requested': return 'bg-blue-200 border-blue-400';
      case 'active': return 'bg-yellow-200 border-yellow-400';
      case 'suppressed': return 'bg-red-200 border-red-400';
      case 'waiting': return 'bg-orange-200 border-orange-400';
      case 'true': return 'bg-purple-200 border-purple-400';
      case 'confirmed': return 'bg-green-200 border-green-400';
      case 'failed': return 'bg-red-300 border-red-600';
      default: return 'bg-gray-200 border-gray-400';
    }
  };

  const getModeIndicator = (mode?: string) => {
    if (!mode || mode === 'explicit') return null;

    const modeColors = {
      neural: 'bg-green-500',
      implicit: 'bg-purple-500',
    };

    return (
      <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${modeColors[mode as keyof typeof modeColors] || 'bg-gray-500'}`} />
    );
  };

  return (
    <div className={`relative px-4 py-2 shadow-md rounded-md border-2 min-w-[120px] ${getStateColor(nodeData.state)} ${
      selected ? 'ring-2 ring-blue-500' : ''
    }`}>
      {/* Input handles */}
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      <Handle type="target" position={Position.Left} className="w-3 h-3" />

      {/* Node content */}
      <div className="text-center">
        <div className="font-semibold text-sm">{data.label}</div>
        <div className="text-xs text-gray-600 capitalize">{nodeData.type}</div>
        <div className="text-xs text-gray-500 capitalize">{nodeData.state}</div>
        {nodeData.activation !== 0 && (
          <div className="text-xs font-mono">
            Act: {nodeData.activation.toFixed(2)}
          </div>
        )}
      </div>

      {/* Mode indicator */}
      {getModeIndicator(nodeData.mode)}

      {/* Output handles */}
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}