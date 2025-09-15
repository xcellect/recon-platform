import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface ReCoNNodeData {
  label: string;
  state: string;
  nodeType: string;
}

export default function ReCoNNode({ data }: NodeProps<ReCoNNodeData>) {
  const getNodeColor = (nodeType: string, state: string): string => {
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
  };

  const getBorderColor = (nodeType: string): string => {
    const colors = {
      script: '#1976d2',
      terminal: '#388e3c',
      hybrid: '#7b1fa2'
    };
    return colors[nodeType as keyof typeof colors] || '#666';
  };

  return (
    <div
      style={{
        backgroundColor: getNodeColor(data.nodeType, data.state),
        border: `2px solid ${getBorderColor(data.nodeType)}`,
        borderRadius: '8px',
        padding: '10px',
        minWidth: '100px',
        textAlign: 'center',
        fontSize: '12px',
        fontWeight: 'bold',
        color: '#333',
        position: 'relative'
      }}
    >
      {/* Top handle for sub/sur (hierarchical) connections */}
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        style={{
          background: '#1976d2',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />
      <Handle
        type="source"
        position={Position.Top}
        id="top-source"
        style={{
          background: '#1976d2',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />

      {/* Left handle for por/ret (sequential) connections */}
      <Handle
        type="target"
        position={Position.Left}
        id="left"
        style={{
          background: '#f57c00',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />
      <Handle
        type="source"
        position={Position.Left}
        id="left-source"
        style={{
          background: '#f57c00',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />

      {/* Right handle for por/ret (sequential) connections */}
      <Handle
        type="target"
        position={Position.Right}
        id="right"
        style={{
          background: '#f57c00',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="right-source"
        style={{
          background: '#f57c00',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />

      {/* Bottom handle for sub/sur (hierarchical) connections */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="bottom"
        style={{
          background: '#1976d2',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom-source"
        style={{
          background: '#1976d2',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />

      {data.label}
    </div>
  );
}