import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface ReCoNNodeData {
  label: string;
  state: string;
  nodeType: string;
}

export default function ReCoNNode({ data }: NodeProps<ReCoNNodeData>) {
  // Extract node name and type for better display
  const nodeName = data.label.split('\n')[0];
  const isObjectTerminal = nodeName.startsWith('object_');
  const isActionTerminal = nodeName.includes('_terminal');
  
  const getNodeColor = (nodeType: string, state: string): string => {
    const stateColors = {
      inactive: '#374151',     // gray-700 - default inactive state
      requested: '#d97706',    // amber-600 (darker) - node has been requested
      active: '#2563eb',       // blue-600 (darker) - node is actively processing
      suppressed: '#4b5563',   // gray-600 - node is suppressed
      waiting: '#d97706',      // amber-600 (darker) - node is waiting
      true: '#059669',         // emerald-600 (darker) - node returned true
      confirmed: '#16a34a',    // green-600 - node is confirmed
      failed: '#dc2626'        // red-600 (darker) - node failed
    };

    return stateColors[state as keyof typeof stateColors] || '#374151';
  };

  const getBorderColor = (nodeType: string): string => {
    const colors = {
      script: '#dc2626',       // red-600
      terminal: '#991b1b',     // red-800
      hybrid: '#ef4444'        // red-500
    };
    return colors[nodeType as keyof typeof colors] || '#6b7280';
  };

  // Get compact display text
  const getDisplayText = () => {
    if (isObjectTerminal) {
      // Show just "obj_N" for object terminals
      const objNum = nodeName.replace('object_', '');
      return `obj_${objNum}`;
    } else if (isActionTerminal) {
      // Show just "act_N" for action terminals  
      const actionPart = nodeName.replace('_terminal', '');
      return `${actionPart}_t`;
    } else {
      // For other nodes, show full name but shorter
      return nodeName.length > 12 ? nodeName.substring(0, 10) + '...' : nodeName;
    }
  };

  // Adjust node size based on type
  const getNodeSize = () => {
    if (isObjectTerminal || isActionTerminal) {
      return { width: '70px', height: '40px', fontSize: '10px', padding: '4px' };
    } else {
      return { width: '120px', height: '50px', fontSize: '11px', padding: '8px' };
    }
  };

  const nodeSize = getNodeSize();

  return (
    <div
      style={{
        backgroundColor: getNodeColor(data.nodeType, data.state),
        border: `2px solid ${getBorderColor(data.nodeType)}`,
        borderRadius: '6px',
        padding: nodeSize.padding,
        minWidth: nodeSize.width,
        height: nodeSize.height,
        textAlign: 'center',
        fontSize: nodeSize.fontSize,
        fontWeight: '500',
        color: '#ffffff',
        position: 'relative',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden'
      }}
    >
      {/* Top handle for sub/sur (hierarchical) connections */}
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        style={{
          background: '#dc2626',
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
          background: '#dc2626',
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
          background: '#6b7280',
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
          background: '#6b7280',
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
          background: '#6b7280',
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
          background: '#6b7280',
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
          background: '#dc2626',
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
          background: '#dc2626',
          width: '8px',
          height: '8px',
          borderRadius: '50%'
        }}
      />

      {/* Node content */}
      <div style={{ lineHeight: '1.2', textAlign: 'center' }}>
        <div style={{ fontWeight: '600' }}>{getDisplayText()}</div>
        <div style={{ fontSize: '9px', opacity: 0.8, marginTop: '2px' }}>
          {data.state}
        </div>
      </div>
    </div>
  );
}