// Toolbar component with network building tools

import React, { useState } from 'react';
import { useNetworkStore } from '../stores/networkStore';
import { ReCoNNodeType, ReCoNLinkType } from '../types/recon';

interface ToolbarProps {
  onAddNode?: (type: ReCoNNodeType) => void;
}

export default function Toolbar({ onAddNode }: ToolbarProps) {
  const {
    currentNetwork,
    loadNetwork,
  } = useNetworkStore();

  const [selectedNodeType, setSelectedNodeType] = useState<ReCoNNodeType>('script');
  const [selectedLinkType, setSelectedLinkType] = useState<ReCoNLinkType>('sub');
  const [networkId, setNetworkId] = useState('');

  // link type selection removed; keep local for possible future use

  const handleCreateNewNetwork = () => {
    // Create empty network locally
    const newId = networkId || `network-${Date.now()}`;
    useNetworkStore.setState({
      currentNetwork: {
        id: newId,
        nodes: [],
        links: [],
        stepCount: 0,
        requestedRoots: [],
      },
      isDirty: false,
    });
    setNetworkId('');
  };

  const handleLoadNetwork = async () => {
    if (!networkId.trim()) return;
    try {
      await loadNetwork(networkId);
    } catch (error) {
      console.error('Failed to load network:', error);
    }
  };

  const handleAddNode = () => {
    onAddNode?.(selectedNodeType);
  };

  // layout controls removed for simplicity

  return (
    <div className="bg-white border-b border-gray-200 p-4">
      <div className="flex flex-wrap items-center gap-4">
        {/* Network Operations */}
        <div className="flex items-center gap-2 border-r border-gray-300 pr-4">
          <input
            type="text"
            placeholder="Network ID"
            value={networkId}
            onChange={(e) => setNetworkId(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          />
          <button
            onClick={handleCreateNewNetwork}
            className="px-3 py-1 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600"
          >
            New
          </button>
          <button
            onClick={handleLoadNetwork}
            disabled={!networkId.trim()}
            className="px-3 py-1 bg-green-500 text-white rounded-md text-sm hover:bg-green-600 disabled:bg-gray-300"
          >
            Load Demo
          </button>
        </div>

        {/* Node Creation */}
        <div className="flex items-center gap-2 border-r border-gray-300 pr-4">
          <label className="text-sm font-medium">Add Node:</label>
          <select
            value={selectedNodeType}
            onChange={(e) => setSelectedNodeType(e.target.value as ReCoNNodeType)}
            className="px-2 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="script">Script</option>
            <option value="terminal">Terminal</option>
            <option value="hybrid">Hybrid</option>
          </select>
          <button
            onClick={handleAddNode}
            className="px-3 py-1 bg-purple-500 text-white rounded-md text-sm hover:bg-purple-600"
          >
            Add Node
          </button>
        </div>

        {/* Link Type Selection removed - derived from handles */}

        {/* Layout Options removed */}

        {/* Network Controls removed - execution handles reset */}

        {/* Network Info */}
        {currentNetwork && (
          <div className="text-sm text-gray-600 ml-auto">
            Network: <span className="font-medium">{currentNetwork.id}</span>
            {' | '}
            Nodes: <span className="font-medium">{currentNetwork.nodes.length}</span>
            {' | '}
            Links: <span className="font-medium">{currentNetwork.links.length}</span>
            {' | '}
            Step: <span className="font-medium">{currentNetwork.stepCount}</span>
          </div>
        )}
      </div>

      {/* Help Text */}
      <div className="mt-2 text-xs text-gray-500">
        Double-click canvas to add node • Click and drag to connect nodes • Click to select • Shift+click for multi-select
      </div>
    </div>
  );
}