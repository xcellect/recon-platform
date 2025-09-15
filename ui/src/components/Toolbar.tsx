// Toolbar component with network building tools

import React, { useState } from 'react';
import { useNetworkStore } from '../stores/networkStore';
import { ReCoNNodeType, ReCoNLinkType } from '../types/recon';

interface ToolbarProps {
  onAddNode?: (type: ReCoNNodeType) => void;
  onLayoutChange?: (layout: string) => void;
}

export default function Toolbar({ onAddNode, onLayoutChange }: ToolbarProps) {
  const {
    currentNetwork,
    createNetwork,
    loadNetwork,
    saveNetwork,
    deleteNetwork,
    resetNetwork,
    isDirty,
  } = useNetworkStore();

  const [selectedNodeType, setSelectedNodeType] = useState<ReCoNNodeType>('script');
  const [selectedLinkType, setSelectedLinkType] = useState<ReCoNLinkType>('sub');
  const [networkId, setNetworkId] = useState('');

  const handleCreateNetwork = async () => {
    try {
      await createNetwork(networkId || undefined);
      setNetworkId('');
    } catch (error) {
      console.error('Failed to create network:', error);
    }
  };

  const handleLoadNetwork = async () => {
    if (!networkId.trim()) return;
    try {
      await loadNetwork(networkId);
    } catch (error) {
      console.error('Failed to load network:', error);
    }
  };

  const handleSaveNetwork = async () => {
    try {
      await saveNetwork();
    } catch (error) {
      console.error('Failed to save network:', error);
    }
  };

  const handleDeleteNetwork = async () => {
    if (!currentNetwork) return;
    if (!confirm(`Delete network "${currentNetwork.id}"?`)) return;

    try {
      await deleteNetwork(currentNetwork.id);
    } catch (error) {
      console.error('Failed to delete network:', error);
    }
  };

  const handleResetNetwork = async () => {
    if (!currentNetwork) return;
    if (!confirm('Reset network to initial state?')) return;

    try {
      await resetNetwork();
    } catch (error) {
      console.error('Failed to reset network:', error);
    }
  };

  const handleAddNode = () => {
    onAddNode?.(selectedNodeType);
  };

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
            onClick={handleCreateNetwork}
            className="px-3 py-1 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600"
          >
            New
          </button>
          <button
            onClick={handleLoadNetwork}
            disabled={!networkId.trim()}
            className="px-3 py-1 bg-green-500 text-white rounded-md text-sm hover:bg-green-600 disabled:bg-gray-300"
          >
            Load
          </button>
          <button
            onClick={handleSaveNetwork}
            disabled={!currentNetwork || !isDirty}
            className="px-3 py-1 bg-yellow-500 text-white rounded-md text-sm hover:bg-yellow-600 disabled:bg-gray-300"
          >
            Save {isDirty && '*'}
          </button>
          <button
            onClick={handleDeleteNetwork}
            disabled={!currentNetwork}
            className="px-3 py-1 bg-red-500 text-white rounded-md text-sm hover:bg-red-600 disabled:bg-gray-300"
          >
            Delete
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

        {/* Link Type Selection */}
        <div className="flex items-center gap-2 border-r border-gray-300 pr-4">
          <label className="text-sm font-medium">Link Type:</label>
          <select
            value={selectedLinkType}
            onChange={(e) => setSelectedLinkType(e.target.value as ReCoNLinkType)}
            className="px-2 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="sub">sub (hierarchy)</option>
            <option value="sur">sur (response)</option>
            <option value="por">por (sequence)</option>
            <option value="ret">ret (return)</option>
            <option value="gen">gen (general)</option>
          </select>
        </div>

        {/* Layout Options */}
        <div className="flex items-center gap-2 border-r border-gray-300 pr-4">
          <label className="text-sm font-medium">Layout:</label>
          <button
            onClick={() => onLayoutChange?.('hierarchical')}
            className="px-3 py-1 bg-gray-500 text-white rounded-md text-sm hover:bg-gray-600"
          >
            Hierarchical
          </button>
          <button
            onClick={() => onLayoutChange?.('sequence')}
            className="px-3 py-1 bg-gray-500 text-white rounded-md text-sm hover:bg-gray-600"
          >
            Sequence
          </button>
          <button
            onClick={() => onLayoutChange?.('force')}
            className="px-3 py-1 bg-gray-500 text-white rounded-md text-sm hover:bg-gray-600"
          >
            Force
          </button>
        </div>

        {/* Network Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleResetNetwork}
            disabled={!currentNetwork}
            className="px-3 py-1 bg-orange-500 text-white rounded-md text-sm hover:bg-orange-600 disabled:bg-gray-300"
          >
            Reset
          </button>
        </div>

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