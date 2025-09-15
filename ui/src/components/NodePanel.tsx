// Node configuration and properties panel

import React, { useState, useEffect } from 'react';
import { useNetworkStore } from '../stores/networkStore';
import { ReCoNNodeType, ExecutionMode } from '../types/recon';

export default function NodePanel() {
  const {
    selectedNode,
    selectedLink,
    updateNode,
    updateLink,
    deleteNode,
    deleteLink,
  } = useNetworkStore();

  // Node editing state
  const [nodeId, setNodeId] = useState('');
  const [nodeType, setNodeType] = useState<ReCoNNodeType>('script');
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('explicit');

  // Link editing state
  const [linkWeight, setLinkWeight] = useState(1.0);

  useEffect(() => {
    if (selectedNode) {
      setNodeId(selectedNode.id);
      setNodeType(selectedNode.type);
      setExecutionMode(selectedNode.mode || 'explicit');
    }
  }, [selectedNode]);

  useEffect(() => {
    if (selectedLink) {
      setLinkWeight(selectedLink.weight);
    }
  }, [selectedLink]);

  const handleUpdateNode = () => {
    if (!selectedNode) return;

    updateNode(selectedNode.id, {
      type: nodeType,
      mode: executionMode,
    });
  };

  const handleUpdateLink = () => {
    if (!selectedLink) return;

    updateLink(selectedLink.id, {
      weight: linkWeight,
    });
  };

  const handleDeleteNode = () => {
    if (!selectedNode) return;
    if (!confirm(`Delete node "${selectedNode.id}"?`)) return;

    deleteNode(selectedNode.id);
  };

  const handleDeleteLink = () => {
    if (!selectedLink) return;
    if (!confirm(`Delete link "${selectedLink.id}"?`)) return;

    deleteLink(selectedLink.id);
  };

  if (!selectedNode && !selectedLink) {
    return (
      <div className="bg-gray-100 p-4 rounded-lg">
        <div className="text-gray-500 text-center">
          Select a node or link to view properties
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
      {selectedNode && (
        <>
          <h3 className="text-lg font-semibold text-gray-800">Node Properties</h3>

          {/* Node ID */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Node ID
            </label>
            <input
              type="text"
              value={nodeId}
              onChange={(e) => setNodeId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled // Node ID changes not implemented yet
            />
          </div>

          {/* Node Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Node Type
            </label>
            <select
              value={nodeType}
              onChange={(e) => setNodeType(e.target.value as ReCoNNodeType)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="script">Script Node</option>
              <option value="terminal">Terminal Node</option>
              <option value="hybrid">Hybrid Node</option>
            </select>
          </div>

          {/* Execution Mode (for hybrid nodes) */}
          {(nodeType === 'hybrid' || selectedNode.type === 'hybrid') && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Execution Mode
              </label>
              <select
                value={executionMode}
                onChange={(e) => setExecutionMode(e.target.value as ExecutionMode)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="explicit">Explicit (Symbolic)</option>
                <option value="neural">Neural (PyTorch)</option>
                <option value="implicit">Implicit (Continuous)</option>
              </select>
            </div>
          )}

          {/* Current State (read-only) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Current State
            </label>
            <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700">
              {selectedNode.state.charAt(0).toUpperCase() + selectedNode.state.slice(1)}
            </div>
          </div>

          {/* Activation Level (read-only) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Activation Level
            </label>
            <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-mono">
              {selectedNode.activation.toFixed(3)}
            </div>
          </div>

          {/* Terminal Node Configuration */}
          {nodeType === 'terminal' && (
            <div className="space-y-3 pt-4 border-t border-gray-200">
              <h4 className="text-md font-medium text-gray-800">Terminal Configuration</h4>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Measurement Function
                </label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option value="default">Default Measurement</option>
                  <option value="random">Random (0.8 threshold)</option>
                  <option value="neural">Neural Model</option>
                  <option value="custom">Custom Function</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confirmation Threshold
                </label>
                <input
                  type="number"
                  defaultValue="0.8"
                  step="0.1"
                  min="0"
                  max="1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          )}

          {/* Neural Configuration */}
          {executionMode === 'neural' && (
            <div className="space-y-3 pt-4 border-t border-gray-200">
              <h4 className="text-md font-medium text-gray-800">Neural Configuration</h4>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model Type
                </label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option value="linear">Linear Model</option>
                  <option value="mlp">Multi-Layer Perceptron</option>
                  <option value="cnn">Convolutional Neural Network</option>
                  <option value="transformer">Transformer</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Input Dimensions
                </label>
                <input
                  type="text"
                  placeholder="e.g., [8, 8] for ARC grids"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-2 pt-4 border-t border-gray-200">
            <button
              onClick={handleUpdateNode}
              className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            >
              Update Node
            </button>
            <button
              onClick={handleDeleteNode}
              className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600"
            >
              Delete
            </button>
          </div>
        </>
      )}

      {selectedLink && (
        <>
          <h3 className="text-lg font-semibold text-gray-800">Link Properties</h3>

          {/* Link Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Source
              </label>
              <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700">
                {selectedLink.source}
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Target
              </label>
              <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700">
                {selectedLink.target}
              </div>
            </div>
          </div>

          {/* Link Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Link Type
            </label>
            <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700">
              {selectedLink.type} - {getLinkDescription(selectedLink.type)}
            </div>
          </div>

          {/* Link Weight */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Weight
            </label>
            <input
              type="number"
              value={linkWeight}
              onChange={(e) => setLinkWeight(parseFloat(e.target.value) || 1.0)}
              step="0.1"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-4 border-t border-gray-200">
            <button
              onClick={handleUpdateLink}
              className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            >
              Update Link
            </button>
            <button
              onClick={handleDeleteLink}
              className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600"
            >
              Delete
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function getLinkDescription(linkType: string): string {
  switch (linkType) {
    case 'sub': return 'Hierarchical subordinate';
    case 'sur': return 'Hierarchical superior';
    case 'por': return 'Sequence predecessor';
    case 'ret': return 'Sequence successor';
    case 'gen': return 'General connection';
    default: return 'Unknown link type';
  }
}