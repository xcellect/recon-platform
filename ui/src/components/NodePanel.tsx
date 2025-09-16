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
  const [nodeActivation, setNodeActivation] = useState('0');
  const [measurementValue, setMeasurementValue] = useState('0.5');
  const [isEditing, setIsEditing] = useState(false);

  // Link editing state
  const [linkWeight, setLinkWeight] = useState('1.0');
  const [subWeight, setSubWeight] = useState('1.0');
  const [surWeight, setSurWeight] = useState('1.0');
  const [porWeight, setPorWeight] = useState('1.0');
  const [retWeight, setRetWeight] = useState('1.0');

  // Only sync when a different node is selected, not on updates
  useEffect(() => {
    if (selectedNode) {
      console.log('NodePanel: new node selected:', selectedNode);
      setNodeId(selectedNode.id);
      setNodeType(selectedNode.type);
      setExecutionMode(selectedNode.mode || 'explicit');
      setNodeActivation(selectedNode.activation.toString());
      // Initialize terminal measurement setting
      setMeasurementValue((selectedNode as any).measurementValue?.toString() || '0.5');
      setIsEditing(false);
    }
  }, [selectedNode?.id]); // Only depend on ID change, not the whole object

  useEffect(() => {
    if (selectedLink) {
      setLinkWeight(selectedLink.weight.toString());
      // Sync combined link weights
      if (selectedLink.type === 'sub/sur') {
        const subLink = (selectedLink as any)._subLink;
        const surLink = (selectedLink as any)._surLink;
        if (subLink) setSubWeight(subLink.weight.toString());
        if (surLink) setSurWeight(surLink.weight.toString());
      } else if (selectedLink.type === 'por/ret') {
        const porLink = (selectedLink as any)._porLink;
        const retLink = (selectedLink as any)._retLink;
        if (porLink) setPorWeight(porLink.weight.toString());
        if (retLink) setRetWeight(retLink.weight.toString());
      }
    }
  }, [selectedLink?.id]); // Only sync on selection change

  const handleUpdateNode = () => {
    if (!selectedNode) return;

    console.log('Updating node:', selectedNode.id, 'with:', { id: nodeId, type: nodeType, mode: executionMode, activation: nodeActivation, measurementValue });
    updateNode(selectedNode.id, {
      id: nodeId, // Allow ID changes
      type: nodeType,
      mode: executionMode,
      activation: parseFloat(nodeActivation) || 0,
      measurementValue: parseFloat(measurementValue) || 0.5,
    } as any);
    setIsEditing(false); // Reset editing flag after update
  };

  const handleUpdateLink = () => {
    if (!selectedLink) return;
    // Only for single links - combined links update directly via onChange
    const numWeight = parseFloat(linkWeight) || 1.0;
    console.log('Updating single link weight to:', numWeight);
    updateLink(selectedLink.id, {
      weight: numWeight,
    });
  };

  const handleDeleteNode = () => {
    if (!selectedNode) return;
    if (!confirm(`Delete node "${selectedNode.id}"?`)) return;

    deleteNode(selectedNode.id);
  };

  const handleDeleteLink = () => {
    if (!selectedLink) return;
    
    const linkPairText = selectedLink.type === 'sub/sur' ? 'sub/sur pair' :
                        selectedLink.type === 'por/ret' ? 'por/ret pair' :
                        `${selectedLink.type} link`;
    
    if (!confirm(`Delete ${linkPairText} between "${selectedLink.source}" and "${selectedLink.target}"?`)) return;

    // Handle combined link deletion
    if (selectedLink.type === 'sub/sur') {
      const subLink = (selectedLink as any)._subLink;
      const surLink = (selectedLink as any)._surLink;
      if (subLink) deleteLink(subLink.id);
      if (surLink) deleteLink(surLink.id);
    } else if (selectedLink.type === 'por/ret') {
      const porLink = (selectedLink as any)._porLink;
      const retLink = (selectedLink as any)._retLink;
      if (porLink) deleteLink(porLink.id);
      if (retLink) deleteLink(retLink.id);
    } else {
      // Single link
      deleteLink(selectedLink.id);
    }
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
              onChange={(e) => {
                console.log('NodePanel: changing nodeId to:', e.target.value);
                setIsEditing(true);
                setNodeId(e.target.value);
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Node Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Node Type
            </label>
            <select
              value={nodeType}
              onChange={(e) => {
                console.log('NodePanel: changing nodeType to:', e.target.value);
                setIsEditing(true);
                setNodeType(e.target.value as ReCoNNodeType);
              }}
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
                onChange={(e) => {
                  setIsEditing(true);
                  setExecutionMode(e.target.value as ExecutionMode);
                }}
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

          {/* Activation Level */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Activation Level
            </label>
            <input
              type="number"
              value={nodeActivation}
              onChange={(e) => {
                console.log('NodePanel: changing activation to:', e.target.value);
                setIsEditing(true);
                setNodeActivation(e.target.value);
              }}
              step="0.001"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
            />
          </div>

          {/* Terminal Node Configuration */}
          {nodeType === 'terminal' && (
            <div className="space-y-3 pt-4 border-t border-gray-200">
              <h4 className="text-md font-medium text-gray-800">Terminal Configuration</h4>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Measurement Value (0.0 - 1.0)
                </label>
                <input
                  type="number"
                  value={measurementValue}
                  onChange={(e) => {
                    setIsEditing(true);
                    setMeasurementValue(e.target.value);
                  }}
                  step="0.1"
                  min="0"
                  max="1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded">
                <strong>Behavior:</strong> {parseFloat(measurementValue) > 0.8 ? '✅ Will CONFIRM' : '❌ Will FAIL'}
                <div className="text-xs mt-1">
                  Threshold: 0.8 (measurement &gt; 0.8 confirms, &lt; 0.8 fails)
                </div>
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

          {/* Combined Link Pair Properties */}
          {(selectedLink.type === 'sub/sur' || selectedLink.type === 'por/ret') && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Link Pair Type
                </label>
                <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700">
                  {selectedLink.type} - Bidirectional {selectedLink.type === 'sub/sur' ? 'hierarchy' : 'sequence'}
                </div>
              </div>

              {/* Individual Link Properties */}
              {selectedLink.type === 'sub/sur' && (selectedLink as any)._subLink && (selectedLink as any)._surLink && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      SUB Weight ({selectedLink.source} → {selectedLink.target})
                    </label>
                    <input
                      type="number"
                      value={subWeight}
                      onChange={(e) => {
                        const value = e.target.value;
                        console.log('Changing SUB weight to:', value);
                        setSubWeight(value);
                      }}
                      step="0.1"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      SUR Weight ({selectedLink.target} → {selectedLink.source})
                    </label>
                    <input
                      type="number"
                      value={surWeight}
                      onChange={(e) => {
                        const value = e.target.value;
                        console.log('Changing SUR weight to:', value);
                        setSurWeight(value);
                      }}
                      step="0.1"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              )}

              {selectedLink.type === 'por/ret' && (selectedLink as any)._porLink && (selectedLink as any)._retLink && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      POR Weight ({selectedLink.source} → {selectedLink.target})
                    </label>
                    <input
                      type="number"
                      value={porWeight}
                      onChange={(e) => {
                        const value = e.target.value;
                        console.log('Changing POR weight to:', value);
                        setPorWeight(value);
                      }}
                      step="0.1"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      RET Weight ({selectedLink.target} → {selectedLink.source})
                    </label>
                    <input
                      type="number"
                      value={retWeight}
                      onChange={(e) => {
                        const value = e.target.value;
                        console.log('Changing RET weight to:', value);
                        setRetWeight(value);
                      }}
                      step="0.1"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              )}

              {/* Update button for combined links */}
              <div className="pt-4 border-t border-gray-200">
                <button
                  onClick={() => {
                    if (selectedLink.type === 'sub/sur') {
                      const subLink = (selectedLink as any)._subLink;
                      const surLink = (selectedLink as any)._surLink;
                      if (subLink) updateLink(subLink.id, { weight: parseFloat(subWeight) || 1.0 });
                      if (surLink) updateLink(surLink.id, { weight: parseFloat(surWeight) || 1.0 });
                      console.log('Updated sub/sur weights:', { sub: subWeight, sur: surWeight });
                    } else if (selectedLink.type === 'por/ret') {
                      const porLink = (selectedLink as any)._porLink;
                      const retLink = (selectedLink as any)._retLink;
                      if (porLink) updateLink(porLink.id, { weight: parseFloat(porWeight) || 1.0 });
                      if (retLink) updateLink(retLink.id, { weight: parseFloat(retWeight) || 1.0 });
                      console.log('Updated por/ret weights:', { por: porWeight, ret: retWeight });
                    }
                  }}
                  className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                >
                  Update Weights
                </button>
              </div>
            </div>
          )}

          {/* Single Link Properties */}
          {selectedLink.type !== 'sub/sur' && selectedLink.type !== 'por/ret' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Link Type
                </label>
                <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700">
                  {selectedLink.type} - {getLinkDescription(selectedLink.type)}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Weight
                </label>
              <input
                type="number"
                value={linkWeight}
                onChange={(e) => {
                  const value = e.target.value;
                  console.log('Changing single link weight to:', value);
                  setLinkWeight(value);
                }}
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-2 pt-4 border-t border-gray-200">
            {selectedLink.type !== 'sub/sur' && selectedLink.type !== 'por/ret' && (
              <button
                onClick={handleUpdateLink}
                className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
              >
                Update Link
              </button>
            )}
            <button
              onClick={handleDeleteLink}
              className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600"
            >
              Delete {selectedLink.type === 'sub/sur' || selectedLink.type === 'por/ret' ? 'Pair' : 'Link'}
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