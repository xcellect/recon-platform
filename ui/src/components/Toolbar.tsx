// Toolbar component with network building tools

import React, { useState } from 'react';
import { useNetworkStore } from '../stores/networkStore';
import { ReCoNNodeType, ReCoNLinkType } from '../types/recon';
import { reconAPI } from '../services/api';

interface ToolbarProps {
  onAddNode?: (type: ReCoNNodeType) => void;
  executionHistory: any[];
  setExecutionHistory: (history: any[]) => void;
  currentStep: number;
  setCurrentStep: (step: number) => void;
  playing: boolean;
  setPlaying: (playing: boolean) => void;
  speed: number;
  setSpeed: (speed: number) => void;
}

export default function Toolbar({ 
  onAddNode, 
  executionHistory, 
  setExecutionHistory, 
  currentStep, 
  setCurrentStep, 
  playing, 
  setPlaying, 
  speed, 
  setSpeed 
}: ToolbarProps) {
  const {
    currentNetwork,
    loadNetwork,
  } = useNetworkStore();

  const [selectedNodeType, setSelectedNodeType] = useState<ReCoNNodeType>('script');
  const [selectedLinkType, setSelectedLinkType] = useState<ReCoNLinkType>('sub');
  const [selectedRootNode, setSelectedRootNode] = useState('Root');

  // link type selection removed; keep local for possible future use

  const handleCreateNewNetwork = () => {
    // Create empty network locally
    const newId = `network-${Date.now()}`;
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
  };

  const handleAddNode = () => {
    onAddNode?.(selectedNodeType);
  };

  // Execution control handlers
  const handleRequestRoot = async () => {
    if (!currentNetwork || !selectedRootNode) return;

    try {
      // Get current client network state and execute directly
      const store = useNetworkStore.getState();
      const networkData = store.exportLocalGraph();
      
      if (!networkData) {
        throw new Error('No network data to execute');
      }
      
      // Configure terminals with their measurement values
      const terminalConfigs = currentNetwork.nodes
        .filter(node => node.type === 'terminal')
        .map(node => ({
          node_id: node.id,
          measurement_value: (node as any).measurementValue || 0.5
        }));
      
      console.log('Terminal configs:', terminalConfigs);
      
      // Execute the current client state directly
      const response = await reconAPI.executeNetworkDirect(networkData, selectedRootNode, 100, terminalConfigs);

      // Set the execution history for playback
      setExecutionHistory(response.steps);
      setCurrentStep(0);
      setPlaying(false);
    } catch (error) {
      console.error('Failed to execute script:', error);
    }
  };

  const handlePlay = () => {
    if (executionHistory.length > 0) {
      setPlaying(!playing);
    }
  };

  const handleStep = () => {
    if (currentStep < executionHistory.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleReset = () => {
    setCurrentStep(0);
    setPlaying(false);
  };

  // layout controls removed for simplicity

  return (
    <div className="bg-gray-900 border-b border-gray-700 p-4">
      <div className="flex flex-wrap items-center gap-4">
        {/* Network Operations */}
        <div className="flex items-center gap-2 border-r border-gray-600 pr-4">
          <button
            onClick={handleCreateNewNetwork}
            className="px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700"
          >
            New Network
          </button>
        </div>

        {/* Node Creation */}
        <div className="flex items-center gap-2 border-r border-gray-600 pr-4">
          <label className="text-sm font-medium text-gray-300">Add Node:</label>
          <select
            value={selectedNodeType}
            onChange={(e) => setSelectedNodeType(e.target.value as ReCoNNodeType)}
            className="px-2 py-1 border border-gray-600 rounded-md text-sm bg-gray-800 text-white"
          >
            <option value="script">Script</option>
            <option value="terminal">Terminal</option>
            <option value="hybrid">Hybrid</option>
          </select>
          <button
            onClick={handleAddNode}
            className="px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700"
          >
            Add Node
          </button>
        </div>

        {/* Execution Controls */}
        <div className="flex items-center gap-2 border-r border-gray-600 pr-4">
          {/* Root Selection */}
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-300">Root:</label>
            <select
              value={selectedRootNode}
              onChange={(e) => setSelectedRootNode(e.target.value)}
              className="px-2 py-1 border border-gray-600 rounded text-sm bg-gray-800 text-white"
            >
              {currentNetwork?.nodes
                .filter(node => node.type === 'script')
                .map(node => (
                  <option key={node.id} value={node.id}>
                    {node.id}
                  </option>
                )) || []}
            </select>
          </div>

          {/* Request Root Button */}
          <button
            onClick={handleRequestRoot}
            className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
          >
            Request
          </button>

          {/* Playback Controls */}
          {executionHistory.length > 0 && (
            <>
              <button
                onClick={handlePlay}
                className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
              >
                {playing ? 'Pause' : 'Play'}
              </button>
              <button
                onClick={handleStep}
                disabled={currentStep >= executionHistory.length - 1}
                className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 disabled:bg-gray-500"
              >
                Step
              </button>
              <button
                onClick={handleReset}
                className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700"
              >
                Reset
              </button>
              
              {/* Speed Control */}
              <div className="flex items-center gap-1 ml-2">
                <label className="text-xs text-gray-300">Speed:</label>
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                  className="w-16"
                />
                <span className="text-xs text-gray-300">{speed}s</span>
              </div>
            </>
          )}
        </div>

        {/* Link Type Selection removed - derived from handles */}

        {/* Layout Options removed */}

        {/* Network Controls removed - execution handles reset */}

      </div>

      {/* Help Text */}
      <div className="mt-2 text-xs text-gray-400">
        Double-click canvas to add node • Click and drag to connect nodes • Click to select • Shift+click for multi-select
      </div>
    </div>
  );
}