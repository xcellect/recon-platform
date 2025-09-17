// State visualization and monitoring component

import React, { useState } from 'react';
import { useNetworkStore } from '../stores/networkStore';

interface StateStats {
  totalNodes: number;
  stateDistribution: Record<string, number>;
  activationStats: {
    min: number;
    max: number;
    avg: number;
  };
}

export default function StateViewer() {
  const { currentNetwork } = useNetworkStore();
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'history'>('overview');

  if (!currentNetwork) {
    return (
      <div className="bg-gray-100 p-4 rounded-lg">
        <div className="text-gray-500 text-center">
          No network loaded
        </div>
      </div>
    );
  }

  const calculateStats = (): StateStats => {
    const nodes = currentNetwork.nodes;
    const stateDistribution: Record<string, number> = {};
    let activationSum = 0;
    let minActivation = Infinity;
    let maxActivation = -Infinity;

    nodes.forEach(node => {
      // Count states
      stateDistribution[node.state] = (stateDistribution[node.state] || 0) + 1;

      // Track activation stats
      activationSum += node.activation;
      minActivation = Math.min(minActivation, node.activation);
      maxActivation = Math.max(maxActivation, node.activation);
    });

    return {
      totalNodes: nodes.length,
      stateDistribution,
      activationStats: {
        min: minActivation === Infinity ? 0 : minActivation,
        max: maxActivation === -Infinity ? 0 : maxActivation,
        avg: nodes.length > 0 ? activationSum / nodes.length : 0,
      },
    };
  };

  const stats = calculateStats();

  const getStateColor = (state: string) => {
    switch (state) {
      case 'inactive': return 'bg-gray-200 text-gray-800';
      case 'requested': return 'bg-blue-200 text-blue-800';
      case 'active': return 'bg-yellow-200 text-yellow-800';
      case 'suppressed': return 'bg-red-200 text-red-800';
      case 'waiting': return 'bg-orange-200 text-orange-800';
      case 'true': return 'bg-purple-200 text-purple-800';
      case 'confirmed': return 'bg-green-200 text-green-800';
      case 'failed': return 'bg-red-300 text-red-900';
      default: return 'bg-gray-200 text-gray-800';
    }
  };

  const renderOverview = () => (
    <div className="space-y-4">
      {/* Network Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{stats.totalNodes}</div>
          <div className="text-sm text-gray-600">Total Nodes</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">{currentNetwork.links.length}</div>
          <div className="text-sm text-gray-600">Total Links</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">{currentNetwork.stepCount}</div>
          <div className="text-sm text-gray-600">Execution Steps</div>
        </div>
      </div>

      {/* State Distribution */}
      <div>
        <h4 className="text-md font-medium text-gray-800 mb-3">State Distribution</h4>
        <div className="space-y-2">
          {Object.entries(stats.stateDistribution).map(([state, count]) => (
            <div key={state} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStateColor(state)}`}>
                  {state}
                </span>
                <span className="text-sm text-gray-600">{count} nodes</span>
              </div>
              <div className="w-24 bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${getStateColor(state).split(' ')[0]}`}
                  style={{ width: `${(count / stats.totalNodes) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Activation Statistics */}
      <div>
        <h4 className="text-md font-medium text-gray-800 mb-3">Activation Statistics</h4>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="text-center p-2 bg-gray-100 rounded">
            <div className="font-mono font-bold">{stats.activationStats.min.toFixed(3)}</div>
            <div className="text-gray-600">Min</div>
          </div>
          <div className="text-center p-2 bg-gray-100 rounded">
            <div className="font-mono font-bold">{stats.activationStats.avg.toFixed(3)}</div>
            <div className="text-gray-600">Avg</div>
          </div>
          <div className="text-center p-2 bg-gray-100 rounded">
            <div className="font-mono font-bold">{stats.activationStats.max.toFixed(3)}</div>
            <div className="text-gray-600">Max</div>
          </div>
        </div>
      </div>

      {/* Requested Roots */}
      {currentNetwork.requestedRoots.length > 0 && (
        <div>
          <h4 className="text-md font-medium text-gray-800 mb-3">Active Executions</h4>
          <div className="space-y-1">
            {currentNetwork.requestedRoots.map(rootId => (
              <div key={rootId} className="flex items-center justify-between p-2 bg-blue-50 rounded">
                <span className="font-medium text-blue-800">{rootId}</span>
                <span className="text-xs text-blue-600">Root Node</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderDetailed = () => (
    <div className="space-y-4">
      <h4 className="text-md font-medium text-gray-800">Node Details</h4>
      <div className="max-h-80 overflow-y-auto space-y-2">
        {currentNetwork.nodes.map(node => (
          <div key={node.id} className="border border-gray-200 rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">{node.id}</span>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStateColor(node.state)}`}>
                {node.state}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
              <div>Type: <span className="font-medium">{node.type}</span></div>
              <div>Activation: <span className="font-mono">{node.activation.toFixed(3)}</span></div>
              {node.mode && (
                <div>Mode: <span className="font-medium">{node.mode}</span></div>
              )}
              <div>Position: <span className="font-mono">({node.position.x.toFixed(0)}, {node.position.y.toFixed(0)})</span></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderHistory = () => (
    <div className="space-y-4">
      <h4 className="text-md font-medium text-gray-800">Execution History</h4>
      <div className="text-sm text-gray-600">
        <div className="space-y-2">
          <div className="p-2 bg-gray-100 rounded">
            Step {currentNetwork.stepCount}: Current state
          </div>
          {/* Placeholder for execution history */}
          <div className="text-gray-500 italic">
            Execution history tracking not yet implemented
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Network State</h3>
        <div className="flex gap-1">
          <button
            onClick={() => setViewMode('overview')}
            className={`px-3 py-1 text-xs rounded ${
              viewMode === 'overview'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setViewMode('detailed')}
            className={`px-3 py-1 text-xs rounded ${
              viewMode === 'detailed'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
            }`}
          >
            Detailed
          </button>
          <button
            onClick={() => setViewMode('history')}
            className={`px-3 py-1 text-xs rounded ${
              viewMode === 'history'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
            }`}
          >
            History
          </button>
        </div>
      </div>

      {viewMode === 'overview' && renderOverview()}
      {viewMode === 'detailed' && renderDetailed()}
      {viewMode === 'history' && renderHistory()}
    </div>
  );
}