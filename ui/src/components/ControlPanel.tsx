// Execution control panel for ReCoN networks

import React, { useState } from 'react';
import { useNetworkStore, useExecutionStore } from '../stores/networkStore';

export default function ControlPanel() {
  const {
    currentNetwork,
    requestNode,
    executeScript,
    propagateStep,
    resetNetwork,
  } = useNetworkStore();

  const {
    isExecuting,
    currentStep,
    executionResult,
    setExecuting,
    setCurrentStep,
    setExecutionResult,
    reset: resetExecution,
  } = useExecutionStore();

  const [selectedRootNode, setSelectedRootNode] = useState('');
  const [maxSteps, setMaxSteps] = useState(100);
  const [stepByStep, setStepByStep] = useState(false);

  const handleRequestNode = async () => {
    if (!selectedRootNode) return;

    try {
      setExecuting(true);
      await requestNode(selectedRootNode);
    } catch (error) {
      console.error('Failed to request node:', error);
    } finally {
      setExecuting(false);
    }
  };

  const handleExecuteScript = async () => {
    if (!selectedRootNode) return;

    try {
      setExecuting(true);
      setExecutionResult(undefined);
      setCurrentStep(0);

      const result = await executeScript(selectedRootNode, maxSteps);
      setExecutionResult(result.result as any);
      setCurrentStep(result.steps_taken);
    } catch (error) {
      console.error('Failed to execute script:', error);
      setExecutionResult('failed');
    } finally {
      setExecuting(false);
    }
  };

  const handlePropagateStep = async () => {
    try {
      setExecuting(true);
      await propagateStep();
      setCurrentStep(currentStep + 1);
    } catch (error) {
      console.error('Failed to propagate step:', error);
    } finally {
      setExecuting(false);
    }
  };

  const handleReset = async () => {
    try {
      await resetNetwork();
      resetExecution();
    } catch (error) {
      console.error('Failed to reset network:', error);
    }
  };

  const getExecutionResultColor = (result?: string) => {
    switch (result) {
      case 'confirmed': return 'text-green-600 bg-green-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'timeout': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (!currentNetwork) {
    return (
      <div className="bg-gray-100 p-4 rounded-lg">
        <div className="text-gray-500 text-center">
          No network loaded. Create or load a network to begin.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Execution Control</h3>

      {/* Root Node Selection */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Root Node for Execution
        </label>
        <select
          value={selectedRootNode}
          onChange={(e) => setSelectedRootNode(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isExecuting}
        >
          <option value="">Select a root node...</option>
          {currentNetwork.nodes
            .filter(node => node.type === 'script' || node.type === 'hybrid')
            .map(node => (
              <option key={node.id} value={node.id}>
                {node.id} ({node.type})
              </option>
            ))}
        </select>
      </div>

      {/* Execution Settings */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Max Steps
          </label>
          <input
            type="number"
            value={maxSteps}
            onChange={(e) => setMaxSteps(parseInt(e.target.value) || 100)}
            min="1"
            max="1000"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isExecuting}
          />
        </div>
        <div className="flex items-center">
          <label className="flex items-center text-sm text-gray-700">
            <input
              type="checkbox"
              checked={stepByStep}
              onChange={(e) => setStepByStep(e.target.checked)}
              className="mr-2"
              disabled={isExecuting}
            />
            Step-by-step execution
          </label>
        </div>
      </div>

      {/* Execution Buttons */}
      <div className="grid grid-cols-2 gap-3">
        <button
          onClick={handleRequestNode}
          disabled={!selectedRootNode || isExecuting}
          className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          Request Node
        </button>

        {stepByStep ? (
          <button
            onClick={handlePropagateStep}
            disabled={isExecuting}
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            Single Step
          </button>
        ) : (
          <button
            onClick={handleExecuteScript}
            disabled={!selectedRootNode || isExecuting}
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            Execute Script
          </button>
        )}

        <button
          onClick={handleReset}
          disabled={isExecuting}
          className="px-4 py-2 bg-orange-500 text-white rounded-md hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed col-span-2"
        >
          Reset Network
        </button>
      </div>

      {/* Execution Status */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-700">Status:</span>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
            isExecuting ? 'text-blue-600 bg-blue-100' : 'text-gray-600 bg-gray-100'
          }`}>
            {isExecuting ? 'Executing...' : 'Ready'}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-700">Current Step:</span>
          <span className="text-sm text-gray-600">{currentNetwork.stepCount}</span>
        </div>

        {executionResult && (
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-700">Result:</span>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getExecutionResultColor(executionResult)}`}>
              {executionResult.charAt(0).toUpperCase() + executionResult.slice(1)}
            </span>
          </div>
        )}

        {currentNetwork.requestedRoots.length > 0 && (
          <div className="mt-3">
            <span className="text-sm font-medium text-gray-700">Requested Roots:</span>
            <div className="mt-1 flex flex-wrap gap-1">
              {currentNetwork.requestedRoots.map(rootId => (
                <span
                  key={rootId}
                  className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                >
                  {rootId}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="pt-4 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Quick Actions</h4>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <button className="px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200">
            View Messages
          </button>
          <button className="px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200">
            Export State
          </button>
          <button className="px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200">
            Debug Info
          </button>
        </div>
      </div>
    </div>
  );
}