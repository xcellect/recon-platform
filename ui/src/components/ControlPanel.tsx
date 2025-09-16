import { useState, useEffect } from 'react';
import { useNetworkStore } from '../stores/networkStore';
import { reconAPI } from '../services/api';

interface ControlPanelProps {
  executionHistory: any[];
  setExecutionHistory: (history: any[]) => void;
  currentStep: number;
  setCurrentStep: (step: number) => void;
  playing: boolean;
  setPlaying: (playing: boolean) => void;
  speed: number;
  setSpeed: (speed: number) => void;
}

export default function ControlPanel({
  executionHistory,
  setExecutionHistory,
  currentStep,
  setCurrentStep,
  playing,
  setPlaying,
  speed,
  setSpeed
}: ControlPanelProps) {
  const { currentNetwork } = useNetworkStore();
  const [selectedRootNode, setSelectedRootNode] = useState('Root');

  // Auto-play through execution history
  useEffect(() => {
    if (playing && currentStep < executionHistory.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(currentStep + 1);
      }, speed * 1000);
      return () => clearTimeout(timer);
    } else if (playing && currentStep >= executionHistory.length - 1) {
      setPlaying(false);
    }
  }, [playing, currentStep, speed, executionHistory.length, setCurrentStep, setPlaying]);

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

  if (!currentNetwork) {
    return (
      <div className="text-gray-500 text-center">
        No network loaded
      </div>
    );
  }

  const currentStepData = executionHistory[currentStep];
  const rootState = currentStepData?.states?.[selectedRootNode] || 'inactive';

  return (
    <div className="space-y-4">
      {/* Main Controls */}
      <div className="flex items-center gap-4">
        {/* Root Selection */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium">Root:</label>
          <select
            value={selectedRootNode}
            onChange={(e) => setSelectedRootNode(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-sm"
          >
            {currentNetwork.nodes
              .filter(node => node.type === 'script')
              .map(node => (
                <option key={node.id} value={node.id}>
                  {node.id}
                </option>
              ))}
          </select>
        </div>

        {/* Request Root Button */}
        <button
          onClick={handleRequestRoot}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Request Root
        </button>

        {/* Playback Controls */}
        {executionHistory.length > 0 && (
          <>
            <button
              onClick={handlePlay}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              {playing ? 'Pause' : 'Play'}
            </button>

            <button
              onClick={handleStep}
              disabled={currentStep >= executionHistory.length - 1}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 disabled:bg-gray-300"
            >
              Step
            </button>

            <button
              onClick={handleReset}
              className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600"
            >
              Reset
            </button>
          </>
        )}
      </div>

      {/* Speed Control */}
      {executionHistory.length > 0 && (
        <div className="flex items-center gap-2">
          <label className="text-sm">Speed:</label>
          <input
            type="range"
            min="0.1"
            max="2"
            step="0.1"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="w-32"
          />
          <span className="text-sm text-gray-600">{speed}s</span>
        </div>
      )}

      {/* Status Display */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <span className="font-medium">Root State:</span>
          <div className={`inline-block ml-2 px-2 py-1 rounded text-xs ${
            rootState === 'confirmed' ? 'bg-green-100 text-green-800' :
            rootState === 'failed' ? 'bg-red-100 text-red-800' :
            rootState === 'active' ? 'bg-blue-100 text-blue-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {rootState}
          </div>
        </div>

        <div>
          <span className="font-medium">Messages:</span>
          <span className="ml-2">{currentStepData?.messages?.length || 0}</span>
        </div>

        <div>
          <span className="font-medium">Step:</span>
          <span className="ml-2">{currentStep} / {executionHistory.length > 0 ? executionHistory.length - 1 : 0}</span>
        </div>
      </div>

      {/* Messages Display */}
      {currentStepData?.messages && currentStepData.messages.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Current Messages:</div>
          <div className="space-y-1">
            {currentStepData.messages.map((msg: any, index: number) => (
              <div key={index} className="text-xs bg-gray-50 p-2 rounded">
                <span className="font-medium">{msg.type}</span>: {msg.from} → {msg.to} ({msg.link})
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}