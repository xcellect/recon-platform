import React, { useState, useCallback, useEffect } from 'react';
import SimpleNetworkCanvas from './SimpleNetworkCanvas';
import { useNetworkStore } from '../stores/networkStore';
import { reconAPI } from '../services/api';

const ARCScorecardTest: React.FC = () => {
  // For now, use a default URL - this could be made configurable later
  const [replayUrl] = useState('https://three.arcprize.org/replay/ft09-f340c8e5138e/872fc28f-ee09-4b7a-a322-42c6a7eac29f');
  
  // ReCoN network state
  const [executionHistory, setExecutionHistory] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  const [selectedRootNode, setSelectedRootNode] = useState('Root');
  const { currentNetwork, loadNetwork, selectNode, selectLink } = useNetworkStore();

  // ReCoN network handlers
  const handleNodeSelect = useCallback((id: string | null) => {
    const { currentNetwork } = useNetworkStore.getState();
    selectNode(id ? (currentNetwork?.nodes.find(n => n.id === id) || null) : null);
  }, [selectNode]);

  const handleEdgeSelect = useCallback((edgeId: string | null) => {
    if (!edgeId) { 
      selectLink(null); 
      return; 
    }
    const { currentNetwork } = useNetworkStore.getState();
    
    // Handle combined edges (same logic as in App.tsx)
    if (edgeId.endsWith('-sub/sur')) {
      const basePart = edgeId.replace('-sub/sur', '');
      const dashIndex = basePart.indexOf('-');
      if (dashIndex > 0) {
        const source = basePart.substring(0, dashIndex);
        const target = basePart.substring(dashIndex + 1);
        const subLink = currentNetwork?.links.find(l => l.source === source && l.target === target && l.type === 'sub');
        const surLink = currentNetwork?.links.find(l => l.source === target && l.target === source && l.type === 'sur');
        
        if (subLink && surLink) {
          selectLink({
            id: edgeId,
            source: source,
            target: target,
            type: 'sub/sur' as any,
            weight: subLink.weight,
            _subLink: subLink,
            _surLink: surLink,
          } as any);
        }
      }
    } else if (edgeId.endsWith('-por/ret')) {
      const basePart = edgeId.replace('-por/ret', '');
      const dashIndex = basePart.indexOf('-');
      if (dashIndex > 0) {
        const source = basePart.substring(0, dashIndex);
        const target = basePart.substring(dashIndex + 1);
        const porLink = currentNetwork?.links.find(l => l.source === source && l.target === target && l.type === 'por');
        const retLink = currentNetwork?.links.find(l => l.source === target && l.target === source && l.type === 'ret');
        
        if (porLink && retLink) {
          selectLink({
            id: edgeId,
            source: source,
            target: target,
            type: 'por/ret' as any,
            weight: porLink.weight,
            _porLink: porLink,
            _retLink: retLink,
          } as any);
        }
      }
    } else {
      // Single link
      const link = currentNetwork?.links.find(l => `${l.source}-${l.target}-${l.type}` === edgeId) || null;
      selectLink(link as any);
    }
  }, [selectLink]);

  // Control panel handlers
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

  // Load demo network on startup
  useEffect(() => {
    const loadDemoNetwork = async () => {
      try {
        await loadNetwork('demo');
      } catch (error) {
        console.error('Failed to load demo network:', error);
      }
    };
    loadDemoNetwork();
  }, [loadNetwork]);

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
  }, [playing, currentStep, speed, executionHistory.length]);

  return (
    <div className="h-full flex overflow-hidden bg-black">
        {/* Left Side: ReCoN Network */}
        <div className="w-1/2 flex flex-col border-r border-gray-700">
          {/* Network Canvas */}
          <div className="flex-1 bg-gray-900">
            <SimpleNetworkCanvas
              executionHistory={executionHistory}
              currentStep={currentStep}
              onNodeSelect={handleNodeSelect}
              onEdgeSelect={handleEdgeSelect}
            />
          </div>
          
          {/* Minimal Toolbar under ReCoN Network */}
          <div className="bg-gray-800 border-t border-gray-700 p-3">
            <div className="flex items-center justify-between">
              {/* Left: Execution Controls */}
              <div className="flex items-center gap-2">
                {/* Root Selection */}
                <div className="flex items-center gap-2">
                  <label className="text-xs font-medium text-gray-300">Root:</label>
                  <select
                    value={selectedRootNode}
                    onChange={(e) => setSelectedRootNode(e.target.value)}
                    className="px-2 py-1 border border-gray-600 rounded text-xs bg-gray-700 text-white"
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
                  className="px-3 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                >
                  Request
                </button>
              </div>

              {/* Right: Playback Controls */}
              <div className="flex items-center gap-2">
                {executionHistory.length > 0 ? (
                  <>
                    <button
                      onClick={handlePlay}
                      className="px-3 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                    >
                      {playing ? 'Pause' : 'Play'}
                    </button>
                    <button
                      onClick={handleStep}
                      disabled={currentStep >= executionHistory.length - 1}
                      className="px-3 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-700 disabled:bg-gray-500"
                    >
                      Step
                    </button>
                    <button
                      onClick={handleReset}
                      className="px-3 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-700"
                    >
                      Reset
                    </button>
                    <div className="flex items-center gap-1 ml-2">
                      <label className="text-xs text-gray-300">Speed:</label>
                      <input
                        type="range"
                        min="0.1"
                        max="2"
                        step="0.1"
                        value={speed}
                        onChange={(e) => setSpeed(parseFloat(e.target.value))}
                        className="w-12"
                      />
                      <span className="text-xs text-gray-300">{speed}s</span>
                    </div>
                    <div className="ml-2 text-xs text-gray-300">
                      {currentStep} / {executionHistory.length - 1}
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-gray-400">
                    Ready to execute - Select root and click "Request Root"
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Side: ARC-AGI Scorecard */}
        <div className="w-1/2 bg-black">
          <iframe
            src={replayUrl}
            width="100%"
            height="100%"
            frameBorder="0"
            allowFullScreen
            className="w-full h-full"
            title="ARC-AGI Scorecard Replay"
          />
        </div>
    </div>
  );
};

export default ARCScorecardTest;
