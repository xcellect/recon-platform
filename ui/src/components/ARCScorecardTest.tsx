import React, { useState, useCallback, useEffect } from 'react';
import SimpleNetworkCanvas from './SimpleNetworkCanvas';
import { reconAPI } from '../services/api';

const ARCScorecardTest: React.FC = () => {
  // ARC replay URLs for the specific games
  const gameUrls = {
    'arcon_as66': 'https://three.arcprize.org/replay/as66-821a4dcad9c2/42783fed-c5e5-4a62-b32f-c200eb8c050f',
    'recon_arc_angel_vc33': 'https://three.arcprize.org/replay/vc33-6ae7bf49eea5/c8b90da1-beee-49f7-a885-c50edc35adab'
  };
  
  // Local network state (independent of shared store)
  const [availableNetworks, setAvailableNetworks] = useState<Array<{id: string, name: string}>>([]);
  const [selectedNetworkId, setSelectedNetworkId] = useState<string>('');
  const [currentNetwork, setCurrentNetwork] = useState<any>(null);
  const [, setSelectedNode] = useState<any>(null);
  const [, setSelectedLink] = useState<any>(null);
  const [replayUrl, setReplayUrl] = useState('https://three.arcprize.org/replay/ft09-f340c8e5138e/872fc28f-ee09-4b7a-a322-42c6a7eac29f');
  const [executionHistory, setExecutionHistory] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  const [selectedRootNode, setSelectedRootNode] = useState('Root');

  // ReCoN network handlers (local state)
  const handleNodeSelect = useCallback((id: string | null) => {
    setSelectedNode(id ? (currentNetwork?.nodes.find((n: any) => n.id === id) || null) : null);
    setSelectedLink(null);
  }, [currentNetwork]);

  const handleEdgeSelect = useCallback((edgeId: string | null) => {
    if (!edgeId) { 
      setSelectedLink(null); 
      return; 
    }
    
    // Handle combined edges (same logic as in App.tsx)
    if (edgeId.endsWith('-sub/sur')) {
      const basePart = edgeId.replace('-sub/sur', '');
      const dashIndex = basePart.indexOf('-');
      if (dashIndex > 0) {
        const source = basePart.substring(0, dashIndex);
        const target = basePart.substring(dashIndex + 1);
        const subLink = currentNetwork?.links.find((l: any) => l.source === source && l.target === target && l.type === 'sub');
        const surLink = currentNetwork?.links.find((l: any) => l.source === target && l.target === source && l.type === 'sur');
        
        if (subLink && surLink) {
          setSelectedLink({
            id: edgeId,
            source: source,
            target: target,
            type: 'sub/sur' as any,
            weight: subLink.weight,
            _subLink: subLink,
            _surLink: surLink,
          });
        }
      }
    } else if (edgeId.endsWith('-por/ret')) {
      const basePart = edgeId.replace('-por/ret', '');
      const dashIndex = basePart.indexOf('-');
      if (dashIndex > 0) {
        const source = basePart.substring(0, dashIndex);
        const target = basePart.substring(dashIndex + 1);
        const porLink = currentNetwork?.links.find((l: any) => l.source === source && l.target === target && l.type === 'por');
        const retLink = currentNetwork?.links.find((l: any) => l.source === target && l.target === source && l.type === 'ret');
        
        if (porLink && retLink) {
          setSelectedLink({
            id: edgeId,
            source: source,
            target: target,
            type: 'por/ret' as any,
            weight: porLink.weight,
            _porLink: porLink,
            _retLink: retLink,
          });
        }
      }
    } else {
      // Single link
      const link = currentNetwork?.links.find((l: any) => `${l.source}-${l.target}-${l.type}` === edgeId) || null;
      setSelectedLink(link);
    }
  }, [currentNetwork]);

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

  const handleNetworkSelect = async (networkId: string) => {
    try {
      setSelectedNetworkId(networkId);
      
      // Load parsed network from logs
      const parsedNetwork = await reconAPI.getParsedNetwork(networkId);
      
      // Convert parsed network to format expected by networkStore
      const network = {
        id: parsedNetwork.network_id,
        nodes: parsedNetwork.nodes.map((node: any) => ({
          id: node.node_id,
          type: node.node_type,
          state: node.state,
          activation: node.activation,
          position: { x: 0, y: 0 } // Reset positions to force fresh layout
        })),
        links: parsedNetwork.links.map((link: any) => ({
          id: `${link.source}-${link.target}-${link.link_type}`,
          source: link.source,
          target: link.target,
          type: link.link_type,
          weight: link.weight
        })),
        stepCount: parsedNetwork.step_count,
        requestedRoots: []
      };
      
      // Debug: Log the network structure
      console.log(`Loading ${networkId}:`, {
        nodes: network.nodes.length,
        links: network.links.length,
        porRetLinks: network.links.filter((l: any) => l.type === 'por' || l.type === 'ret').length,
        linkTypes: [...new Set(network.links.map((l: any) => l.type))]
      });
      
      // Set the network in local state (no shared store interference)
      setCurrentNetwork(network);
      
      // Load execution history from the parsed network
      const executionHistory = await reconAPI.getParsedNetworkExecutionHistory(networkId);
      setExecutionHistory(executionHistory.steps);
      setCurrentStep(0);
      setPlaying(false);
      
      // Set appropriate root node and replay URL
      const rootNode = networkId.startsWith('arcon') ? 'score_increase_hypothesis' : 'frame_change_hypothesis';
      setSelectedRootNode(rootNode);
      
      // Set appropriate replay URL
      const gameUrl = gameUrls[networkId as keyof typeof gameUrls];
      if (gameUrl) {
        setReplayUrl(gameUrl);
      }
    } catch (error) {
      console.error('Failed to load network:', error);
    }
  };

  const handleRequestRoot = async () => {
    if (!currentNetwork || !selectedRootNode) return;

    // For parsed networks, we already have execution history - just reset to beginning
    if (executionHistory.length > 0) {
      setCurrentStep(0);
      setPlaying(false);
      return;
    }
  };

  // Load available parsed networks on startup but don't auto-select
  useEffect(() => {
    const loadAvailableNetworks = async () => {
      try {
        // Load parsed networks from logs
        const parsedNetworks = await reconAPI.listParsedNetworks();
        setAvailableNetworks(parsedNetworks.map(n => ({ id: n.id, name: n.name })));
        
        // Only load first network if no network is currently selected
        if (parsedNetworks.length > 0 && !selectedNetworkId) {
          setSelectedNetworkId(parsedNetworks[0].id);
          await handleNetworkSelect(parsedNetworks[0].id);
        }
      } catch (error) {
        console.error('Failed to load networks:', error);
        setAvailableNetworks([]);
      }
    };
    loadAvailableNetworks();
  }, [selectedNetworkId]);

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
              networkOverride={currentNetwork}
            />
          </div>
          
          {/* Minimal Toolbar under ReCoN Network */}
          <div className="bg-gray-800 border-t border-gray-700 p-3">
            <div className="flex items-center justify-between">
              {/* Left: Network and Execution Controls */}
              <div className="flex items-center gap-2">
                {/* Network Selection */}
                <div className="flex items-center gap-2">
                  <label className="text-xs font-medium text-gray-300">Network:</label>
                  <select
                    value={selectedNetworkId}
                    onChange={(e) => handleNetworkSelect(e.target.value)}
                    className="px-2 py-1 border border-gray-600 rounded text-xs bg-gray-700 text-white"
                  >
                    {availableNetworks.map(network => (
                      <option key={network.id} value={network.id}>
                        {network.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Root Selection */}
                {/* <div className="flex items-center gap-2">
                  <label className="text-xs font-medium text-gray-300">Root:</label>
                  <select
                    value={selectedRootNode}
                    onChange={(e) => setSelectedRootNode(e.target.value)}
                    className="px-2 py-1 border border-gray-600 rounded text-xs bg-gray-700 text-white"
                  >
                    {currentNetwork?.nodes
                      .filter((node: any) => node.type === 'script')
                      .map((node: any) => (
                        <option key={node.id} value={node.id}>
                          {node.id}
                        </option>
                      )) || []}
                  </select>
                </div> */}

                {/* Request Root Button */}
                <button
                  onClick={handleRequestRoot}
                  className="px-3 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                >
                  Replay
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
