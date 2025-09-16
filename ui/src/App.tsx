import { useState, useEffect, useCallback } from 'react';
import SimpleNetworkCanvas from './components/SimpleNetworkCanvas';
import ControlPanel from './components/ControlPanel';
import Toolbar from './components/Toolbar';
import NodePanel from './components/NodePanel';
import ImportExport from './components/ImportExport';
import ARCScorecardTest from './components/ARCScorecardTest';
import { useNetworkStore } from './stores/networkStore';

type TabType = 'recon' | 'scorecard';

function App() {
  const { currentNetwork, loadNetwork } = useNetworkStore();
  const [executionHistory, setExecutionHistory] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  const [activeTab, setActiveTab] = useState<TabType>('recon');
  // link type is inferred from node handles; no state needed

  const { addNode, selectNode, selectLink } = useNetworkStore();

  const handleAddNode = useCallback(async (type: 'script' | 'terminal' | 'hybrid') => {
    // Place new nodes near origin; layout will adjust
    addNode({
      type: type === 'hybrid' ? 'script' : type, // backend supports script|terminal
      state: 'inactive' as any,
      activation: 0,
      position: { x: 0, y: 0 },
    });
  }, [addNode]);

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

  const renderTabContent = () => {
    switch (activeTab) {
      case 'recon':
        return (
          <div className="flex h-full overflow-hidden">
            {/* Left: Canvas */}
            <div className="flex-1 h-full relative">
              <SimpleNetworkCanvas
                executionHistory={executionHistory}
                currentStep={currentStep}
                onNodeSelect={(id) => selectNode(id ? (useNetworkStore.getState().currentNetwork?.nodes.find(n => n.id === id) || null) : null)}
                onEdgeSelect={(edgeId) => {
                  if (!edgeId) { selectLink(null); return; }
                  const net = useNetworkStore.getState().currentNetwork;
                  
                  // For combined edges, create a special selection object that represents both links
                  if (edgeId.endsWith('-sub/sur')) {
                    const basePart = edgeId.replace('-sub/sur', '');
                    const dashIndex = basePart.indexOf('-');
                    if (dashIndex > 0) {
                      const source = basePart.substring(0, dashIndex);
                      const target = basePart.substring(dashIndex + 1);
                      const subLink = net?.links.find(l => l.source === source && l.target === target && l.type === 'sub');
                      const surLink = net?.links.find(l => l.source === target && l.target === source && l.type === 'sur');
                      
                      // Create a combined link object for the panel
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
                      const porLink = net?.links.find(l => l.source === source && l.target === target && l.type === 'por');
                      const retLink = net?.links.find(l => l.source === target && l.target === source && l.type === 'ret');
                      
                      // Create a combined link object for the panel
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
                    // Single link (gen, etc.)
                    const link = net?.links.find(l => `${l.source}-${l.target}-${l.type}` === edgeId) || null;
                    selectLink(link as any);
                  }
                }}
              />
            </div>
            {/* Right: Panels */}
            <div className="w-96 bg-white border-l border-gray-200 p-4 overflow-auto">
              <NodePanel />
              <div className="mt-4">
                <ImportExport />
              </div>
            </div>
          </div>
        );
      case 'scorecard':
        return <ARCScorecardTest />;
      default:
        return null;
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200">
        <div className="flex">
          <button
            onClick={() => setActiveTab('recon')}
            className={`px-6 py-3 text-sm font-medium border-b-2 ${
              activeTab === 'recon'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            ReCoN Networks
          </button>
          <button
            onClick={() => setActiveTab('scorecard')}
            className={`px-6 py-3 text-sm font-medium border-b-2 ${
              activeTab === 'scorecard'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            ARC-AGI Scorecard Test
          </button>
        </div>
      </div>

      {/* Conditional Toolbar - only show for ReCoN tab */}
      {activeTab === 'recon' && (
        <Toolbar onAddNode={handleAddNode} />
      )}

      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden">
        {renderTabContent()}
      </div>

      {/* Conditional Bottom Panels - only show for ReCoN tab */}
      {activeTab === 'recon' && (
        <>
          {/* Bottom: Control Panel */}
          <div className="bg-white border-t border-gray-200 p-4">
            <ControlPanel
              executionHistory={executionHistory}
              setExecutionHistory={setExecutionHistory}
              currentStep={currentStep}
              setCurrentStep={setCurrentStep}
              playing={playing}
              setPlaying={setPlaying}
              speed={speed}
              setSpeed={setSpeed}
            />
          </div>

          {/* Status Bar */}
          <div className="bg-gray-100 border-t border-gray-200 px-4 py-2 text-sm text-gray-600">
            {currentNetwork && (
              <div className="flex items-center justify-between">
                <span>Network: {currentNetwork.id}</span>
                <span>Step: {currentStep} / {executionHistory.length > 0 ? executionHistory.length - 1 : 0}</span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default App;
