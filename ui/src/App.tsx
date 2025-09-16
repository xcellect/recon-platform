import { useState, useEffect, useCallback } from 'react';
import SimpleNetworkCanvas from './components/SimpleNetworkCanvas';
import ControlPanel from './components/ControlPanel';
import Toolbar from './components/Toolbar';
import NodePanel from './components/NodePanel';
import ImportExport from './components/ImportExport';
import { useNetworkStore } from './stores/networkStore';
import { hierarchicalLayout, sequenceLayout, forceDirectedLayout, autoLayout } from './utils/layout';

function App() {
  const { currentNetwork, loadNetwork } = useNetworkStore();
  const [executionHistory, setExecutionHistory] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  // link type is inferred from node handles; no state needed

  const { addNode, selectNode, selectLink, updateNode } = useNetworkStore();

  const handleAddNode = useCallback(async (type: 'script' | 'terminal' | 'hybrid') => {
    // Place new nodes near origin; layout will adjust
    const id = `${type}-${Math.random().toString(36).slice(2, 7)}`;
    await addNode({
      id,
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

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Toolbar */}
      <Toolbar
        onAddNode={handleAddNode}
        onLayoutChange={(layout) => {
          const net = useNetworkStore.getState().currentNetwork;
          if (!net) return;
          let layouted = net.nodes;
          if (layout === 'hierarchical') {
            layouted = hierarchicalLayout(net.nodes, net.links);
          } else if (layout === 'sequence') {
            layouted = sequenceLayout(net.nodes, net.links);
          } else if (layout === 'force') {
            layouted = forceDirectedLayout(net.nodes, net.links);
          } else {
            layouted = autoLayout(net.nodes, net.links);
          }
          layouted.forEach(n => updateNode(n.id, { position: n.position }));
        }}
      />

      {/* Main Area */}
      <div className="flex flex-1 overflow-hidden">
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
    </div>
  );
}

export default App;
