// Main App component integrating all UI components

import { useState, useCallback, useEffect } from 'react';
import SimpleNetworkCanvas from './components/SimpleNetworkCanvas';
import Toolbar from './components/Toolbar';
import ControlPanel from './components/ControlPanel';
import NodePanel from './components/NodePanel';
import StateViewer from './components/StateViewer';
import ImportExport from './components/ImportExport';
import { useNetworkStore } from './stores/networkStore';
import type { ReCoNNodeType } from './types/recon';
import { autoLayout, hierarchicalLayout, sequenceLayout, forceDirectedLayout } from './utils/layout';

function App() {
  const { currentNetwork, addNode, updateNode, loadNetwork } = useNetworkStore();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  // Load demo network on startup
  useEffect(() => {
    const loadDemoNetwork = async () => {
      try {
        await loadNetwork('demo');
        // Apply auto layout after loading
        if (currentNetwork?.nodes.length) {
          const layoutedNodes = autoLayout(currentNetwork.nodes, currentNetwork.links);
          layoutedNodes.forEach(node => {
            updateNode(node.id, { position: node.position });
          });
        }
      } catch (error) {
        console.error('Failed to load demo network:', error);
      }
    };

    loadDemoNetwork();
  }, [loadNetwork, updateNode]);

  // Handle adding nodes from toolbar
  const handleAddNode = useCallback((type: ReCoNNodeType) => {
    const nodeId = `${type}_${Date.now()}`;
    const position = {
      x: Math.random() * 400 + 100,
      y: Math.random() * 300 + 100,
    };

    // Generate unique ID and create node
    const newNode = {
      id: nodeId,
      type,
      state: 'inactive' as const,
      activation: 0,
      mode: type === 'hybrid' ? ('explicit' as const) : undefined,
      position,
    };

    // Add to store (temporarily cast to work around type issues)
    addNode(newNode as any);
  }, [addNode]);

  // Handle layout changes
  const handleLayoutChange = useCallback((layout: string) => {
    if (!currentNetwork) return;

    let layoutedNodes;
    switch (layout) {
      case 'hierarchical':
        layoutedNodes = hierarchicalLayout(currentNetwork.nodes, currentNetwork.links);
        break;
      case 'sequence':
        layoutedNodes = sequenceLayout(currentNetwork.nodes, currentNetwork.links);
        break;
      case 'force':
        layoutedNodes = forceDirectedLayout(currentNetwork.nodes, currentNetwork.links);
        break;
      default:
        layoutedNodes = autoLayout(currentNetwork.nodes, currentNetwork.links);
    }

    // Update node positions
    layoutedNodes.forEach(node => {
      updateNode(node.id, { position: node.position });
    });
  }, [currentNetwork, updateNode]);

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="px-6 py-4">
          <h1 className="text-2xl font-bold text-gray-900">
            ReCoN Network Builder
          </h1>
          <p className="text-sm text-gray-600">
            Visual interface for creating and executing Request Confirmation Networks
          </p>
        </div>
      </div>

      {/* Toolbar */}
      <Toolbar onAddNode={handleAddNode} onLayoutChange={handleLayoutChange} />

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-800">Controls</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            <ControlPanel />
            <NodePanel />
            <ImportExport />
          </div>
        </div>

        {/* Main Canvas */}
        <div className="flex-1 relative">
          <SimpleNetworkCanvas
            onNodeSelect={setSelectedNodeId}
            onEdgeSelect={setSelectedEdgeId}
          />

          {/* Canvas Overlay - Instructions */}
          {!currentNetwork && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-50 bg-opacity-80">
              <div className="text-center p-8 bg-white rounded-lg shadow-lg max-w-md">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                  Welcome to ReCoN Builder
                </h3>
                <div className="text-left space-y-2 text-sm text-gray-600">
                  <p>• Create a new network or load an existing one</p>
                  <p>• Double-click the canvas to add nodes</p>
                  <p>• Drag between nodes to create links</p>
                  <p>• Click nodes/links to configure properties</p>
                  <p>• Use the control panel to execute scripts</p>
                </div>
                <div className="mt-6">
                  <button
                    onClick={() => window.open('https://github.com/anthropics/claude-code', '_blank')}
                    className="text-blue-500 hover:text-blue-700 text-sm"
                  >
                    Learn more about ReCoN →
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar */}
        <div className="w-80 bg-white border-l border-gray-200 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-800">State Monitor</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <StateViewer />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-white border-t border-gray-200 px-6 py-2">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div>
            ReCoN Platform UI - Request Confirmation Networks for Neuro-Symbolic Script Execution
          </div>
          <div className="flex items-center gap-4">
            {currentNetwork && (
              <>
                <span>Network: {currentNetwork.id}</span>
                <span>•</span>
                <span>
                  {currentNetwork.nodes.length} nodes, {currentNetwork.links.length} links
                </span>
                <span>•</span>
                <span>Step: {currentNetwork.stepCount}</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
