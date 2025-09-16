// Import/Export functionality for ReCoN networks

import React, { useState, useRef } from 'react';
import { useNetworkStore } from '../stores/networkStore';
import { reconAPI } from '../services/api';

export default function ImportExport() {
  const { currentNetwork, loadNetwork, exportLocalGraph } = useNetworkStore();
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  // exportFormat removed - all exports are client-side now
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = async () => {
    if (!currentNetwork) return;

    try {
      setIsExporting(true);
      // Use local graph state instead of server state
      const data = exportLocalGraph();
      if (!data) {
        throw new Error('No network data to export');
      }

      // Create and download file
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentNetwork.id}_local.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please check the console for details.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsImporting(true);
      const text = await file.text();
      const data = JSON.parse(text);

      // Import the network
      const result = await reconAPI.importNetwork(data);

      // Load the imported network
      await loadNetwork(result.network_id);

      alert(`Network imported successfully with ID: ${result.network_id}`);
    } catch (error) {
      console.error('Import failed:', error);
      alert('Import failed. Please check the file format and try again.');
    } finally {
      setIsImporting(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleExportReactFlow = async () => {
    if (!currentNetwork) return;

    try {
      setIsExporting(true);
      const localData = exportLocalGraph();
      if (!localData) {
        throw new Error('No network data to export');
      }

      // Convert to React Flow format
      const reactFlowData = {
        nodes: localData.nodes.map((node: any) => ({
          id: node.id,
          type: node.type,
          position: { x: 0, y: 0 }, // Would need layout
          data: {
            label: node.id,
            nodeData: {
              id: node.id,
              type: node.type,
              state: node.state,
              activation: node.activation,
            },
          },
        })),
        edges: localData.links.map((link: any) => ({
          id: `${link.source}-${link.target}-${link.type}`,
          source: link.source,
          target: link.target,
          type: 'custom',
          data: {
            linkType: link.type,
            weight: link.weight,
          },
        })),
      };

      const blob = new Blob([JSON.stringify(reactFlowData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentNetwork.id}_reactflow.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('React Flow export failed:', error);
      alert('React Flow export failed. Please check the console for details.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportCytoscape = async () => {
    if (!currentNetwork) return;

    try {
      setIsExporting(true);
      const localData = exportLocalGraph();
      if (!localData) {
        throw new Error('No network data to export');
      }

      // Convert to Cytoscape format
      const cytoscapeData = {
        elements: [
          ...localData.nodes.map((node: any) => ({
            data: {
              id: node.id,
              label: node.id,
              type: node.type,
              state: node.state,
              activation: node.activation,
            },
          })),
          ...localData.links.map((link: any) => ({
            data: {
              id: `${link.source}-${link.target}`,
              source: link.source,
              target: link.target,
              label: link.type,
              type: link.type,
              weight: link.weight,
            },
          })),
        ],
      };

      const blob = new Blob([JSON.stringify(cytoscapeData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentNetwork.id}_cytoscape.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Cytoscape export failed:', error);
      alert('Cytoscape export failed. Please check the console for details.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleShareNetwork = async () => {
    if (!currentNetwork) return;

    try {
      const data = exportLocalGraph();
      if (!data) {
        throw new Error('No network data to share');
      }
      const jsonString = JSON.stringify(data);

      // Copy to clipboard
      await navigator.clipboard.writeText(jsonString);
      alert('Network data copied to clipboard!');
    } catch (error) {
      console.error('Share failed:', error);
      alert('Failed to copy network data to clipboard.');
    }
  };

  if (!currentNetwork) {
    return (
      <div className="bg-gray-100 p-4 rounded-lg">
        <div className="text-gray-500 text-center">
          No network loaded
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Import/Export</h3>

      {/* Export Section */}
      <div className="space-y-3">
        <h4 className="text-md font-medium text-gray-700">Export Network</h4>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-sm"
          >
            {isExporting ? 'Exporting...' : 'Export JSON'}
          </button>

          <button
            onClick={handleExportReactFlow}
            disabled={isExporting}
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-sm"
          >
            Export React Flow
          </button>

          <button
            onClick={handleExportCytoscape}
            disabled={isExporting}
            className="px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-sm"
          >
            Export Cytoscape
          </button>

          <button
            onClick={handleShareNetwork}
            className="px-4 py-2 bg-orange-500 text-white rounded-md hover:bg-orange-600 text-sm"
          >
            Copy to Clipboard
          </button>
        </div>

        {/* Format selection removed - all exports are client-side JSON now */}
      </div>

      {/* Import Section */}
      <div className="space-y-3 pt-4 border-t border-gray-200">
        <h4 className="text-md font-medium text-gray-700">Import Network</h4>

        <div className="flex gap-3">
          <button
            onClick={handleImportClick}
            disabled={isImporting}
            className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-sm"
          >
            {isImporting ? 'Importing...' : 'Import from File'}
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>

        <div className="text-xs text-gray-500">
          Supported formats: JSON exported from ReCoN Platform
        </div>
      </div>

      {/* Network Info */}
      <div className="pt-4 border-t border-gray-200">
        <h4 className="text-md font-medium text-gray-700 mb-2">Current Network</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <div>ID: <span className="font-mono">{currentNetwork.id}</span></div>
          <div>Nodes: <span className="font-medium">{currentNetwork.nodes.length}</span></div>
          <div>Links: <span className="font-medium">{currentNetwork.links.length}</span></div>
          <div>Steps: <span className="font-medium">{currentNetwork.stepCount}</span></div>
        </div>
      </div>
    </div>
  );
}