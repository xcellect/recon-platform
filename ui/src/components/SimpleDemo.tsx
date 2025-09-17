// Simple demo component to test basic functionality
import { useState } from 'react';

export default function SimpleDemo() {
  const [message, setMessage] = useState('ReCoN UI loaded successfully!');

  return (
    <div className="p-8 bg-white rounded-lg shadow-lg max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold text-blue-600 mb-4">ReCoN Network Builder</h1>
      <p className="text-gray-700 mb-6">{message}</p>

      <div className="space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-blue-100 rounded-lg text-center">
            <div className="text-2xl font-bold text-blue-600">8</div>
            <div className="text-sm text-gray-600">Node States</div>
          </div>
          <div className="p-4 bg-green-100 rounded-lg text-center">
            <div className="text-2xl font-bold text-green-600">5</div>
            <div className="text-sm text-gray-600">Link Types</div>
          </div>
          <div className="p-4 bg-purple-100 rounded-lg text-center">
            <div className="text-2xl font-bold text-purple-600">3</div>
            <div className="text-sm text-gray-600">Node Types</div>
          </div>
        </div>

        <div className="border-t pt-4">
          <h3 className="text-lg font-semibold mb-2">Features</h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700">
            <li>Visual network building with React Flow</li>
            <li>Real-time execution and state visualization</li>
            <li>Support for Script, Terminal, and Hybrid nodes</li>
            <li>Request-confirmation message passing</li>
            <li>Import/export in multiple formats</li>
            <li>Auto-layout algorithms</li>
          </ul>
        </div>

        <button
          onClick={() => setMessage('Ready to build ReCoN networks!')}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Get Started
        </button>
      </div>
    </div>
  );
}