import { useState, useEffect } from 'react';
import SimpleNetworkCanvas from './components/SimpleNetworkCanvas';
import ControlPanel from './components/ControlPanel';
import { useNetworkStore } from './stores/networkStore';

function App() {
  const { currentNetwork, loadNetwork } = useNetworkStore();
  const [executionHistory, setExecutionHistory] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.5);

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
      {/* Network Canvas */}
      <div className="h-[70vh] relative">
        <SimpleNetworkCanvas
          executionHistory={executionHistory}
          currentStep={currentStep}
        />
      </div>

      {/* Control Panel */}
      <div className="flex-1 bg-white border-t border-gray-200 p-4">
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
