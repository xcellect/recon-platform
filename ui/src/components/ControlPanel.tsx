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

  // Auto-play through execution history - still needed for timing
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

  if (!currentNetwork) {
    return (
      <div className="text-gray-400 text-center">
        No network loaded
      </div>
    );
  }

  const currentStepData = executionHistory[currentStep];

  return (
    <div className="flex items-center justify-between text-sm text-gray-300">
      {/* Left: Network Info */}
      <div className="flex items-center gap-6">
        <div>
          <span className="font-medium">Network:</span>
          <span className="ml-2">{currentNetwork.id}</span>
        </div>
        <div>
          <span className="font-medium">Nodes:</span>
          <span className="ml-2">{currentNetwork.nodes.length}</span>
        </div>
        <div>
          <span className="font-medium">Links:</span>
          <span className="ml-2">{currentNetwork.links.length}</span>
        </div>
      </div>

      {/* Right: Execution Info */}
      <div className="flex items-center gap-6">
        <div>
          <span className="font-medium">Messages:</span>
          <span className="ml-2">{currentStepData?.messages?.length || 0}</span>
        </div>
        <div>
          <span className="font-medium">Step:</span>
          <span className="ml-2">{currentStep} / {executionHistory.length > 0 ? executionHistory.length - 1 : 0}</span>
        </div>
      </div>
    </div>
  );
}