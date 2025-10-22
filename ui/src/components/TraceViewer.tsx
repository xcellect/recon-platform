/**
 * Trace Viewer Component
 * Visualizes game frames with ReCoN network states
 */

import React, { useState, useEffect, useMemo } from 'react';
import ARCGrid from './ARCGrid';
import SimpleNetworkCanvas from './SimpleNetworkCanvas';

interface TraceMeta {
  action_count: number;
  score: number;
  timestamp: string;
  available_actions: string[];
}

interface TraceOutcome {
  action: string | null;
  coords: [number, number] | null;
  object_index: number | null;
}

interface TraceStep {
  step: number;
  meta: TraceMeta;
  frame: number[][];
  frame_data?: {
    grid: number[][];
    objects: Array<{
      index: number;
      centroid: [number, number];
      area: number;
      regularity: number;
      color: number;
    }>;
  };
  action_visualization?: {
    action_type: string | null;
    click_coords: [number, number] | null;
    object_index: number | null;
    button_pressed: string | null;
  };
  outcome: TraceOutcome;
  network: any;
  execution_history?: Array<{
    step: number;
    states: Record<string, string>;
    messages: any[];
  }>;
}

interface TraceData {
  game_id: string;
  level: string;
  total_steps: number;
  steps: TraceStep[];
}

interface GameIndex {
  games: Array<{
    game_id: string;
    levels: string[];
  }>;
}

export default function TraceViewer() {
  const [index, setIndex] = useState<GameIndex | null>(null);
  const [selectedGame, setSelectedGame] = useState<string>('');
  const [selectedLevel, setSelectedLevel] = useState<string>('');
  const [traceData, setTraceData] = useState<TraceData | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentPropagationStep, setCurrentPropagationStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  // Load index on mount
  useEffect(() => {
    fetch('/trace_data/index.json')
      .then(res => res.json())
      .then(data => {
        setIndex(data);
        // Auto-select first game and level
        if (data.games && data.games.length > 0) {
          const firstGame = data.games[0];
          setSelectedGame(firstGame.game_id);
          if (firstGame.levels && firstGame.levels.length > 0) {
            setSelectedLevel(firstGame.levels[0]);
          }
        }
      })
      .catch(err => {
        setError('Failed to load trace index');
        console.error(err);
      });
  }, []);

  // Load trace data when game/level selected
  useEffect(() => {
    if (!selectedGame || !selectedLevel) return;

    setLoading(true);
    setError('');

    fetch(`/trace_data/game_${selectedGame}_level_${selectedLevel}.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setTraceData(data);
        setCurrentStep(0);
        setLoading(false);
      })
      .catch(err => {
        setError(`Failed to load trace: ${err.message}`);
        setTraceData(null);
        setLoading(false);
        console.error(err);
      });
  }, [selectedGame, selectedLevel]);

  const currentStepData = traceData?.steps[currentStep];
  const availableLevels = index?.games.find(g => g.game_id === selectedGame)?.levels || [];

  // Transform network data from ReCoN format (nodes as dict) to SimpleNetworkCanvas format (nodes as array)
  const transformedNetwork = useMemo(() => {
    if (!currentStepData?.network) return null;

    const network = currentStepData.network;

    // If nodes is already an array, return as-is
    if (Array.isArray(network.nodes)) {
      return network;
    }

    // If nodes is a dictionary, convert to array format
    if (typeof network.nodes === 'object' && network.nodes !== null) {
      const nodesArray = Object.entries(network.nodes).map(([nodeId, nodeData]: [string, any]) => ({
        id: nodeId,
        type: nodeData.type || 'unknown',
        state: nodeData.state || 'inactive',
        position: { x: 0, y: 0 }, // Default position, SimpleNetworkCanvas will apply layout
        ...nodeData, // Include all other node properties
      }));

      return {
        ...network,
        nodes: nodesArray,
      };
    }

    return network;
  }, [currentStepData]);

  // Reset propagation step when action step changes
  useEffect(() => {
    setCurrentPropagationStep(0);
  }, [currentStep]);

  // Get execution history from current step data
  const executionHistory = currentStepData?.execution_history || [];
  const maxPropagationStep = Math.max(0, executionHistory.length - 1);

  const handlePrevStep = () => {
    setCurrentStep(prev => Math.max(0, prev - 1));
  };

  const handleNextStep = () => {
    setCurrentStep(prev => Math.min((traceData?.total_steps || 1) - 1, prev + 1));
  };

  const handlePrevPropagation = () => {
    setCurrentPropagationStep(prev => Math.max(0, prev - 1));
  };

  const handleNextPropagation = () => {
    setCurrentPropagationStep(prev => Math.min(maxPropagationStep, prev + 1));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') handlePrevStep();
    if (e.key === 'ArrowRight') handleNextStep();
    if (e.key === 'ArrowUp') handlePrevPropagation();
    if (e.key === 'ArrowDown') handleNextPropagation();
  };

  return (
    <div className="h-full w-full flex flex-col bg-gray-50" onKeyDown={handleKeyPress} tabIndex={0}>
      {/* Header Controls */}
      <div className="bg-white border-b border-gray-200 p-4 shadow-sm">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Game:</label>
            <select
              value={selectedGame}
              onChange={(e) => {
                setSelectedGame(e.target.value);
                const game = index?.games.find(g => g.game_id === e.target.value);
                if (game && game.levels.length > 0) {
                  setSelectedLevel(game.levels[0]);
                }
              }}
              className="px-3 py-1 border border-gray-300 rounded text-sm"
            >
              {index?.games.map(game => (
                <option key={game.game_id} value={game.game_id}>
                  {game.game_id}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Level:</label>
            <select
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded text-sm"
            >
              {availableLevels.map(level => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          </div>

          {currentStepData && (
            <>
              <div className="ml-auto flex items-center gap-3 text-sm">
                <span className="text-gray-600">
                  Score: <span className="font-bold text-blue-600">{currentStepData.meta.score}</span>
                </span>
                <span className="text-gray-600">
                  Action: <span className="font-mono text-purple-600">{currentStepData.outcome.action || 'none'}</span>
                </span>
                {currentStepData.outcome.coords && (
                  <span className="text-gray-600">
                    Click: <span className="font-mono text-red-600">
                      ({currentStepData.outcome.coords[0]}, {currentStepData.outcome.coords[1]})
                    </span>
                  </span>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {loading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-gray-500">Loading trace data...</div>
          </div>
        )}

        {error && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-red-500">{error}</div>
          </div>
        )}

        {!loading && !error && currentStepData && (
          <>
            {/* Left: ARC Frame */}
            <div className="w-1/2 flex items-center justify-center p-6 bg-gray-100">
              <div className="flex flex-col items-center gap-4">
                <h3 className="text-lg font-semibold text-gray-800">Game Frame</h3>
                <ARCGrid
                  grid={currentStepData.frame_data?.grid || currentStepData.frame}
                  clickCoords={currentStepData.action_visualization?.click_coords || currentStepData.outcome.coords}
                  objects={currentStepData.frame_data?.objects}
                  width={480}
                  height={480}
                />
                <div className="text-xs text-gray-500 font-mono">
                  64×64 Grid | Step {currentStepData.step}
                  {currentStepData.action_visualization?.button_pressed && (
                    <span className="ml-2 text-purple-600">
                      | Button: {currentStepData.action_visualization.button_pressed.toUpperCase()}
                    </span>
                  )}
                </div>
                {currentStepData.frame_data?.objects && currentStepData.frame_data.objects.length > 0 && (
                  <div className="text-xs text-gray-600">
                    {currentStepData.frame_data.objects.length} objects detected
                  </div>
                )}
              </div>
            </div>

            {/* Right: ReCoN Network */}
            <div className="w-1/2 flex flex-col bg-white">
              <div className="p-4 border-b border-gray-200 bg-gray-50">
                <h3 className="text-lg font-semibold text-gray-800">ReCoN Network State</h3>
                <p className="text-sm text-gray-600 mt-1">
                  Action {currentStepData.step} | Propagation step {currentPropagationStep + 1} / {executionHistory.length}
                </p>
              </div>
              <div className="flex-1 relative">
                {transformedNetwork && transformedNetwork.nodes ? (
                  <SimpleNetworkCanvas
                    executionHistory={executionHistory}
                    currentStep={currentPropagationStep}
                    onNodeSelect={() => {}}
                    onEdgeSelect={() => {}}
                    networkOverride={transformedNetwork}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    No network data available
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Bottom: Step Controls */}
      {traceData && (
        <div className="bg-white border-t border-gray-200 p-4 shadow-sm space-y-3">
          {/* Action Step Controls */}
          <div className="flex items-center justify-center gap-4">
            <span className="text-xs font-semibold text-gray-600 uppercase w-32 text-right">Action Step:</span>
            <button
              onClick={handlePrevStep}
              disabled={currentStep === 0}
              className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-600"
            >
              ← Previous
            </button>

            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0"
                max={(traceData.total_steps || 1) - 1}
                value={currentStep}
                onChange={(e) => setCurrentStep(parseInt(e.target.value))}
                className="w-64"
              />
              <span className="text-sm font-medium text-gray-700 min-w-[100px]">
                {currentStep + 1} / {traceData.total_steps}
              </span>
            </div>

            <button
              onClick={handleNextStep}
              disabled={currentStep >= (traceData.total_steps - 1)}
              className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-600"
            >
              Next →
            </button>
          </div>

          {/* Propagation Step Controls */}
          {executionHistory.length > 0 && (
            <div className="flex items-center justify-center gap-4">
              <span className="text-xs font-semibold text-gray-600 uppercase w-32 text-right">Propagation:</span>
              <button
                onClick={handlePrevPropagation}
                disabled={currentPropagationStep === 0}
                className="px-4 py-2 bg-purple-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-purple-600"
              >
                ↑ Previous
              </button>

              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0"
                  max={maxPropagationStep}
                  value={currentPropagationStep}
                  onChange={(e) => setCurrentPropagationStep(parseInt(e.target.value))}
                  className="w-64"
                />
                <span className="text-sm font-medium text-gray-700 min-w-[100px]">
                  {currentPropagationStep + 1} / {executionHistory.length}
                </span>
              </div>

              <button
                onClick={handleNextPropagation}
                disabled={currentPropagationStep >= maxPropagationStep}
                className="px-4 py-2 bg-purple-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-purple-600"
              >
                Next ↓
              </button>
            </div>
          )}

          <div className="text-center text-xs text-gray-500">
            Use arrow keys • Left/Right: Action steps • Up/Down: Propagation steps
          </div>
        </div>
      )}
    </div>
  );
}
