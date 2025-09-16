import React, { useState } from 'react';

const ARCScorecardTest: React.FC = () => {
  const [game, setGame] = useState('ft09-b8377d4b7815');
  const [sessionId, setSessionId] = useState('b9d74f37-7a2e-4b3e-9ea9-10175278f56b');
  const [replayUrl, setReplayUrl] = useState('');

  // Sample replay URLs for testing
  const sampleReplayUrls = [
    'https://three.arcprize.org/replay/ft09-f340c8e5138e/872fc28f-ee09-4b7a-a322-42c6a7eac29f',
    'https://three.arcprize.org/replay/ls20-016295f7601e/794795bf-d05f-4bf5-885a-b8a8f37a89fd',
  ];

  const generateReplayUrl = () => {
    if (game && sessionId) {
      return `https://three.arcprize.org/replay/${game}/${sessionId}`;
    }
    return '';
  };

  const handleGenerateUrl = () => {
    const url = generateReplayUrl();
    setReplayUrl(url);
  };

  return (
    <div className="h-full flex flex-col bg-gray-50">
      <div className="bg-white border-b border-gray-200 p-4">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">ARC-AGI Scorecard Replay</h1>
        
        {/* Game and Session ID Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Game ID
            </label>
            <input
              type="text"
              value={game}
              onChange={(e) => setGame(e.target.value)}
              placeholder="e.g., ft09-b8377d4b7815"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Session ID
            </label>
            <input
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="e.g., b9d74f37-7a2e-4b3e-9ea9-10175278f56b"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Generated URL Display */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Generated Replay URL
          </label>
          <div className="flex space-x-2">
            <input
              type="url"
              value={generateReplayUrl()}
              readOnly
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md bg-gray-50 text-gray-600"
            />
            <button
              onClick={handleGenerateUrl}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Load Replay
            </button>
          </div>
        </div>

        {/* Sample URLs */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Sample Replay URLs
          </label>
          <div className="space-y-2">
            {sampleReplayUrls.map((url, index) => (
              <button
                key={index}
                onClick={() => setReplayUrl(url)}
                className="block w-full text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-md text-sm font-mono"
              >
                {url}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 overflow-auto">
        {/* Iframe Display */}
        {(replayUrl || (game && sessionId)) && (
          <div className="border border-gray-300 rounded-lg overflow-hidden">
            <div className="bg-gray-100 px-4 py-2 border-b border-gray-300">
              <h3 className="font-medium text-gray-800">ARC-AGI Scorecard Replay with Interactive Controls</h3>
            </div>
            <iframe
              src={replayUrl || generateReplayUrl()}
              width="100%"
              height="600"
              frameBorder="0"
              allowFullScreen
              className="w-full"
              title="ARC-AGI Scorecard Replay"
            />
          </div>
        )}

        {/* Instructions */}
        {!replayUrl && (!game || !sessionId) && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-medium text-blue-800 mb-2">Instructions</h3>
            <ol className="list-decimal list-inside space-y-1 text-sm text-blue-700">
              <li>Enter the Game ID (e.g., "ft09-b8377d4b7815")</li>
              <li>Enter the Session ID (e.g., "b9d74f37-7a2e-4b3e-9ea9-10175278f56b")</li>
              <li>Click "Load Replay" or use one of the sample URLs</li>
              <li>The embedded replay will show with full interactive controls (play, pause, step, speed control)</li>
            </ol>
          </div>
        )}
      </div>
    </div>
  );
};

export default ARCScorecardTest;
