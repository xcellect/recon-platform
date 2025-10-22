/**
 * ARC Grid Renderer
 * Displays a 64x64 ARC grid with object overlays and click visualization
 */

import React, { useEffect, useRef } from 'react';

interface ARCGridProps {
  grid: number[][];  // 64x64 grid of color values (0-15)
  clickCoords?: [number, number] | null;  // [x, y] coordinates of click
  objects?: Array<{centroid: [number, number]; mask?: any}>;  // Object centroids
  width?: number;
  height?: number;
}

// ARC color palette (standard 16 colors)
const ARC_COLORS = [
  '#000000',  // 0: black
  '#0074D9',  // 1: blue
  '#FF4136',  // 2: red
  '#2ECC40',  // 3: green
  '#FFDC00',  // 4: yellow
  '#AAAAAA',  // 5: grey
  '#F012BE',  // 6: magenta
  '#FF851B',  // 7: orange
  '#7FDBFF',  // 8: sky blue
  '#870C25',  // 9: brown
  '#1ABC9C',  // 10: teal
  '#FF1493',  // 11: pink
  '#00FF00',  // 12: lime
  '#FFD700',  // 13: gold
  '#8B4513',  // 14: saddle brown
  '#FFFFFF',  // 15: white
];

export default function ARCGrid({
  grid,
  clickCoords,
  objects = [],
  width = 512,
  height = 512
}: ARCGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !grid || grid.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = grid.length;
    const cols = grid[0]?.length || 0;

    if (rows === 0 || cols === 0) return;

    const cellWidth = width / cols;
    const cellHeight = height / rows;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw grid
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const colorValue = grid[y][x];
        ctx.fillStyle = ARC_COLORS[colorValue] || ARC_COLORS[0];
        ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
      }
    }

    // Draw grid lines (subtle)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 0.5;

    for (let x = 0; x <= cols; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cellWidth, 0);
      ctx.lineTo(x * cellWidth, height);
      ctx.stroke();
    }

    for (let y = 0; y <= rows; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * cellHeight);
      ctx.lineTo(width, y * cellHeight);
      ctx.stroke();
    }

    // Draw object centroids
    if (objects && objects.length > 0) {
      objects.forEach((obj, idx) => {
        if (obj.centroid && obj.centroid.length === 2) {
          const [cx, cy] = obj.centroid;

          // Draw circle at centroid
          ctx.fillStyle = 'rgba(255, 255, 0, 0.5)';
          ctx.strokeStyle = 'rgba(255, 255, 0, 1)';
          ctx.lineWidth = 2;

          ctx.beginPath();
          ctx.arc(
            cx * cellWidth + cellWidth / 2,
            cy * cellHeight + cellHeight / 2,
            cellWidth * 1.5,
            0,
            2 * Math.PI
          );
          ctx.fill();
          ctx.stroke();

          // Draw object index
          ctx.fillStyle = '#000';
          ctx.font = `${cellWidth * 0.8}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            String(idx),
            cx * cellWidth + cellWidth / 2,
            cy * cellHeight + cellHeight / 2
          );
        }
      });
    }

    // Draw click marker
    if (clickCoords && clickCoords.length === 2) {
      const [clickX, clickY] = clickCoords;

      // Draw red X
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 3;

      const centerX = clickX * cellWidth + cellWidth / 2;
      const centerY = clickY * cellHeight + cellHeight / 2;
      const size = cellWidth * 0.8;

      // X mark
      ctx.beginPath();
      ctx.moveTo(centerX - size, centerY - size);
      ctx.lineTo(centerX + size, centerY + size);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(centerX + size, centerY - size);
      ctx.lineTo(centerX - size, centerY + size);
      ctx.stroke();

      // Circle around X
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, cellWidth * 1.2, 0, 2 * Math.PI);
      ctx.stroke();
    }

  }, [grid, clickCoords, objects, width, height]);

  if (!grid || grid.length === 0) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center bg-gray-100 rounded"
      >
        <span className="text-gray-500">No frame data</span>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border-2 border-gray-300 rounded"
      style={{ imageRendering: 'pixelated' }}
    />
  );
}
