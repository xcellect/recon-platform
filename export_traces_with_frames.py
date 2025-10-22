#!/usr/bin/env python3
"""
Export ReCoN traces with frame data for visualization.
Combines trace files from recon_log/ into consolidated JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, "/workspace/repo-update/recon-platform")

def load_trace_file(filepath: str) -> Dict[str, Any]:
    """Load a single trace file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_frame_from_snapshot(snapshot: Dict[str, Any]) -> List[List[int]]:
    """Extract frame grid from network snapshot (if available)."""
    # Frame data will be added to traces in the next step
    # For now, return empty grid
    return [[0] * 64 for _ in range(64)]

def find_all_trace_files(recon_log_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Find all trace files organized by game and level.
    Returns: {game_id: {level: [filepath1, filepath2, ...]}}
    """
    traces = {}

    if not os.path.exists(recon_log_dir):
        print(f"ReCoN log directory not found: {recon_log_dir}")
        return traces

    # Iterate through run prefixes
    for run_prefix in os.listdir(recon_log_dir):
        run_path = os.path.join(recon_log_dir, run_prefix)
        if not os.path.isdir(run_path):
            continue

        # Iterate through games
        for game_dir in os.listdir(run_path):
            if not game_dir.startswith('game_'):
                continue

            game_id = game_dir.replace('game_', '')
            game_path = os.path.join(run_path, game_dir)

            if game_id not in traces:
                traces[game_id] = {}

            # Iterate through levels
            for level_dir in os.listdir(game_path):
                if not level_dir.startswith('level_'):
                    continue

                level = level_dir.replace('level_', '')
                level_path = os.path.join(game_path, level_dir)

                # Collect all trace files for this level
                trace_files = []
                for filename in sorted(os.listdir(level_path)):
                    if filename.endswith('.json'):
                        trace_files.append(os.path.join(level_path, filename))

                if trace_files:
                    key = f"{game_id}_{level}"
                    traces[game_id][level] = trace_files

    return traces

def export_level_traces(game_id: str, level: str, trace_files: List[str], output_dir: str):
    """Export all traces for a game level into a single JSON file."""

    steps = []

    for i, filepath in enumerate(sorted(trace_files)):
        trace_data = load_trace_file(filepath)
        if not trace_data:
            continue

        # Extract key information
        meta = trace_data.get('meta', {})
        outcome = trace_data.get('outcome', {})
        recon_steps = trace_data.get('recon_steps', [])
        frame_data = trace_data.get('frame_data', {})
        action_viz = trace_data.get('action_visualization', {})

        # Get the final network state
        final_network_state = recon_steps[-1] if recon_steps else {}

        # Build execution history from all recon_steps for propagation animation
        execution_history = []
        for step_idx, recon_step in enumerate(recon_steps):
            # Extract node states from this propagation step
            nodes = recon_step.get('nodes', {})
            states = {node_id: node_data.get('state', 'inactive') for node_id, node_data in nodes.items()}

            execution_history.append({
                "step": step_idx,
                "states": states,
                "messages": recon_step.get('messages', [])
            })

        step_data = {
            "step": i,
            "meta": {
                "action_count": meta.get('action_count', i),
                "score": meta.get('score'),
                "timestamp": meta.get('timestamp_utc'),
                "available_actions": meta.get('available_actions', [])
            },
            "frame": frame_data.get('grid') or extract_frame_from_snapshot(final_network_state),
            "frame_data": frame_data,  # Include full frame data with objects
            "action_visualization": action_viz,  # Include action visualization
            "outcome": {
                "action": outcome.get('selected_action'),
                "coords": outcome.get('selected_coords'),
                "object_index": outcome.get('selected_object_index')
            },
            "network": final_network_state,
            "execution_history": execution_history  # Add all propagation steps
        }

        steps.append(step_data)

    if not steps:
        return

    # Create output
    output = {
        "game_id": game_id,
        "level": level,
        "total_steps": len(steps),
        "steps": steps
    }

    # Write to file
    output_file = os.path.join(output_dir, f"game_{game_id}_level_{level}.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Exported {len(steps)} steps for game {game_id}, level {level}")

def main():
    # Configuration
    recon_log_dir = "/workspace/repo-update/recon-platform/recon_log"
    output_dir = "/workspace/repo-update/recon-platform/ui/public/trace_data"

    print(f"Scanning for traces in: {recon_log_dir}")

    # Find all traces
    traces = find_all_trace_files(recon_log_dir)

    if not traces:
        print("No trace files found!")
        return

    print(f"Found traces for {len(traces)} game(s)")

    # Export each game/level combination
    total_exported = 0
    for game_id, levels in traces.items():
        for level, trace_files in levels.items():
            export_level_traces(game_id, level, trace_files, output_dir)
            total_exported += 1

    print(f"\n✓ Exported {total_exported} game/level combinations to {output_dir}")

    # Create index file
    index = {
        "games": []
    }

    for game_id, levels in traces.items():
        game_entry = {
            "game_id": game_id,
            "levels": list(levels.keys())
        }
        index["games"].append(game_entry)

    index_file = os.path.join(output_dir, "index.json")
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"✓ Created index file: {index_file}")

if __name__ == "__main__":
    main()
