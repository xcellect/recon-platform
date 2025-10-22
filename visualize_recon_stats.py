#!/usr/bin/env python3
"""
ReCoN Network Statistics Visualization Tool

Analyzes and visualizes statistics from ReCoN trace files.

Usage:
    python visualize_recon_stats.py <run_directory>

Example:
    python visualize_recon_stats.py recon_log/recon_arc_angel_20251020T153808Z
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_trace_file(filepath: str) -> Dict[str, Any]:
    """Load a single trace JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def find_all_trace_files(run_dir: str) -> Dict[str, List[str]]:
    """
    Find all trace files in a run directory.
    Returns: {game_level_key: [trace_file_paths]}
    """
    traces = {}

    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        return traces

    # Iterate through games
    for game_dir in os.listdir(run_dir):
        if not game_dir.startswith('game_'):
            continue

        game_id = game_dir.replace('game_', '')
        game_path = os.path.join(run_dir, game_dir)

        if not os.path.isdir(game_path):
            continue

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
                key = f"{game_id}_level_{level}"
                traces[key] = trace_files

    return traces


def analyze_recon_steps(trace_files: List[str]) -> Dict[str, Any]:
    """Analyze recon_steps from trace files."""
    stats = {
        'state_counts_per_step': defaultdict(lambda: defaultdict(int)),
        'activation_per_node': defaultdict(list),
        'message_counts': [],
        'total_steps': 0,
        'total_actions': len(trace_files),
        'node_states_history': defaultdict(list),
        'state_transitions': Counter(),
    }

    all_propagation_steps = []

    for trace_file in trace_files:
        trace_data = load_trace_file(trace_file)
        if not trace_data:
            continue

        recon_steps = trace_data.get('recon_steps', [])

        for step_idx, recon_step in enumerate(recon_steps):
            nodes = recon_step.get('nodes', {})

            # Track state distribution
            for node_id, node_data in nodes.items():
                state = node_data.get('state', 'unknown')
                activation = node_data.get('activation', 0)

                stats['state_counts_per_step'][step_idx][state] += 1
                stats['activation_per_node'][node_id].append(activation)
                stats['node_states_history'][node_id].append(state)

            # Track messages
            messages = recon_step.get('messages', 0)
            if isinstance(messages, int):
                stats['message_counts'].append(messages)
            elif isinstance(messages, list):
                stats['message_counts'].append(len(messages))

            all_propagation_steps.append(step_idx)

    stats['total_steps'] = len(set(all_propagation_steps))

    # Analyze state transitions
    for node_id, state_history in stats['node_states_history'].items():
        for i in range(len(state_history) - 1):
            transition = f"{state_history[i]} → {state_history[i+1]}"
            if state_history[i] != state_history[i+1]:
                stats['state_transitions'][transition] += 1

    return stats


def plot_state_distribution(stats: Dict[str, Any], output_dir: str, game_level: str):
    """Plot state distribution over propagation steps."""
    state_counts = stats['state_counts_per_step']

    if not state_counts:
        print(f"No state data to plot for {game_level}")
        return

    # Get all unique states
    all_states = set()
    for step_data in state_counts.values():
        all_states.update(step_data.keys())

    all_states = sorted(all_states)
    steps = sorted(state_counts.keys())

    # Create data matrix
    data = np.zeros((len(all_states), len(steps)))
    for step_idx, step_num in enumerate(steps):
        for state_idx, state in enumerate(all_states):
            data[state_idx, step_idx] = state_counts[step_num].get(state, 0)

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {
        'inactive': '#95a5a6',
        'requested': '#3498db',
        'waiting': '#f39c12',
        'confirmed': '#2ecc71',
        'failed': '#e74c3c',
    }

    ax.stackplot(steps, *data,
                 labels=all_states,
                 colors=[colors.get(s, '#7f8c8d') for s in all_states],
                 alpha=0.8)

    ax.set_xlabel('Propagation Step', fontsize=12)
    ax.set_ylabel('Node Count', fontsize=12)
    ax.set_title(f'Node State Distribution Over Time\n{game_level}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{game_level}_state_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved state distribution: {output_path}")


def plot_activation_heatmap(stats: Dict[str, Any], output_dir: str, game_level: str):
    """Plot activation heatmap for all nodes."""
    activation_data = stats['activation_per_node']

    if not activation_data:
        print(f"No activation data to plot for {game_level}")
        return

    # Prepare data for heatmap
    node_ids = sorted(activation_data.keys())
    max_steps = max(len(activations) for activations in activation_data.values())

    # Limit to top 30 most active nodes for readability
    node_max_activations = {
        node_id: max(activations) if activations else 0
        for node_id, activations in activation_data.items()
    }
    top_nodes = sorted(node_max_activations.items(), key=lambda x: x[1], reverse=True)[:30]
    top_node_ids = [node_id for node_id, _ in top_nodes]

    # Create matrix
    matrix = np.zeros((len(top_node_ids), max_steps))
    for node_idx, node_id in enumerate(top_node_ids):
        activations = activation_data[node_id]
        for step_idx, activation in enumerate(activations):
            if step_idx < max_steps:
                matrix[node_idx, step_idx] = activation

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 10))

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xlabel('Propagation Step', fontsize=12)
    ax.set_ylabel('Node ID', fontsize=12)
    ax.set_title(f'Node Activation Heatmap (Top 30 Nodes)\n{game_level}', fontsize=14, fontweight='bold')

    # Set y-axis labels
    ax.set_yticks(range(len(top_node_ids)))
    ax.set_yticklabels(top_node_ids, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Level', rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{game_level}_activation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved activation heatmap: {output_path}")


def plot_state_transitions(stats: Dict[str, Any], output_dir: str, game_level: str):
    """Plot state transition counts."""
    transitions = stats['state_transitions']

    if not transitions:
        print(f"No state transitions to plot for {game_level}")
        return

    # Get top 15 transitions
    top_transitions = transitions.most_common(15)

    if not top_transitions:
        return

    labels = [t[0] for t in top_transitions]
    counts = [t[1] for t in top_transitions]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(range(len(labels)), counts, color='steelblue', alpha=0.8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Transition Count', fontsize=12)
    ax.set_title(f'Top State Transitions\n{game_level}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + max(counts) * 0.01, i, str(count),
                va='center', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{game_level}_state_transitions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved state transitions: {output_path}")


def print_summary(all_stats: Dict[str, Dict[str, Any]], run_dir: str):
    """Print summary statistics."""
    print("\n" + "="*80)
    print(f"ReCoN Network Statistics Summary")
    print(f"Run Directory: {run_dir}")
    print("="*80)

    total_actions = sum(s['total_actions'] for s in all_stats.values())
    total_games = len(all_stats)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total games/levels analyzed: {total_games}")
    print(f"  Total actions across all games: {total_actions}")

    for game_level, stats in all_stats.items():
        print(f"\n🎮 {game_level}:")
        print(f"  Actions: {stats['total_actions']}")
        print(f"  Avg propagation steps: {stats['total_steps']}")
        print(f"  Total unique nodes: {len(stats['activation_per_node'])}")
        print(f"  State transitions observed: {len(stats['state_transitions'])}")

        # Show most common final states
        if stats['node_states_history']:
            final_states = Counter()
            for node_id, history in stats['node_states_history'].items():
                if history:
                    final_states[history[-1]] += 1

            print(f"  Final state distribution:")
            for state, count in final_states.most_common():
                print(f"    {state}: {count} nodes")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_recon_stats.py <run_directory>")
        print("Example: python visualize_recon_stats.py recon_log/recon_arc_angel_20251020T153808Z")
        sys.exit(1)

    run_dir = sys.argv[1]

    # Make path absolute if relative
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(os.getcwd(), run_dir)

    print(f"Analyzing ReCoN run: {run_dir}\n")

    # Find all trace files
    trace_files_by_game = find_all_trace_files(run_dir)

    if not trace_files_by_game:
        print("No trace files found!")
        sys.exit(1)

    print(f"Found {len(trace_files_by_game)} game/level combinations\n")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(run_dir), f"visualizations_{os.path.basename(run_dir)}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Analyze each game/level
    all_stats = {}

    for game_level, trace_files in trace_files_by_game.items():
        print(f"Processing {game_level} ({len(trace_files)} action traces)...")

        stats = analyze_recon_steps(trace_files)
        all_stats[game_level] = stats

        # Generate plots
        plot_state_distribution(stats, output_dir, game_level)
        plot_activation_heatmap(stats, output_dir, game_level)
        plot_state_transitions(stats, output_dir, game_level)

        print()

    # Print summary
    print_summary(all_stats, run_dir)

    print(f"\n✅ Visualization complete! Check: {output_dir}\n")


if __name__ == "__main__":
    main()
