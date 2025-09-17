#!/usr/bin/env python3
"""
Multi-level log parser to convert ReCoN trace logs to frontend-compatible network format.
Processes multiple levels for each game to create comprehensive execution histories.
"""

import json
import os
import glob
from typing import Dict, List, Any

def find_latest_log_in_level(level_dir: str) -> str:
    """Find the latest log file in a level directory."""
    pattern = os.path.join(level_dir, "*recon_trace_step_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No log files found in {level_dir}")
    
    # Sort by filename (timestamp) and get the latest
    files.sort()
    return files[-1]

def parse_arcon_level(log_path: str) -> Dict[str, Any]:
    """Parse a single arcon level log."""
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    meta = data['meta']
    history = data['history']
    
    return {
        'level': meta['score'],
        'action_count': meta['action_count'],
        'timestamp': meta['timestamp_utc'],
        'execution_steps': history['steps'],
        'meta': meta
    }

def parse_recon_arc_angel_level(log_path: str) -> Dict[str, Any]:
    """Parse a single recon_arc_angel level log."""
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    meta = data['meta']
    recon_steps = data['recon_steps']
    
    # Convert recon_steps to execution steps format
    execution_steps = []
    for i, step_data in enumerate(recon_steps):
        # Convert node snapshots to states format
        states = {}
        if "nodes" in step_data:
            for node_id, node_info in step_data["nodes"].items():
                states[node_id] = node_info.get("state", "inactive")
        
        execution_steps.append({
            "step": i,
            "states": states,
            "messages": []  # ReCoN Arc Angel logs don't have message data
        })
    
    return {
        'level': meta['score'],
        'action_count': meta['action_count'],
        'timestamp': meta['timestamp_utc'],
        'execution_steps': execution_steps,
        'meta': meta
    }

def parse_arcon_multi_level(game_dir: str, levels: List[int]) -> Dict[str, Any]:
    """Parse multiple levels for arcon game."""
    level_data = []
    
    # Get network structure from level 0
    level_0_dir = os.path.join(game_dir, "level_0")
    level_0_log = find_latest_log_in_level(level_0_dir)
    
    with open(level_0_log, 'r') as f:
        base_data = json.load(f)
    
    base_meta = base_data['meta']
    base_history = base_data['history']
    
    # Extract network structure from level 0
    final_step = base_history['steps'][-1] if base_history['steps'] else base_history['steps'][0]
    states = final_step['states']
    
    # Create nodes from states
    nodes = []
    for node_id, state in states.items():
        node_type = 'terminal' if any(x in node_id for x in ['terminal', 'detector', 'ready']) else 'script'
        nodes.append({
            'node_id': node_id,
            'node_type': node_type,
            'state': state,
            'activation': 1.0 if state == 'confirmed' else 0.5 if state == 'true' else 0.0
        })
    
    # Extract links from execution history messages
    all_messages = []
    for step in base_history['steps']:
        all_messages.extend(step.get('messages', []))
    
    links = []
    link_set = set()
    
    for message in all_messages:
        if message.get('type') in ['request', 'inhibit_request', 'inhibit_confirm', 'confirm']:
            link_type = message.get('link', 'sub')
            source = message.get('from', '')
            target = message.get('to', '')
            
            if source and target:
                link_key = (source, target, link_type)
                if link_key not in link_set:
                    links.append({
                        'source': source,
                        'target': target,
                        'link_type': link_type,
                        'weight': 1.0
                    })
                    link_set.add(link_key)
    
    # Parse each level
    for level in levels:
        level_dir = os.path.join(game_dir, f"level_{level}")
        if os.path.exists(level_dir):
            try:
                level_log = find_latest_log_in_level(level_dir)
                level_info = parse_arcon_level(level_log)
                level_data.append(level_info)
                print(f"Parsed arcon level {level}: {len(level_info['execution_steps'])} steps")
            except Exception as e:
                print(f"Failed to parse arcon level {level}: {e}")
    
    return {
        'network_id': f"arcon_{base_meta['game_id']}",
        'nodes': nodes,
        'links': links,
        'levels': level_data,
        'total_levels': len(level_data),
        'meta': base_meta
    }

def parse_recon_arc_angel_multi_level(game_dir: str, levels: List[int]) -> Dict[str, Any]:
    """Parse multiple levels for recon_arc_angel game."""
    level_data = []
    
    # Get network structure from level 0
    level_0_dir = os.path.join(game_dir, "level_0")
    level_0_log = find_latest_log_in_level(level_0_dir)
    
    with open(level_0_log, 'r') as f:
        base_data = json.load(f)
    
    base_meta = base_data['meta']
    base_recon_steps = base_data['recon_steps']
    
    # Extract network structure from level 0
    final_step = base_recon_steps[-1] if base_recon_steps else base_recon_steps[0]
    nodes_data = final_step['nodes']
    links_data = final_step.get('links', [])
    
    # Extract nodes
    nodes = []
    for node_id, node_info in nodes_data.items():
        nodes.append({
            'node_id': node_id,
            'node_type': node_info['type'],
            'state': node_info['state'],
            'activation': node_info['activation']
        })
    
    # Extract links
    links = []
    link_set = set()
    
    for link_info in links_data:
        source = link_info['source']
        target = link_info['target']
        link_type = link_info['type']
        weight = link_info['weight']
        
        link_key = (source, target, link_type)
        if link_key not in link_set:
            links.append({
                'source': source,
                'target': target,
                'link_type': link_type,
                'weight': weight
            })
            link_set.add(link_key)
    
    # Parse each level
    for level in levels:
        level_dir = os.path.join(game_dir, f"level_{level}")
        if os.path.exists(level_dir):
            try:
                level_log = find_latest_log_in_level(level_dir)
                level_info = parse_recon_arc_angel_level(level_log)
                level_data.append(level_info)
                print(f"Parsed recon_arc_angel level {level}: {len(level_info['execution_steps'])} steps")
            except Exception as e:
                print(f"Failed to parse recon_arc_angel level {level}: {e}")
    
    return {
        'network_id': f"recon_arc_angel_{base_meta['game_id']}",
        'nodes': nodes,
        'links': links,
        'levels': level_data,
        'total_levels': len(level_data),
        'meta': base_meta
    }

def main():
    """Parse multi-level logs for both games."""
    
    # Paths and level ranges
    arcon_game_dir = "/workspace/recon-platform/recon_log/arcon_20250917T192437Z/game_as66-821a4dcad9c2"
    recon_arc_angel_game_dir = "/workspace/recon-platform/recon_log/recon_arc_angel_20250917T193110Z/game_vc33-6ae7bf49eea5"
    
    arcon_levels = [0, 1, 2, 3]  # levels 0-3 for arcon as66
    recon_arc_angel_levels = [0, 1, 2]  # levels 0-2 for recon_arc_angel vc33
    
    # Output directory
    output_dir = "/workspace/recon-platform/ui/parsed_networks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse arcon multi-level
    if os.path.exists(arcon_game_dir):
        print(f"Parsing arcon multi-level: {arcon_game_dir}")
        try:
            arcon_network = parse_arcon_multi_level(arcon_game_dir, arcon_levels)
            
            with open(f"{output_dir}/arcon_as66_multilevel.json", 'w') as f:
                json.dump(arcon_network, f, indent=2)
            print(f"Saved arcon multi-level network: {output_dir}/arcon_as66_multilevel.json")
            print(f"Total levels: {arcon_network['total_levels']}")
        except Exception as e:
            print(f"Failed to parse arcon multi-level: {e}")
    else:
        print(f"Arcon game directory not found: {arcon_game_dir}")
    
    # Parse recon_arc_angel multi-level
    if os.path.exists(recon_arc_angel_game_dir):
        print(f"Parsing recon_arc_angel multi-level: {recon_arc_angel_game_dir}")
        try:
            recon_network = parse_recon_arc_angel_multi_level(recon_arc_angel_game_dir, recon_arc_angel_levels)
            
            with open(f"{output_dir}/recon_arc_angel_vc33_multilevel.json", 'w') as f:
                json.dump(recon_network, f, indent=2)
            print(f"Saved recon_arc_angel multi-level network: {output_dir}/recon_arc_angel_vc33_multilevel.json")
            print(f"Total levels: {recon_network['total_levels']}")
        except Exception as e:
            print(f"Failed to parse recon_arc_angel multi-level: {e}")
    else:
        print(f"ReCoN Arc Angel game directory not found: {recon_arc_angel_game_dir}")

if __name__ == "__main__":
    main()
