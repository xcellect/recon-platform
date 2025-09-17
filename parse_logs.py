#!/usr/bin/env python3
"""
Simple log parser to convert ReCoN trace logs to frontend-compatible network format.
Processes the two specific game logs: arcon (as66) and recon_arc_angel (vc33).
"""

import json
import os
from typing import Dict, List, Any

def parse_arcon_log(log_path: str) -> Dict[str, Any]:
    """Parse arcon ReCoN log to network format."""
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    meta = data['meta']
    history = data['history']
    
    # Get the final step to extract network structure
    final_step = history['steps'][-1] if history['steps'] else history['steps'][0]
    states = final_step['states']
    
    # Extract nodes from states
    nodes = []
    links = []
    
    # Create nodes from states
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
    for step in history['steps']:
        all_messages.extend(step.get('messages', []))
    
    # Create links based on actual message patterns in the log
    link_set = set()  # To avoid duplicates
    
    for message in all_messages:
        # Extract links from all message types (request, inhibit_request, etc.)
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
    
    # If no messages found, fall back to reconstructing from known arcon structure
    if not links:
        # Root -> basic_actions and click_hypothesis
        if 'score_increase_hypothesis' in states and 'basic_action_branch' in states:
            links.append({
                'source': 'score_increase_hypothesis',
                'target': 'basic_action_branch', 
                'link_type': 'sub',
                'weight': 1.0
            })
        
        if 'score_increase_hypothesis' in states and 'click_hypothesis' in states:
            links.append({
                'source': 'score_increase_hypothesis',
                'target': 'click_hypothesis',
                'link_type': 'sub', 
                'weight': 1.0
            })
        
        # Basic action sequence with por connections
        for i in range(1, 6):
            action_id = f'action_{i}'
            terminal_id = f'action_{i}_terminal'
            if action_id in states:
                if 'basic_action_branch' in states:
                    links.append({
                        'source': 'basic_action_branch',
                        'target': action_id,
                        'link_type': 'sub',
                        'weight': 1.0
                    })
                if terminal_id in states:
                    links.append({
                        'source': action_id,
                        'target': terminal_id,
                        'link_type': 'sub',
                        'weight': 1.0
                    })
                # Sequential por connections: action_1 -> action_2 -> ... -> action_5
                if i > 1:
                    prev_action = f'action_{i-1}'
                    if prev_action in states:
                        links.append({
                            'source': prev_action,
                            'target': action_id,
                            'link_type': 'por',
                            'weight': 1.0
                        })
        
        # Click hypothesis structure
        for node in ['perceive_objects', 'select_object', 'verify_change']:
            if node in states and 'click_hypothesis' in states:
                links.append({
                    'source': 'click_hypothesis',
                    'target': node,
                    'link_type': 'sub',
                    'weight': 1.0
                })
        
        # Sequential por connections in click hypothesis: perceive -> select -> verify
        if 'perceive_objects' in states and 'select_object' in states:
            links.append({
                'source': 'perceive_objects',
                'target': 'select_object',
                'link_type': 'por',
                'weight': 1.0
            })
        
        if 'select_object' in states and 'verify_change' in states:
            links.append({
                'source': 'select_object',
                'target': 'verify_change',
                'link_type': 'por',
                'weight': 1.0
            })
        
        # Object terminals under select_object
        for node_id in states.keys():
            if node_id.startswith('object_') and 'select_object' in states:
                links.append({
                    'source': 'select_object',
                    'target': node_id,
                    'link_type': 'sub',
                    'weight': 0.8
                })
        
        # Terminals
        if 'perception_ready' in states and 'perceive_objects' in states:
            links.append({
                'source': 'perceive_objects',
                'target': 'perception_ready',
                'link_type': 'sub',
                'weight': 1.0
            })
        
        if 'change_detector' in states and 'verify_change' in states:
            links.append({
                'source': 'verify_change', 
                'target': 'change_detector',
                'link_type': 'sub',
                'weight': 1.0
            })
    
    return {
        'network_id': f"arcon_{meta['game_id']}",
        'nodes': nodes,
        'links': links,
        'step_count': len(history['steps']),
        'execution_history': history['steps'],
        'meta': meta
    }

def parse_recon_arc_angel_log(log_path: str) -> Dict[str, Any]:
    """Parse recon_arc_angel ReCoN log to network format."""
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    meta = data['meta']
    recon_steps = data['recon_steps']
    
    # Get the final step to extract network structure
    final_step = recon_steps[-1] if recon_steps else recon_steps[0]
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
    
    # Extract links directly from the log data (includes por/ret/sub/sur)
    links = []
    link_set = set()  # To avoid duplicates
    
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
    
    return {
        'network_id': f"recon_arc_angel_{meta['game_id']}", 
        'nodes': nodes,
        'links': links,
        'step_count': len(recon_steps),
        'execution_history': recon_steps,
        'meta': meta
    }

def main():
    """Parse the two specific log files and save them as JSON."""
    
    # Paths to the specific log files
    arcon_log = "/workspace/recon-platform/recon_log/arcon_20250917T192437Z/game_as66-821a4dcad9c2/level_0/bs_recon_trace_step_00010_20250917T192445815911Z.json"
    recon_arc_angel_log = "/workspace/recon-platform/recon_log/recon_arc_angel_20250917T193110Z/game_vc33-6ae7bf49eea5/level_0/recon_trace_step_00002_20250917T193114088940Z.json"
    
    # Output directory
    output_dir = "/workspace/recon-platform/ui/parsed_networks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse arcon log
    if os.path.exists(arcon_log):
        print(f"Parsing arcon log: {arcon_log}")
        arcon_network = parse_arcon_log(arcon_log)
        
        with open(f"{output_dir}/arcon_as66.json", 'w') as f:
            json.dump(arcon_network, f, indent=2)
        print(f"Saved arcon network: {output_dir}/arcon_as66.json")
    else:
        print(f"Arcon log not found: {arcon_log}")
    
    # Parse recon_arc_angel log
    if os.path.exists(recon_arc_angel_log):
        print(f"Parsing recon_arc_angel log: {recon_arc_angel_log}")
        recon_network = parse_recon_arc_angel_log(recon_arc_angel_log)
        
        with open(f"{output_dir}/recon_arc_angel_vc33.json", 'w') as f:
            json.dump(recon_network, f, indent=2)
        print(f"Saved recon_arc_angel network: {output_dir}/recon_arc_angel_vc33.json")
    else:
        print(f"ReCoN ARC Angel log not found: {recon_arc_angel_log}")

if __name__ == "__main__":
    main()
