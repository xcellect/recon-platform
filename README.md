# ReCoN Platform

Request Confirmation Networks for neuro‑symbolic script execution — a faithful, production‑grade implementation of Bach & Herger (2015) with a modern visualization UI and live ARC‑AGI agent demos.

• Paper: “Request Confirmation Networks for Neuro‑Symbolic Script Execution” (Bach & Herger, 2015)

## TL;DR

- Complete ReCoN engine with exact 8‑state semantics and Table 1 message passing
- Dynamic React Flow UI for building and executing ReCoN graphs
- Live demo of ReCoN solving an ARC representation problem via the ARCON agent
 - Live demos of ReCoN solving ARC representation problems via the ARCON and ReCoN ARC Angel agents
- Bonus components: hybrid nodes, neural terminals, continuous messages, rich exporters

## Demo

**🚀 Live Web Demo:**
[https://melodic-chaja-8295bd.netlify.app/](https://melodic-chaja-8295bd.netlify.app/)

**Video Walkthrough:**
[![ReCoN Platform Demo](https://img.youtube.com/vi/S_MzzxUEFcQ/maxresdefault.jpg)](https://youtu.be/S_MzzxUEFcQ?si=lJVZrC9SaRp405EQ)

**Solving ARC on ReCoN:**
<div align="center">
  <img src="demo/demo_recon.gif" alt="ReCoN Platform Demo" width="800"/>
  <br>
  <em>ReCoN Platform in action: Dynamic network visualization with live ARC-AGI task execution, showing hierarchical node states and real-time confirmation propagation. Agent: ReCoN ARC Angel.</em>
</div>

## Features

1) Working implementation + dynamic visualization
- Core engine implements all paper semantics; 132 tests pass (including hypothesis tests to catch randomized edge cases)
- UI provides live execution, state coloring, import/export, and auto‑layout

2) Demonstrate ReCoN solving a representation problem
- ARCON reframes “which object to click” as a ReCoN hypothesis: perceive → select → verify; with `exploration_rate=0`, decisions come from the ReCoN script
- ReCoN ARC Angel uses a CNN terminal to drive `sub/sur` link weights in a hierarchical hypothesis (with `por/ret` sequencing for ACTION6), yielding mask‑aware coordinate selection and background suppression

3) Bonus components
- Hybrid nodes (explicit/implicit/neural), PyTorch neural terminals, continuous message protocol, multi‑format exporters, ARC mappings, production UI

## Why ReCoN

ReCoNs execute hierarchical, sequential scripts distributively via stateful nodes and typed links:
- States: inactive, requested, active, suppressed, waiting, true, confirmed, failed
- Links: sub/sur (hierarchy), por/ret (sequence)
- Terminals confirm based on measurements; confirmations aggregate bottom‑up through `sur`

This implementation matches the paper’s Table 1 semantics and supports the compact arithmetic rules for neural integration.

## Engine Overview (`recon_engine/`)

- `node.py`: 8‑state ReCoN node and message processing
- `graph.py`: Network execution, propagation, execution history, and exporters (React Flow, Cytoscape, D3, Graphviz)
- `messages.py`: Discrete/continuous message handling
- `hybrid_node.py`: Hybrid nodes (explicit/implicit/neural) with mode switching
- `neural_terminal.py`: Wrap PyTorch models as terminals (value, probability, classification, embedding)

Key properties
- Exact link constraints (terminals: only target by `sub`, source `sur`)
- por/ret sequence inhibition and timing; sub/sur hierarchy
- Bottom‑up confirmation scaled by link weights and sender activation
- Execution history and export for visualization

## ARCON Demo (ARC‑AGI tasks)

Paths:
- Agent: `recon_agents/arcon/agent.py`
- State graph + ReCoN hypothesis: `recon_agents/arcon/state_graph.py`
- Harness adapter: `ARC-AGI-3-Agents/agents/arcon_harness.py`

What’s demonstrated
- Hypothesis node `score_increase_hypothesis` with two branches: basic actions and click
- Click branch is a 3‑step script: perceive → select (terminals per object) → verify
- Object terminals confirm using quality measures; link weights bias bottom‑up confirmation

Recommended demo config
- Set `exploration_rate=0.0` in `arcon_harness.py` to ensure decisions are made by the ReCoN script

## ReCoN ARC Angel Demo (ARC‑AGI tasks)

Paths:
- Production agent: `recon_agents/recon_arc_angel/improved_production_agent.py`
- Improved hierarchy + sequence and mask‑aware coupling: `recon_agents/recon_arc_angel/improved_hierarchy_manager.py`
- Learning loop (CNN + optional ResNet value): `recon_agents/recon_arc_angel/learning_manager.py`
- Harness adapter: `ARC-AGI-3-Agents/agents/recon_arc_angel.py`

What’s demonstrated
- CNN terminal (`CNNValidActionTerminal`) produces action and 64×64 coordinate probabilities that become `sub` link weights
- Proper ACTION6 sequence: `action_click → click_cnn → click_objects` with `por/ret`
- Mask‑aware coordinate selection within object masks; background suppression and comprehensive object scoring; stickiness for frame‑change persistence
- Step‑by‑step execution traces and link snapshots exported to `recon_log/`

Runner note
- Use the ARC harness with agent `recon_arc_angel` (see adapter above). The agent returns `GameAction` directly.

## How to run both demos (ARC harness)

Prereq (uv):

```bash
pip install --user uv  # or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

From inside `recon-platform/ARC-AGI-3-Agents`:

```bash
# ReCoN ARC Angel (improved production agent)
export PATH="$HOME/.local/bin:$PATH" && uv run python main.py -a reconarcangel

# ARCON (ReCoN-backed adapter)
export PATH="$HOME/.local/bin:$PATH" && uv run python main.py -a arconrecon
```

Notes:
- If running as root, use `/root/.local/bin` in PATH as you did.
- For ARCON, set `exploration_rate=0.0` in `ARC-AGI-3-Agents/agents/arcon_harness.py` to force pure ReCoN selection.
- ReCoN execution traces are written to `recon_log/` (per game/level step JSON).

## Visualization UI (`ui/`)

- React Flow network canvas with state coloring and live updates
- Node inspectors (script, terminal, hybrid) and control panel for execution
- Import/export JSON; supports engine exporters out of the box

Build and run

```bash
cd ui
npm install
npm run dev   # or: npm run build && npx serve -s dist
```

You can import `demo_export.json` via the UI’s Import panel to browse the pre‑built graphs.

## Quick Start (Engine)

```python
from recon_engine import ReCoNGraph

graph = ReCoNGraph()
graph.add_node("root", "script")
graph.add_node("sensor", "terminal")
graph.add_link("root", "sensor", "sub")

graph.request_root("root")
result = graph.execute_script("root")
print(result)  # 'confirmed' or 'failed'
```

## Install & Tests

```bash
pip install -r requirements.txt
pytest -q
```

Status: **132 tests passing** (state machine, message passing, hierarchy, sequence, hybrid integration, exporters)

## Bonus Components

- Hybrid Node Architecture: explicit ↔ implicit ↔ neural, with state‑preserving mode switches
- Neural Terminals: ResNet/CNN integration, value/probability/classification/embedding outputs
- Enhanced Message Protocol: auto conversion between discrete messages and tensor activations
- Exporters: React Flow, Cytoscape, D3, Graphviz; JSON serialization with auto‑layout helpers

## Roadmap

- Integrate learned affordance directly into object terminals in ARCON
- Graph caching and top‑K object pruning for faster click arbitration
- WebSocket streaming of execution traces to the UI
- Template library and debugging tools (breakpoints, step‑through)

## Citation

Bach, J., & Herger, P. (2015). Request Confirmation Networks for Neuro‑Symbolic Script Execution. Cognitive Architectures Conference.

—

Built for neuro‑symbolic research and practical demos of ReCoN in action.
