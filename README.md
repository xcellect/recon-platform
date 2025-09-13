# ReCoN Platform MVP

A fresh implementation of Request Confirmation Networks (ReCoNs) as described in "Request Confirmation Networks for Neuro-Symbolic Script Execution" (Bach & Herger, 2015).

## Overview

This implementation addresses the theoretical framework from the paper while building a modern, testable platform for user-created networks solving ARC-like puzzles.

## Key Features

- **Rigorous Testing**: 100% compliance with paper's theoretical specification
- **Modern Architecture**: FastAPI backend with type hints and clean separation
- **Subsymbolic Ready**: PyTorch integration for neural components
- **No Dependencies on Legacy Code**: Fresh implementation avoiding MicroPsi2 limitations

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -v
```

## Phase 1: Core ReCoN Engine

This phase focuses on:
1. State machine implementation (8 states: inactive, requested, active, suppressed, waiting, true, confirmed, failed)
2. Message passing semantics (inhibit_request, inhibit_confirm, wait, confirm, fail)
3. Link types (por/ret for sequences, sub/sur for hierarchies)
4. Compact implementation with arithmetic rules from section 3.1