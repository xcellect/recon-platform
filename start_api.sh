#!/bin/bash

# ReCon Platform API Startup Script
# This script sets up the environment and starts the API server

set -e

# Change to the correct directory
cd /workspace/recon-platform

# Set Python path
export PYTHONPATH=/workspace/recon-platform

# Start the API server
exec python3 api/app.py
