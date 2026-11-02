#!/bin/bash
# Activate the project environment for DFloat11-TT
# Usage: source setup_env.sh

export TT_METAL_HOME="$HOME/tt-metal"
export PYTHONPATH="$TT_METAL_HOME:$PYTHONPATH"

# Use the tt-metal Python virtual environment
source "$TT_METAL_HOME/python_env/bin/activate"

# Add our project src to PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Set working directory to project root
cd "$SCRIPT_DIR"

echo "=== DFloat11-TT Environment ==="
echo "  Python:      $(which python3)"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo "  Project dir: $SCRIPT_DIR"
echo "  PYTHONPATH includes: src/"
echo "==============================="
