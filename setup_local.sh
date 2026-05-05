#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Logo Generation ML — Local Setup ==="
echo "Project: $SCRIPT_DIR"

# 1. Install PyTorch for CUDA 12.8 (RTX 5080 / Blackwell)
echo ""
echo "Step 1: Installing PyTorch 2.6+ with CUDA 12.8..."
pip install torch>=2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install project dependencies
echo ""
echo "Step 2: Installing project dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# 3. Clone ai-toolkit (needed for LoRA training)
echo ""
echo "Step 3: Setting up ai-toolkit..."
if [ ! -d "$PARENT_DIR/ai-toolkit" ]; then
    git clone https://github.com/ostris/ai-toolkit "$PARENT_DIR/ai-toolkit"
    pip install -r "$PARENT_DIR/ai-toolkit/requirements.txt"
    echo "ai-toolkit installed at $PARENT_DIR/ai-toolkit"
else
    echo "ai-toolkit already exists at $PARENT_DIR/ai-toolkit"
fi

# 4. System dependencies note
echo ""
echo "=== IMPORTANT ==="
echo "cairosvg requires system libcairo2:"
echo "  Ubuntu/Debian: sudo apt install libcairo2"
echo "  Arch:          sudo pacman -S cairo"
echo ""
echo "HuggingFace auth (needed for FLUX.1-dev):"
echo "  huggingface-cli login"
echo ""
echo "=== Setup complete ==="
echo "Run notebooks with: jupyter lab"
echo "ai-toolkit at: $PARENT_DIR/ai-toolkit"
