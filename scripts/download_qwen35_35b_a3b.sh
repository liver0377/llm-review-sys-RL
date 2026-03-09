#!/bin/bash
set -e

echo "========================================"
echo "Download Qwen3.5-35B-A3B Model"
echo "========================================"
echo ""

cd /data/wudy/RL/llm-review-sys-RL

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Download model
echo "Starting download..."
echo "Model: Qwen/Qwen3.5-35B-A3B"
echo "Destination: models/qwen3.5-35b-a3b"
echo ""

python scripts/download_qwen35_35b_a3b.py

echo ""
echo "========================================"
echo "Download completed!"
echo "Model location: models/qwen3.5-35b-a3b"
echo "========================================"
