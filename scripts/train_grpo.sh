#!/bin/bash

set -e

echo "========================================"
echo "GRPO Training with veRL Framework"
echo "Paper Review Generation"
echo "========================================"

cd /data/wudw/llm-review-sys-RL

REWARD_MODEL_PORT=8002

echo ""
echo "Configuration:"
echo "  Policy Model:     models/Qwen3-8B-Base"
echo "  Reward Model:     External Service (port $REWARD_MODEL_PORT)"
echo "  Training GPUs:    0,1,2,3"
echo "  Framework:        veRL 0.8.0"
echo "  Algorithm:        GRPO"
echo "  Output:           models/grpo_qwen3_8b_verl"
echo ""

echo "[Step 1] Checking RM Service..."
echo "----------------------------------------"

if ! curl -s http://127.0.0.1:$REWARD_MODEL_PORT/health > /dev/null 2>&1; then
    echo "❌ Error: RM Service is not running on port $REWARD_MODEL_PORT"
    echo ""
    echo "Please start the RM service first:"
    echo "  bash scripts/start_rm_service.sh"
    exit 1
fi

echo "✓ RM Service is running on port $REWARD_MODEL_PORT"
echo ""

echo "[Step 2] Checking Data..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/train.parquet" ]; then
    echo "⚠️  Training data not in parquet format"
    echo ""
    echo "Converting data to parquet format..."
    bash scripts/prepare_data.sh
else
    echo "✓ Training data found"
fi
echo ""

echo "[Step 3] Checking Model..."
echo "----------------------------------------"

if [ ! -d "models/Qwen3-8B-Base" ]; then
    echo "❌ Error: Model not found at models/Qwen3-8B-Base"
    exit 1
fi

echo "✓ Model found"
echo ""

echo "[Step 4] Activating Environment..."
echo "----------------------------------------"

source /data/wudw/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/data/wudw/miniconda3/envs/verl/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH

echo "✓ veRL environment activated"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  LD_LIBRARY_PATH set"
echo ""

echo "[Step 5] Starting GRPO Training..."
echo "----------------------------------------"

export WANDB_PROJECT=grpo_verl_openreview

echo "Training command:"
echo "  bash scripts/run_grpo_verl.sh"
echo ""
echo "Logs will be saved to: logs/grpo_training_$(date +%Y%m%d_%H%M%S).log"
echo ""

mkdir -p logs

bash scripts/run_grpo_verl.sh 2>&1 | tee logs/grpo_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "Training Completed!"
echo "========================================"
echo ""
echo "Model saved to: models/grpo_qwen3_8b_verl/"
echo ""