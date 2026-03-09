#!/bin/bash
set -e

echo "========================================"
echo "GRPO Training Pipeline"
echo "========================================"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "[Step 0] Checking environment..."
echo "----------------------------------------"

if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

if ! python -c "import swift" 2>/dev/null; then
    echo "Installing ms-swift..."
    pip install ms-swift -U
fi

echo "Python: $(python --version)"
echo "Swift: $(python -c 'import swift; print(swift.__version__)')"

echo ""
echo "[Step 1] Preparing data..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/rm_train.json" ]; then
    echo "Converting DPO data to RM format..."
    python scripts/convert_dpo_to_rm.py
else
    echo "RM data already exists, skipping conversion"
fi

if [ ! -f "data/openreview_dataset/grpo_train.json" ]; then
    echo "Preparing GRPO data..."
    python scripts/prepare_grpo_data.py
else
    echo "GRPO data already exists, skipping preparation"
fi

echo ""
echo "[Step 2] Training Reward Model..."
echo "----------------------------------------"

if [ ! -d "models/reward_model_qwen3_8b" ]; then
    echo "Training Reward Model..."
    python train/code/train_reward_model.py --config configs/reward_model_config.yaml
    
    echo ""
    echo "Reward Model training completed!"
    echo "Model saved to: models/reward_model_qwen3_8b"
else
    echo "Reward Model already exists, skipping training"
    echo "To retrain, remove: models/reward_model_qwen3_8b"
fi

echo ""
echo "[Step 3] Training GRPO..."
echo "----------------------------------------"

python train/code/train_grpo.py --config configs/grpo_config.yaml

echo ""
echo "========================================"
echo "GRPO Training Pipeline Completed!"
echo "========================================"
echo ""
echo "Output models:"
echo "  - Reward Model: models/reward_model_qwen3_8b"
echo "  - GRPO Model:   models/grpo_qwen3_8b"
echo ""
echo "To evaluate the GRPO model:"
echo "  python eval/eval.py --model_name grpo_qwen3_8b"
