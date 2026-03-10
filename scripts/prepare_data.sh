#!/bin/bash

set -e

echo "========================================"
echo "Preparing Data for veRL GRPO Training"
echo "========================================"

cd /data/wudw/llm-review-sys-RL

# Check input data
if [ ! -f "data/openreview_dataset/grpo_train.json" ]; then
    echo "❌ Error: Training data not found at data/openreview_dataset/grpo_train.json"
    exit 1
fi

if [ ! -f "data/openreview_dataset/grpo_val.json" ]; then
    echo "❌ Error: Validation data not found at data/openreview_dataset/grpo_val.json"
    exit 1
fi

echo "✓ Input data found"
echo ""

# Convert training data
echo "[Step 1] Converting training data..."
python scripts/data/prepare_openreview_parquet.py \
    --input_json data/openreview_dataset/grpo_train.json \
    --output_parquet data/openreview_dataset/train.parquet

echo ""

# Convert validation data
echo "[Step 2] Converting validation data..."
python scripts/data/prepare_openreview_parquet.py \
    --input_json data/openreview_dataset/grpo_val.json \
    --output_parquet data/openreview_dataset/val.parquet

echo ""

echo "========================================"
echo "Data Preparation Completed!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - data/openreview_dataset/train.parquet"
echo "  - data/openreview_dataset/val.parquet"
echo ""