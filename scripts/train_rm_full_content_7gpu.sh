#!/bin/bash
set -e

cd /data/wudy/RL/llm-review-sys-RL

echo "========================================"
echo "Starting Reward Model Training (35B Generative RM)"
echo "Date: $(date)"
echo "========================================"

# Get current date in MM_DD format
DATE=$(date +%m_%d)
OUTPUT_DIR="models/reward_model_qwen35_35b/${DATE}"

# Create date directory and clean up only today's old model
mkdir -p "models/reward_model_qwen35_35b"
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing existing reward model for today (${DATE})..."
    rm -rf "${OUTPUT_DIR}"
fi

# Kill any stuck processes
pkill -f "swift/cli/rlhf.py.*rm_train.json" 2>/dev/null || true

# Start training with full paper content using Qwen3.5-35B-A3B as Generative Reward Model
# Using 6 GPUs for the larger 35B model
# max_length=16384 covers 99.4% samples (P99=15168, only 0.6% exceed)
# Increased gradient_accumulation_steps to 4 for 35B model memory management
MODEL_PATH="models/qwen3.5-35b-a3b"

WANDB_PROJECT=reward_model_grpo \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=18579 \
NPROC_PER_NODE=6 \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type rm \
    --model ${MODEL_PATH} \
    --dataset data/openreview_dataset/rm_train.json \
    --val_dataset data/openreview_dataset/rm_val.json \
    --output_dir ${OUTPUT_DIR} \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_length 16384 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --gradient_checkpointing true \
    --bf16 true \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --deepspeed configs/deepspeed_zero2_config.json \
    --beta 0.1 \
    --report_to wandb \
    --run_name rm_qwen35_35b_generative_${DATE}

echo "========================================"
echo "Training completed!"
echo "Date: $(date)"
echo "========================================"