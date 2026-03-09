#!/bin/bash
set -e

cd /data/wudy/RL/llm-review-sys-RL

echo "========================================"
echo "Starting Reward Model Training"
echo "Date: $(date)"
echo "========================================"

# Clean up any existing model
if [ -d "models/reward_model_qwen3_8b" ]; then
    echo "Removing existing reward model..."
    rm -rf models/reward_model_qwen3_8b
fi

# Start training
WANDB_PROJECT=reward_model_grpo \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=18579 \
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
swift rlhf \
    --rlhf_type rm \
    --model models/qwen3-8b-base \
    --dataset data/openreview_dataset/rm_train.json \
    --val_dataset data/openreview_dataset/rm_val.json \
    --columns '{"query": "query", "chosen": "chosen", "rejected": "rejected"}' \
    --output_dir models/reward_model_qwen3_8b \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --gradient_checkpointing true \
    --bf16 true \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --deepspeed configs/deepspeed_zero3_config.json \
    --beta 0.1 \
    --report_to wandb \
    --run_name rm_qwen3_8b_v1

echo "========================================"
echo "Training completed!"
echo "Date: $(date)"
echo "========================================"
