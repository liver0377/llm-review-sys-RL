#!/bin/bash

set -e

echo "========================================"
echo "GRPO Training - Step by Step"
echo "========================================"

cd /data/wudy/RL/llm-review-sys-RL

echo ""
echo "[Step 1] Converting DPO data to RM format..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/rm_train.json" ]; then
    python scripts/convert_dpo_to_rm.py \
        --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
        --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
        --max_train_samples 5000 \
        --random_sample \
        --seed 42
else
    echo "RM data exists, skipping..."
fi

echo ""
echo "[Step 2] Preparing GRPO data..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/grpo_train.json" ]; then
    python scripts/prepare_grpo_data.py \
        --max_train_samples 3000 \
        --random_sample \
        --seed 42
else
    echo "GRPO data exists, skipping..."
fi

echo ""
echo "[Step 3] Training Reward Model..."
echo "----------------------------------------"

if [ ! -d "models/reward_model_qwen3_8b" ]; then
    WANDB_PROJECT=reward_model_grpo MASTER_ADDR=127.0.0.1 MASTER_PORT=18579 NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    swift rlhf \
        --rlhf_type rm \
        --model models/qwen3-8b-base \
        --dataset data/openreview_dataset/rm_train.json \
        --val_dataset data/openreview_dataset/rm_val.json \
        --custom_dataset_info configs/custom_dataset_info.json \
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
        --deepspeed configs/deepspeed_zero2_config.json \
        --beta 0.1 \
        --report_to wandb \
        --run_name rm_qwen3_8b_v1
    
    echo "Reward Model training completed!"
else
    echo "Removing existing reward model directory..."
    rm -rf models/reward_model_qwen3_8b
    WANDB_PROJECT=reward_model_grpo MASTER_ADDR=127.0.0.1 MASTER_PORT=18579 NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    swift rlhf \
        --rlhf_type rm \
        --model models/qwen3-8b-base \
        --dataset data/openreview_dataset/rm_train.json \
        --val_dataset data/openreview_dataset/rm_val.json \
        --custom_dataset_info configs/custom_dataset_info.json \
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
    
    echo "Reward Model training completed!"
fi

echo ""
echo "[Step 4] Training GRPO with Generative Reward Model..."
echo "----------------------------------------"

REWARD_MODEL_PATH="models/qwen3.5-35b-a3b"

# Check if reward model exists
if [ ! -d "${REWARD_MODEL_PATH}" ]; then
    echo "Error: Reward model not found at ${REWARD_MODEL_PATH}"
    echo "Please download it first using:"
    echo "  bash scripts/download_qwen35_35b_a3b.sh"
    exit 1
fi

echo "Using Generative Reward Model: ${REWARD_MODEL_PATH}"
echo "Reward Model Type: qwen"
echo "Reward Function: Combined (Generative RM + Format Score)"
echo ""

WANDB_PROJECT=grpo_training \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=18579 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model models/qwen3-8b-base \
    --reward_model ${REWARD_MODEL_PATH} \
    --reward_model_type qwen \
    --reward_func train.code.reward_function:combined_reward \
    --alpha 1.0 \
    --format_weight 1.0 \
    --dataset data/openreview_dataset/grpo_train.json \
    --val_dataset data/openreview_dataset/grpo_val.json \
    --output_dir models/grpo_qwen3_8b_generative_rm \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_length 16384 \
    --max_new_tokens 2000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --bf16 true \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed configs/deepspeed_zero3_config.json \
    --num_generations 4 \
    --temperature 0.7 \
    --top_p 0.9 \
    --beta 0.1 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 16384 \
    --vllm_enforce_eager true \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --report_to wandb \
    --run_name grpo_qwen3_8b_generative_rm_v1

echo ""
echo "========================================"
echo "GRPO Training Completed!"
echo "========================================"
echo ""
echo "Models saved to:"
echo "  - Generative Reward Model: models/qwen3.5-35b-a3b (pre-trained, not modified)"
echo "  - GRPO Model:              models/grpo_qwen3_8b_generative_rm"
echo ""
echo "Note: Using Qwen3.5-35B-A3B as a generative reward model for scoring."
