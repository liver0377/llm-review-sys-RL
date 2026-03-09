#!/bin/bash
set -e

echo "========================================"
echo "GRPO Training with Generative Reward Model"
echo "Using Qwen3.5-35B-A3B as Generative RM"
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
echo "[Step 3] Checking Generative Reward Model..."
echo "----------------------------------------"

REWARD_MODEL_PATH="models/qwen3.5-35b-a3b"

if [ ! -d "${REWARD_MODEL_PATH}" ]; then
    echo "Error: Generative Reward Model not found at ${REWARD_MODEL_PATH}"
    echo ""
    echo "Please download it first using:"
    echo "  bash scripts/download_qwen35_35b_a3b.sh"
    echo ""
    echo "Or manually:"
    echo "  python scripts/download_qwen35_35b_a3b.py"
    exit 1
fi

echo "✓ Generative Reward Model found: ${REWARD_MODEL_PATH}"
echo "  Model: Qwen3.5-35B-A3B"
echo "  Type: Generative Reward Model"
echo ""

echo ""
echo "[Step 4] Starting GRPO Training..."
echo "----------------------------------------"

echo "Configuration:"
echo "  Policy Model:        models/qwen3-8b-base"
echo "  Generative RM:       ${REWARD_MODEL_PATH}"
echo "  Reward Function:     Combined (RM + Format Score)"
echo "  Alpha (RM weight):   1.0"
echo "  Format Weight:       1.0"
echo "  GPUs:                8 (CUDA:0-7)"
echo "  Batch Size:          1 x 8 = 8 (accumulation)"
echo "  Max Length:          16384"
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
    --reward_model_plugin train.code.genrm_plugin:get_review_genrm_plugin \
    --external_plugins train/code/genrm_plugin.py \
    --alpha 1.0 \
    --format_weight 1.0 \
    --dataset data/openreview_dataset/grpo_train.json \
    --val_dataset data/openreview_dataset/grpo_val.json \
    --output_dir models/grpo_qwen3_8b_grm \
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
    --run_name grpo_qwen3_8b_grm_v1

echo ""
echo "========================================"
echo "GRPO Training Completed!"
echo "========================================"
echo ""
echo "Models:"
echo "  ✓ Policy Model:      models/grpo_qwen3_8b_grm"
echo "  ✓ Generative RM:     ${REWARD_MODEL_PATH} (unchanged)"
echo ""
echo "Reward Configuration:"
echo "  - Type:              Generative Reward Model (Qwen3.5-35B-A3B)"
echo "  - Format Score:      0-4 points"
echo "  - RM Score:          0-10 points"
echo "  - Combined Formula:  format_weight * format + alpha * rm_score"
echo "  - Weights:           format_weight=1.0, alpha=1.0"
echo ""
echo "To evaluate the model:"
echo "  python eval/eval.py --model_name grpo_qwen3_8b_grm"
