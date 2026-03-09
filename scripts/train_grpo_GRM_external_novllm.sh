#!/bin/bash
set -e

echo "========================================"
echo "GRPO Training with Generative Reward Model"
echo "External Deployment (vLLM Serve) - No vLLM Generation"
echo "Using Qwen3.5-35B-A3B as Generative RM"
echo "========================================"

cd /data/wudw/llm-review-sys-RL

# 配置
RM_SERVICE_PORT=8002
RM_SERVICE_URL="http://127.0.0.1:${RM_SERVICE_PORT}/v1"
TRAIN_GPUS="1,2,3"
RM_GPUS="4,5,6,7"

echo ""
echo "Configuration:"
echo "  Training GPUs:       $TRAIN_GPUS (GPU 1-3)"
echo "  RM Service GPUs:     $RM_GPUS (GPU 4-7)"
echo "  RM Service Port:     $RM_SERVICE_PORT"
echo "  RM Service URL:     $RM_SERVICE_URL"
echo ""

echo ""
echo "[Step 1] Preparing GRPO data..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/grpo_train.json" ]; then
    python scripts/data/prepare_grpo_data.py \
        --max_train_samples 3000 \
        --random_sample \
        --seed 42
else
    echo "GRPO data exists, skipping..."
fi

echo ""
echo "[Step 2] Checking Generative Reward Model..."
echo "----------------------------------------"

REWARD_MODEL_PATH="models/Qwen3.5-35B-A3B-Base"

if [ ! -d "${REWARD_MODEL_PATH}" ]; then
    echo "Error: Generative Reward Model not found at ${REWARD_MODEL_PATH}"
    echo ""
    echo "Please download it first using:"
    echo "  bash scripts/data/download_qwen35_35b_a3b.sh"
    echo ""
    echo "Or manually:"
    echo "  python scripts/data/download_qwen35_35b_a3b.py"
    exit 1
fi

echo "✓ Generative Reward Model found: ${REWARD_MODEL_PATH}"
echo "  Model: Qwen3.5-35B-A3B"
echo "  Type: Generative Reward Model (External Deployment)"
echo ""

echo ""
echo "[Step 3] Starting Reward Model Service..."
echo "----------------------------------------"

# 检查端口是否已被占用
SKIP_RM_STARTUP=false
if lsof -i :$RM_SERVICE_PORT > /dev/null 2>&1; then
    echo "⚠️  Port $RM_SERVICE_PORT is already in use!"
    echo ""
    echo "Detected RM service may already be running on this port."
    echo ""
    echo "Please choose an option:"
    echo "  1) Stop existing service and restart (recommended for fresh start)"
    echo "  2) Skip startup, use existing service (faster, if service is healthy)"
    echo "  3) Cancel and exit"
    echo ""
    read -p "Your choice [1/2/3]: " -n 1 -r
    echo
    echo ""

    case $REPLY in
        1)
            echo "Stopping existing RM service..."
            bash scripts/stop_rm_service.sh 2>/dev/null || true
            sleep 2
            echo "✓ Existing service stopped"
            echo ""
            # 继续执行启动流程
            ;;
        2)
            echo "Skipping RM service startup..."
            echo "Will use existing service on port $RM_SERVICE_PORT"
            SKIP_RM_STARTUP=true
            ;;
        3)
            echo "Cancelled by user."
            echo "To manually manage:"
            echo "  - Check service: lsof -i :$RM_SERVICE_PORT"
            echo "  - Stop service: bash scripts/stop_rm_service.sh"
            echo "  - View logs: tail -f logs/rm_service.log"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
fi

# 启动 RM 服务 (如果未跳过)
if [ "$SKIP_RM_STARTUP" != "true" ]; then
    echo "Starting RM service on GPUs $RM_GPUS..."
    bash scripts/start_rm_service.sh

    # 等待服务启动完成
    if [ -f "/tmp/rm_service_ready" ]; then
        echo "✓ RM Service is ready"
        rm -f /tmp/rm_service_ready
    else
        echo "⚠️  Warning: RM service readiness signal not found"
        echo "  Attempting to continue anyway..."
    fi
else
    echo "✓ Using existing RM service"
fi

echo ""
echo "[Step 4] Verifying RM Service..."
echo "----------------------------------------"

# 健康检查
MAX_RETRIES=5
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$RM_SERVICE_URL/health" > /dev/null 2>&1; then
        echo "✓ RM Service health check passed"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Retry $RETRY_COUNT/$MAX_RETRIES..."
    sleep 2
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "❌ Error: RM Service health check failed"
    echo ""
    echo "Please check:"
    echo "  1. Service status: curl $RM_SERVICE_URL/health"
    echo "  2. Service logs: tail -100 logs/rm_service.log"
    exit 1
fi

# 测试调用
echo ""
echo "Testing RM service..."
python -c "
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='$RM_SERVICE_URL'
)

try:
    response = client.chat.completions.create(
        model='$REWARD_MODEL_PATH',
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=10
    )
    print('✓ RM Service test successful')
except Exception as e:
    print(f'❌ RM Service test failed: {e}')
    exit(1)
" || exit 1

echo ""
echo "[Step 5] Starting GRPO Training..."
echo "----------------------------------------"

# 配置日志
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/grpo_training_${TIMESTAMP}.log"
mkdir -p logs

# 创建符号链接到最新日志
ln -sf "$(basename "$LOG_FILE")" logs/grpo_training_latest.log

echo "Training Configuration:"
echo "  Policy Model:        models/Qwen3.5-9B-Base"
echo "  Training GPUs:        $TRAIN_GPUS"
echo "  RM Service:          $RM_SERVICE_URL"
echo "  Reward Function:      Combined (RM + Format Score)"
echo "  Alpha (RM weight):    1.0"
echo "  Format Weight:        1.0"
echo "  Batch Size:           1 x 8 = 8 (accumulation)"
echo "  Max Length:           16384"
echo "  Generation Mode:      Native (no vLLM)"
echo ""
echo "Logging:"
echo "  Log file:            $LOG_FILE"
echo "  Latest link:         logs/grpo_training_latest.log"
echo "  Monitor:             tail -f $LOG_FILE"
echo ""

WANDB_PROJECT=grpo_training \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=18579 \
NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
swift rlhf \
    --rlhf_type grpo \
    --model models/Qwen3.5-9B-Base \
    --reward_model ${REWARD_MODEL_PATH} \
    --reward_model_plugin train.code.genrm_plugin_external:get_review_genrm_plugin_external \
    --external_plugins train/code/genrm_plugin_external.py \
    --reward_model_api_base $RM_SERVICE_URL \
    --reward_model_api_key EMPTY \
    --alpha 1.0 \
    --format_weight 1.0 \
    --dataset data/openreview_dataset/grpo_train.json \
    --val_dataset data/openreview_dataset/grpo_val.json \
    --output_dir models/grpo_qwen3_9b_grm_external_novllm \
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
    --num_generations 3 \
    --temperature 0.7 \
    --top_p 0.9 \
    --beta 0.1 \
    --sleep_level 1 \
    --report_to wandb \
    --run_name grpo_qwen3_9b_grm_external_novllm_v1 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "GRPO Training Completed!"
echo "========================================"
echo ""
echo "Training Log:"
echo "  - Full log:          $LOG_FILE"
echo "  - Latest log:        logs/grpo_training_latest.log"
echo "  - View log:          less $LOG_FILE"
echo ""
echo "Models:"
echo "  ✓ Policy Model:      models/grpo_qwen3_9b_grm_external_novllm"
echo "  ✓ Generative RM:     ${REWARD_MODEL_PATH} (unchanged)"
echo ""
echo "Reward Configuration:"
echo "  - Type:              External Deployment (vLLM Serve)"
echo "  - RM Service URL:     $RM_SERVICE_URL"
echo "  - Format Score:      0-4 points"
echo "  - RM Score:          0-10 points"
echo "  - Combined Formula:  format_weight × format + alpha × rm_score"
echo "  - Weights:           format_weight=1.0, alpha=1.0"
echo ""
echo "Performance Notes:"
echo "  - Training uses GPUs $TRAIN_GPUS (3 GPUs)"
echo "  - RM Service uses GPUs $RM_GPUS (4 GPUs)"
echo "  - Generation: Native (slower than vLLM, but no extra setup)"
echo "  - RM evaluation time: ~2s (vs ~7.6s in internal)"
echo "  - Trade-off: Simpler setup, slower generation"
echo ""
echo "To evaluate the model:"
echo "  python eval/eval.py --model_name grpo_qwen3_9b_grm_external_novllm"
echo ""
echo "To stop the RM service:"
echo "  bash scripts/stop_rm_service.sh"
echo ""
