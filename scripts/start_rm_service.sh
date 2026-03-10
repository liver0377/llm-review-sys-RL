#!/bin/bash
set -e

echo "========================================"
echo "Starting Reward Model Service"
echo "Using Qwen3-8B-Base with vLLM"
echo "========================================"

cd /data/wudw/llm-review-sys-RL

REWARD_MODEL_PATH="models/Qwen3-8B-Base"

# 检查模型是否存在
if [ ! -d "${REWARD_MODEL_PATH}" ]; then
    echo "Error: Reward Model not found at ${REWARD_MODEL_PATH}"
    echo ""
    echo "Please download it first using:"
    echo "  bash scripts/download_qwen35_35b_a3b.sh"
    echo ""
    echo "Or manually:"
    echo "  python scripts/download_qwen35_35b_a3b.py"
    exit 1
fi

echo "✓ Reward Model found: ${REWARD_MODEL_PATH}"
echo ""

# 检查端口是否被占用
PORT=8002
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use!"
    echo ""
    echo "Please stop the service using:"
    echo "  bash scripts/stop_rm_service.sh"
    echo ""
    echo "Or use a different port by modifying this script."
    exit 1
fi

echo "Configuration:"
echo "  Reward Model:        ${REWARD_MODEL_PATH}"
echo "  GPUs:               4,5,6,7"
echo "  Tensor Parallel:    4"
echo "  Port:               $PORT"
echo "  Max Model Length:   8192"
echo "  Max Num Seqs:       128"
echo "  Max Batched Tokens: 65536"
echo "  GPU Memory Util:    0.9"
echo ""

echo "[Step 1] Checking GPU availability..."
echo "----------------------------------------"

# 检查 GPU 4-7 是否可用
echo "Checking GPUs 4,5,6,7..."
if nvidia-smi --list-gpus | grep -E "^GPU [4-7]:" > /dev/null; then
    echo "✓ GPUs 4-7 are available"
else
    echo "❌ Error: GPUs 4-7 are not available"
    exit 1
fi

echo ""
echo "[Step 2] Starting vLLM Service..."
echo "----------------------------------------"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 激活 conda 环境
source /data/wudw/miniconda3/etc/profile.d/conda.sh
conda activate swift

# 启动 vLLM 服务
echo "Command:"
echo "  CUDA_VISIBLE_DEVICES=4,5,6,7 \\"
echo "  conda activate swift \\"
echo "  vllm serve ${REWARD_MODEL_PATH} \\"
echo "    --tensor-parallel-size 4 \\"
echo "    --port $PORT \\"
echo "    --host 0.0.0.0 \\"
echo "    --max-model-len 8192 \\"
echo "    --max-num-seqs 128 \\"
echo "    --max-num-batched-tokens 65536 \\"
echo "    --gpu-memory-utilization 0.9 \\"
echo "    --dtype bfloat16 \\"
echo "    --trust-remote-code"
echo ""

# 启动服务（后台运行）
nohup vllm serve "${REWARD_MODEL_PATH}" \
    --tensor-parallel-size 4 \
    --port $PORT \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --trust-remote-code \
    > logs/rm_service.log 2>&1 &
RM_PID=$!

echo "✓ RM Service started!"
echo "  Process ID: $RM_PID"
echo "  Log file: logs/rm_service.log"
echo ""

# 保存 PID 到文件
echo $RM_PID > /tmp/rm_service.pid
echo "✓ PID saved to /tmp/rm_service.pid"

echo ""
echo "[Step 3] Waiting for service to be ready..."
echo "----------------------------------------"

# 等待服务启动
MAX_WAIT=300  # 最多等待 5 分钟
WAIT_TIME=5
TOTAL_WAIT=0
SIGNAL_FILE="/tmp/rm_service_ready"

while [ $TOTAL_WAIT -lt $MAX_WAIT ]; do
    sleep $WAIT_TIME
    TOTAL_WAIT=$((TOTAL_WAIT + WAIT_TIME))
    
    # 检查服务是否就绪
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "✓ RM Service is ready!"
        touch $SIGNAL_FILE
        break
    fi
    
    if [ $((TOTAL_WAIT % 10)) -eq 0 ]; then
        echo "  Waiting... ($TOTAL_WAIT/$MAX_WAIT seconds)"
    fi
done

if [ $TOTAL_WAIT -ge $MAX_WAIT ]; then
    echo ""
    echo "❌ Error: RM Service failed to start within $MAX_WAIT seconds"
    echo ""
    echo "Please check the log:"
    echo "  tail -100 logs/rm_service.log"
    exit 1
fi

echo ""
echo "========================================"
echo "RM Service Started Successfully!"
echo "========================================"
echo ""
echo "Service Info:"
echo "  - PID:          $RM_PID"
echo "  - Port:         $PORT"
echo "  - Base URL:     http://127.0.0.1:$PORT/v1"
echo "  - Health Check: http://127.0.0.1:$PORT/health"
echo ""
echo "To check logs:"
echo "  tail -f logs/rm_service.log"
echo ""
echo "To stop the service:"
echo "  bash scripts/stop_rm_service.sh"
echo "  kill $RM_PID"
echo ""
