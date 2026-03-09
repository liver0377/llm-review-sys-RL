#!/bin/bash
set -e

echo "========================================"
echo "Stopping Reward Model Service"
echo "========================================"

PID_FILE="/tmp/rm_service.pid"

# 检查 PID 文件是否存在
if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  PID file not found: $PID_FILE"
    echo ""
    echo "Trying to find vLLM process on port 8000..."
    
    # 尝试找到占用端口 8000 的进程
    PID=$(lsof -t -i :8000 | head -1 | awk '{print $2}')
    
    if [ -z "$PID" ]; then
        echo "❌ No vLLM service found running on port 8000"
        exit 1
    fi
    
    echo "Found vLLM service with PID: $PID"
else
    # 从 PID 文件读取
    PID=$(cat "$PID_FILE")
    
    # 检查进程是否存在
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Process $PID is not running"
        rm -f "$PID_FILE"
        exit 0
    fi
fi

echo "Stopping RM Service (PID: $PID)..."

# 发送 SIGTERM 信号（优雅关闭）
kill -TERM $PID

# 等待进程结束
WAIT_TIME=30
ELAPSED=0
while [ $ELAPSED -lt $WAIT_TIME ]; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✓ RM Service stopped successfully"
        rm -f "$PID_FILE"
        
        # 清理信号文件
        SIGNAL_FILE="/tmp/rm_service_ready"
        rm -f "$SIGNAL_FILE"
        
        exit 0
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# 如果优雅关闭失败，强制关闭
echo "⚠️  Service did not stop gracefully, forcing..."
kill -9 $PID
sleep 2

if ! ps -p $PID > /dev/null 2>&1; then
    echo "✓ RM Service stopped successfully (forced)"
    rm -f "$PID_FILE"
    rm -f "/tmp/rm_service_ready"
    exit 0
else
    echo "❌ Error: Failed to stop RM Service"
    exit 1
fi
