# Reward Model Training Guide

> 完整的 Reward Model 训练指南，包括数据合成、训练脚本使用、参数调优等  
> Last Updated: 2026-03-06

---

## 目录

1. [概述](#概述)
2. [数据准备](#数据准备)
3. [Token 分布分析](#token-分布分析)
4. [训练配置](#训练配置)
5. [训练脚本使用](#训练脚本使用)
6. [训练监控](#训练监控)
7. [常见问题](#常见问题)
8. [优化建议](#优化建议)

---

## 概述

### 目标

训练一个 Reward Model，用于评估论文评审质量，能够：
- 识别评审是否指出真实缺陷
- 检测评审是否捏造内容
- 评估评审的准确性和建设性

### 架构

```
Base Model: Qwen3-8B (models/qwen3-8b-base)
Task: Sequence Classification (Reward Modeling)
Output: Scalar Reward Score
```

### 训练策略

**Query 包含完整论文内容**，提供最大上下文：
- ✅ 99.4% 样本覆盖 (max_length=16384)
- ✅ 可检测具体的捏造内容
- ✅ 能验证评审的准确性

---

## 数据准备

### 1. 数据来源

使用 DPO 数据作为基础：
```bash
data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json  # 训练集
data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json    # 验证集
```

### 2. 数据格式转换

#### 原始 DPO 格式

```json
{
  "prompt": "You are an academic paper reviewer...\n\nPaper Details:\n- Title: ...\n- Content: ...",
  "chosen": "### Key Points\n...",
  "rejected": "### Key Points\n..."
}
```

#### Swift RM 期望格式

```json
{
  "messages": [
    {"role": "user", "content": "{instruction}\n\nPaper Title: {title}\n\nPaper Content:\n{full_content}"},
    {"role": "assistant", "content": "{chosen_response}"}
  ],
  "rejected_messages": [
    {"role": "user", "content": "{instruction}\n\nPaper Title: {title}\n\nPaper Content:\n{full_content}"},
    {"role": "assistant", "content": "{rejected_response}"}
  ]
}
```

#### 转换脚本

```bash
python scripts/convert_dpo_to_rm_swift_format.py \
    --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
    --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
    --max_train_samples 5000 \
    --random_sample \
    --seed 42
```

**参数说明**：
- `--max_train_samples`: 训练样本数量（建议 5000）
- `--max_val_samples`: 验证样本数量（建议使用全部）
- `--random_sample`: 随机采样（推荐）
- `--seed`: 随机种子（保证可复现）

**输出**：
```
data/openreview_dataset/rm_train.json  # 5000 samples
data/openreview_dataset/rm_val.json    # 1052 samples
```

### 3. 数据验证

```bash
python3 << 'EOF'
import json

# 加载数据
with open('data/openreview_dataset/rm_train.json') as f:
    data = json.load(f)

# 检查格式
print(f"Total samples: {len(data)}")
print(f"Sample 0 keys: {list(data[0].keys())}")
print(f"Messages length: {len(data[0]['messages'])}")
print(f"Rejected messages length: {len(data[0]['rejected_messages'])}")

# 检查字段
assert all('messages' in item and 'rejected_messages' in item for item in data)
print("✅ Data format valid!")
EOF
```

---

## Token 分布分析

### 运行分析脚本

```bash
python scripts/analyze_rm_token_distribution.py
```

### 关键统计信息

#### 完整论文数据（当前使用）

| 指标 | Query | Chosen | Rejected | Q+C Total | Q+R Total |
|------|-------|--------|----------|-----------|-----------|
| **Mean** | 8624 | 585 | 537 | 9208 | 9161 |
| **Median** | 8370 | 578 | 527 | 8970 | 8943 |
| **P95** | 12030 | 868 | 769 | 12582 | 12469 |
| **P99** | 14531 | 1028 | 930 | 15168 | 15050 |
| **Max** | 25427 | 1862 | 1902 | 25901 | 25885 |

#### 不同 max_length 的覆盖率

| max_length | 超出样本 | 覆盖率 | 推荐 |
|-----------|---------|-------|------|
| 4096 | 98.8% | ❌ 不可用 | - |
| 8192 | 73.1% | ❌ 不可用 | - |
| **16384** | **0.6%** | ✅ **99.4%** | ⭐⭐⭐ |
| 32768 | 0.0% | ✅ 100% | ⭐⭐ |

**推荐**: `max_length=16384`

### 查看详细分析

```bash
# Token 统计
cat docs/rm_token_stats.txt

# 分布图
open docs/rm_token_distribution.png

# 完整分析
cat docs/rm_query_token_distribution_comprehensive.txt
```

---

## 训练配置

### 推荐配置（7 GPU）

```bash
# 基础配置
max_length=16384                    # 覆盖 99.4% 样本
per_device_train_batch_size=2       # 每GPU 2个样本
gradient_accumulation_steps=8       # 梯度累积
effective_batch_size=112            # 2 × 8 GPUs × 8 = 112 (实际 7 GPU: 2×7×8=112)

# 优化器
learning_rate=1e-5                  # 学习率
weight_decay=0.01                   # 权重衰减
warmup_ratio=0.1                    # 预热比例
num_train_epochs=2                  # 训练轮数

# DeepSpeed
deepspeed=configs/deepspeed_zero2_config.json  # ZeRO-2

# 其他
gradient_checkpointing=true         # 梯度检查点
bf16=true                           # BF16 混合精度
```

### 显存估算

| 配置 | batch_size | ZeRO Stage | 显存/GPU | 利用率 | 备注 |
|------|-----------|------------|---------|-------|------|
| 当前 | 2 | ZeRO-2 | ~77 GiB | 96% | ⭐ 推荐 |
| 保守 | 1 | ZeRO-3 | ~45 GiB | 56% | 显存紧张时使用 |
| 激进 | 4 | ZeRO-2 | ~85 GiB | 106% | ❌ 会 OOM |

### DeepSpeed 配置

#### ZeRO-2（推荐）

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"},
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "bf16": {"enabled": "auto"},
  "gradient_accumulation_steps": "auto"
}
```

**优点**：
- ✅ 通信开销小
- ✅ 速度快
- ✅ 适合 7-8 GPU

**缺点**：
- ⚠️ 显存占用稍高

#### ZeRO-3（保守）

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {"enabled": "auto"}
}
```

**优点**：
- ✅ 显存占用低
- ✅ 适合显存紧张

**缺点**：
- ⚠️ 通信开销大
- ⚠️ 速度慢 ~30%

---

## 训练脚本使用

### 快速开始

```bash
# 1. 准备数据
python scripts/convert_dpo_to_rm_swift_format.py \
    --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
    --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
    --max_train_samples 5000 \
    --random_sample

# 2. 启动训练
bash scripts/train_rm_full_content_7gpu.sh
```

### 训练脚本详解

```bash
#!/bin/bash
set -e

cd /data/wudy/RL/llm-review-sys-RL

# 环境变量
export WANDB_PROJECT=reward_model_grpo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=18579
export NPROC_PER_NODE=7
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7  # 跳过 GPU 4

# 训练命令
swift rlhf \
    --rlhf_type rm \                          # Reward Model 类型
    --model models/qwen3-8b-base \            # 基础模型
    --dataset data/openreview_dataset/rm_train.json \
    --val_dataset data/openreview_dataset/rm_val.json \
    --output_dir models/reward_model_qwen3_8b \
    --tuner_type full \                       # 全参数微调
    --torch_dtype bfloat16 \                  # 数据类型
    --max_length 16384 \                      # 最大序列长度
    --per_device_train_batch_size 2 \         # 每GPU batch size
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \         # 梯度累积
    --learning_rate 1e-5 \                    # 学习率
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \                    # 训练轮数
    --gradient_checkpointing true \           # 梯度检查点
    --bf16 true \                             # BF16
    --eval_strategy steps \                   # 评估策略
    --eval_steps 100 \                        # 每 100 步评估
    --save_steps 200 \                        # 每 200 步保存
    --save_total_limit 2 \                    # 最多保存 2 个 checkpoint
    --logging_steps 10 \                      # 每 10 步记录日志
    --deepspeed configs/deepspeed_zero2_config.json \
    --beta 0.1 \                              # RM loss 参数
    --report_to wandb \                       # 报告到 WandB
    --run_name rm_qwen3_8b_full_paper_v1
```

### 后台运行

```bash
# 使用 nohup
nohup bash scripts/train_rm_full_content_7gpu.sh > logs/rm_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 或使用 screen
screen -S rm_training
bash scripts/train_rm_full_content_7gpu.sh
# Ctrl+A, D 分离
```

### 参数调整

#### 调整 batch size

```bash
# 更大 batch size（需要更多显存）
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \

# 更小 batch size（节省显存）
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
```

#### 调整 max_length

```bash
# 更小 max_length（节省显存，但覆盖率下降）
--max_length 12288 \  # 覆盖 ~95% 样本

# 更大 max_length（覆盖更多样本，但需要更多显存）
--max_length 20000 \  # 覆盖 ~99.5% 样本
```

---

## 训练监控

### WandB 监控

训练日志自动上传到 WandB：
```
https://wandb.ai/liverspecial-dlut-edu-cn/reward_model_grpo
```

**关键指标**：
- `loss`: 训练损失（初始 ~0.69，应逐渐下降）
- `rewards/accuracies`: 奖励准确率（应逐渐上升）
- `rewards/margins`: chosen 与 rejected 的奖励差（应逐渐上升）
- `learning_rate`: 当前学习率
- `memory(GiB)`: GPU 显存使用

### 日志监控

```bash
# 实时查看日志
tail -f logs/rm_training_*.log

# 检查训练进度
grep "Train:" logs/rm_training_*.log | tail -20

# 检查 loss
grep "loss" logs/rm_training_*.log | tail -20

# 检查显存
grep "memory" logs/rm_training_*.log | tail -20
```

### 检查点管理

```bash
# 查看保存的 checkpoint
ls -lh models/reward_model_qwen3_8b/v0-*/checkpoint-*/

# 典型输出
checkpoint-200/
  - adapter_config.json
  - adapter_model.safetensors
  - optimizer.pt
  - rng_state_0.pth
  - scheduler.pt
  - trainer_state.json
  - training_args.bin
```

### 训练时长估算

**当前配置**：
- 样本数: 5000
- 有效 batch size: 112 (2 × 7 × 8)
- 每 epoch 步数: 5000 / 112 ≈ 45 步
- 总步数: 45 × 2 epochs = 90 步
- 每步时长: ~320 秒
- **预计总时长**: 90 × 320 / 3600 ≈ **8 小时**

---

## 常见问题

### 1. `inputs.rejected is None` 错误

**原因**: 数据格式不正确

**解决**: 使用 Swift 格式（messages + rejected_messages）

```bash
# 正确格式
{
  "messages": [...],
  "rejected_messages": [...]
}

# 错误格式
{
  "query": "...",
  "chosen": "...",
  "rejected": "..."
}
```

### 2. CUDA Out of Memory

**原因**: 显存不足

**解决方案**：

```bash
# 方案 1: 减小 batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \

# 方案 2: 减小 max_length
--max_length 12288 \

# 方案 3: 使用 ZeRO-3
--deepspeed configs/deepspeed_zero3_config.json \
```

### 3. GPU 被占用

**检查**:
```bash
nvidia-smi
```

**解决**:
```bash
# 方案 1: 等待进程结束
# 方案 2: 使用其他 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 bash scripts/train_rm_full_content_7gpu.sh

# 方案 3: 杀掉进程（谨慎）
kill -9 <PID>
```

### 4. 训练速度慢

**可能原因**:
- max_length 太大
- gradient_checkpointing 开启
- ZeRO-3 通信开销

**优化**:
```bash
# 减小 max_length
--max_length 12288 \

# 增大 batch size（如果显存允许）
--per_device_train_batch_size 4 \

# 使用 ZeRO-2
--deepspeed configs/deepspeed_zero2_config.json \
```

### 5. Loss 不下降

**可能原因**:
- 学习率太小/太大
- 数据质量问题
- 模型容量不足

**解决**:
```bash
# 调整学习率
--learning_rate 2e-5 \  # 尝试 5e-6 ~ 5e-5

# 检查数据质量
python scripts/analyze_rm_token_distribution.py

# 增加训练轮数
--num_train_epochs 3 \
```

### 6. `--columns` 参数问题

**问题**: 是否需要使用 `--columns` 或 `--custom_dataset_info`？

**答案**: ❌ **不需要**

**原因**:
1. 数据已经使用正确的字段名（`messages`, `rejected_messages`）
2. Swift 会自动识别这些标准字段
3. 使用这些参数可能导致 bug

**正确做法**:
```bash
# ✅ 推荐：直接加载数据
swift rlhf --dataset data/openreview_dataset/rm_train.json ...

# ❌ 不推荐：多余的配置
swift rlhf --dataset data/openreview_dataset/rm_train.json \
    --columns '{"query": "query", ...}' \  # 不需要
    --custom_dataset_info configs/... \    # 不需要
```

---

## 优化建议

### 显存优化

#### 当前配置（推荐）

```
batch_size=2, ZeRO-2, max_length=16384
显存: ~77 GiB / 80 GiB (96%)
速度: ~320 秒/步
```

#### 激进配置（如果显存充足）

```bash
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--deepspeed configs/deepspeed_zero2_config.json \

# 预计显存: ~85 GiB (可能 OOM，需要 8 GPU)
# 预计速度: ~200 秒/步（提速 40%）
```

#### 保守配置（如果显存紧张）

```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--deepspeed configs/deepspeed_zero3_config.json \
--max_length 12288 \

# 预计显存: ~35 GiB
# 预计速度: ~400 秒/步
```

### 速度优化

#### 1. 增大 batch size

```bash
# 如果显存允许
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \  # 保持相同有效 batch size
```

**提速**: ~40%

#### 2. 使用更少 GPU 但更大 batch size

```bash
# 7 GPU, batch_size=2
有效 batch size = 2 × 7 × 8 = 112

# 8 GPU, batch_size=2
有效 batch size = 2 × 8 × 8 = 128

# 选择：根据 GPU 可用性
```

#### 3. 调整 gradient checkpointing

```bash
# 如果显存充足，关闭 gradient checkpointing
--gradient_checkpointing false \

# 节省显存: ~10 GiB
# 提速: ~15%
# 风险: 显存可能不足
```

### 模型质量优化

#### 1. 学习率调整

```bash
# 当前
--learning_rate 1e-5 \

# 实验建议
--learning_rate 5e-6 \   # 保守，更稳定
--learning_rate 2e-5 \   # 激进，更快收敛
```

#### 2. 数据增强

```bash
# 使用更多数据
--max_train_samples 8000 \  # 从 5000 增加到 8000

# 或使用全部数据（~9000 samples）
# 不指定 --max_train_samples
```

#### 3. 训练轮数

```bash
# 当前
--num_train_epochs 2 \

# 建议
--num_train_epochs 3 \  # 更充分训练
```

---

## 训练结果示例

### 正常训练曲线

```
Step 1:   loss=0.705, acc=0.48, margin=-0.01
Step 10:  loss=0.691, acc=0.74, margin=0.00
Step 20:  loss=0.650, acc=0.82, margin=0.05
Step 50:  loss=0.580, acc=0.88, margin=0.12
Step 90:  loss=0.520, acc=0.92, margin=0.18
```

**指标解释**：
- `loss`: 应从 ~0.70 下降到 ~0.50
- `acc`: 应从 ~0.50 上升到 ~0.90+
- `margin`: 应从 ~0.00 上升到 ~0.15+

### 异常情况

#### Loss 不下降

```
Step 1:   loss=0.705
Step 10:  loss=0.703
Step 20:  loss=0.704
```

**可能原因**:
- 学习率太小
- 数据问题

**解决**: 调大学习率或检查数据

#### Loss 爆炸

```
Step 1:   loss=0.705
Step 10:  loss=1.234
Step 20:  loss=5.678
```

**可能原因**:
- 学习率太大
- 梯度爆炸

**解决**: 减小学习率或添加 gradient clipping

---

## 下一步

训练完成后：

1. **评估 RM 模型**
   ```bash
   python scripts/evaluate_rm.py \
       --model models/reward_model_qwen3_8b/v0-*/checkpoint-final \
       --test_data data/openreview_dataset/rm_val.json
   ```

2. **准备 GRPO 训练**
   - 参考: `docs/GRPO_Training_README.md`
   - 使用训练好的 RM 作为奖励函数

3. **模型推理**
   ```python
   from transformers import AutoModelForSequenceClassification
   
   model = AutoModelForSequenceClassification.from_pretrained(
       "models/reward_model_qwen3_8b/v0-*/checkpoint-final"
   )
   
   # 计算奖励分数
   reward = model(**inputs).logits
   ```

---

## 参考资料

- [Swift 文档](https://github.com/modelscope/swift)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Reward Modeling 论文](https://arxiv.org/abs/2203.02155)
- [WandB 监控](https://wandb.ai/)

---

## 更新日志

### 2026-03-06
- ✅ 创建完整训练指南
- ✅ 添加数据合成流程
- ✅ 优化 batch_size 到 2
- ✅ 使用 ZeRO-2 提升显存利用率
- ✅ 添加常见问题解答

---

## 联系方式

如有问题，请查看：
- 训练日志: `logs/rm_training_*.log`
- WandB: https://wandb.ai/liverspecial-dlut-edu-cn/reward_model_grpo
- 文档: `docs/RM_QUERY_STRATEGY_FINAL.md`
