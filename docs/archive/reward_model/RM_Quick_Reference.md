# Reward Model 训练快速参考

> 一页纸快速参考，包含最常用的命令和配置

---

## 快速开始

```bash
# 1. 准备数据
python scripts/convert_dpo_to_rm_swift_format.py \
    --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
    --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
    --max_train_samples 5000 --random_sample

# 2. 启动训练
nohup bash scripts/train_rm_full_content_7gpu.sh > logs/rm_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 3. 监控
tail -f logs/rm_training_*.log
```

---

## 训练配置速查表

### 推荐配置（当前）

| 参数 | 值 | 说明 |
|------|-----|------|
| max_length | 16384 | 覆盖 99.4% 样本 |
| batch_size | 2 | 每GPU 2个样本 |
| grad_accum | 8 | 梯度累积 8 步 |
| 有效 BS | 112 | 2×7×8 |
| 学习率 | 1e-5 | 稳定训练 |
| 显存 | ~77 GiB | 96% 利用率 |
| DeepSpeed | ZeRO-2 | 速度快 |

### 保守配置（显存紧张）

```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--deepspeed configs/deepspeed_zero3_config.json \
--max_length 12288 \
# 显存: ~35 GiB, 速度: ~400s/步
```

### 激进配置（显存充足）

```bash
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--deepspeed configs/deepspeed_zero2_config.json \
# 显存: ~85 GiB (可能OOM), 速度: ~200s/步
```

---

## 监控命令

### 训练状态

```bash
# 实时日志
tail -f logs/rm_training_*.log

# GPU 状态
watch -n 1 nvidia-smi

# 进程状态
ps aux | grep "swift rlhf" | grep -v grep

# 检查点
ls -lh models/reward_model_qwen3_8b/v0-*/checkpoint-*/
```

### 训练指标

```bash
# Loss
grep "loss" logs/rm_training_*.log | tail -20

# 显存
grep "memory" logs/rm_training_*.log | tail -20

# 进度
grep "Train:" logs/rm_training_*.log | tail -20

# WandB
https://wandb.ai/liverspecial-dlut-edu-cn/reward_model_grpo
```

---

## 故障排查

### OOM（显存不足）

```bash
# 1. 减小 batch size
--per_device_train_batch_size 1

# 2. 减小 max_length
--max_length 12288

# 3. 使用 ZeRO-3
--deepspeed configs/deepspeed_zero3_config.json
```

### GPU 被占用

```bash
# 检查
nvidia-smi

# 使用其他 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 bash scripts/train_rm_full_content_7gpu.sh

# 或杀掉进程
kill -9 <PID>
```

### Loss 不下降

```bash
# 调整学习率
--learning_rate 2e-5  # 尝试 5e-6 ~ 5e-5

# 增加训练轮数
--num_train_epochs 3
```

---

## 数据分析

```bash
# Token 分布
python scripts/analyze_rm_token_distribution.py

# 详细统计
cat docs/rm_token_stats.txt

# 完整分析
cat docs/rm_query_token_distribution_comprehensive.txt
```

---

## 估算时间

### 当前配置

```
样本数: 5000
有效 BS: 112
步数: 5000 / 112 = 45 步/epoch
总步数: 45 × 2 = 90 步
每步时长: ~320 秒
总时长: 90 × 320 / 3600 ≈ 8 小时
```

### 不同配置对比

| 配置 | 每步时长 | 总时长 | 显存 |
|------|---------|-------|------|
| batch=1, ZeRO-3 | ~400s | 10h | 35 GiB |
| **batch=2, ZeRO-2** | **~320s** | **8h** | **77 GiB** |
| batch=4, ZeRO-2 | ~200s | 5h | 85 GiB |

---

## 常用参数

### 学习率

```bash
# 保守
--learning_rate 5e-6

# 当前（推荐）
--learning_rate 1e-5

# 激进
--learning_rate 2e-5
```

### max_length

```bash
# 小（节省显存）
--max_length 12288  # 95% 覆盖

# 中（推荐）
--max_length 16384  # 99.4% 覆盖

# 大（覆盖更多）
--max_length 20000  # 99.5% 覆盖
```

### Batch Size

```bash
# 小
--per_device_train_batch_size 1

# 中（推荐）
--per_device_train_batch_size 2

# 大
--per_device_train_batch_size 4
```

---

## 关键指标

### 正常训练

```
Step 1:   loss=0.70, acc=0.48, margin=-0.01
Step 10:  loss=0.69, acc=0.74, margin=0.00
Step 50:  loss=0.58, acc=0.88, margin=0.12
Step 90:  loss=0.52, acc=0.92, margin=0.18
```

**关键**:
- loss: 0.70 → 0.52 ✅
- acc: 0.48 → 0.92 ✅
- margin: 0.00 → 0.18 ✅

### 异常情况

```
# Loss 不下降
loss 保持 ~0.70  → 调大学习率

# Loss 爆炸
loss > 1.0       → 调小学习率

# Acc 不上升
acc 保持 ~0.50   → 检查数据
```

---

## 完整文档

详细内容请查看：
- **[完整训练指南](docs/Reward_Model_Training_Guide.md)**
- [Query 策略分析](docs/RM_QUERY_STRATEGY_FINAL.md)
- [GRPO 训练](docs/GRPO_Training_README.md)

---

## 联系与支持

- WandB: https://wandb.ai/liverspecial-dlut-edu-cn/reward_model_grpo
- 日志: `logs/rm_training_*.log`
- 问题: 查看文档或联系维护者
