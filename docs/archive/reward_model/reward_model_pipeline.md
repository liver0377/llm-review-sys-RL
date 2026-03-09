# Reward Model Training Pipeline

本 Pipeline 用于从 SFT 模型权重初始化训练 Reward Model，保留 SFT 阶段学到的知识。

## 架构概述

```
SFT Model (Qwen3-8B + QLoRA)
    ↓ Merge Adapter
Reward Model (Full Fine-tuning)
    ├─ Base Model (Qwen3-8B, unfrozen)
    └─ Value Head → Scalar Reward Score
```

## 文件结构

```
train/code/
  ├── convert_dpo_to_rm.py      # DPO → Reward Model 数据格式转换
  ├── reward_model.py            # Reward Model 架构（Value Head）
  └── train_reward_model.py      # 训练脚本

configs/
  └── reward_model_config.yaml   # 训练配置

eval/
  └── eval_reward_model.py       # 评估脚本

data/openreview_dataset/
  ├── dpo_base_as_rejected_train_cleaned.json  # 原始 DPO 数据
  └── rm_train.json              # 转换后的 Reward Model 数据

models/
  └── reward_model_qwen3_8b/     # 训练输出目录
      ├── best_model/            # 最佳模型（验证集 loss 最低）
      ├── final_model/           # 最终模型
      └── checkpoint-epoch-N/    # 中间 checkpoint
```

## 使用流程

### 1. 数据转换

将 DPO 格式转换为 Reward Model 格式：

#### 转换全部数据
```bash
python train/code/convert_dpo_to_rm.py \
  --input data/openreview_dataset/dpo_base_as_rejected_train_cleaned.json \
  --output data/openreview_dataset/rm_train.json

python train/code/convert_dpo_to_rm.py \
  --input data/openreview_dataset/dpo_base_as_rejected_val_cleaned.json \
  --output data/openreview_dataset/rm_val.json
```

#### 转换部分数据（推荐：3000条样本）
使用 `--max_samples` 参数限制样本数量，`--shuffle` 确保随机采样：

```bash
python train/code/convert_dpo_to_rm.py \
  --input data/openreview_dataset/dpo_base_as_rejected_train_cleaned.json \
  --output data/openreview_dataset/rm_train_3k.json \
  --max_samples 3000 \
  --shuffle \
  --seed 42
```

**参数说明：**
- `--max_samples N`: 最多转换 N 条样本（默认：全部）
- `--shuffle`: 转换前打乱数据顺序
- `--seed N`: 随机种子（确保可重复性）

**数据格式说明：**
- **DPO 格式**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- **RM 格式**: `{"chosen_text": "prompt + chosen", "rejected_text": "prompt + rejected"}`

### 2. 训练 Reward Model

#### 单 GPU 训练
```bash
python train/code/train_reward_model.py \
  --config configs/reward_model_config.yaml \
  --train_data data/openreview_dataset/rm_train.json \
  --val_data data/openreview_dataset/rm_val.json \
  --output_dir models/reward_model_qwen3_8b
```

#### 多 GPU 训练 (推荐)
```bash
accelerate launch --config_file configs/accelerate_config_8gpu.yaml \
  train/code/train_reward_model.py \
  --config configs/reward_model_config.yaml \
  --train_data data/openreview_dataset/rm_train.json \
  --val_data data/openreview_dataset/rm_val.json \
  --output_dir models/reward_model_qwen3_8b
```

### 3. 评估 Reward Model

```bash
python eval/eval_reward_model.py \
  --model_path models/reward_model_qwen3_8b/best_model \
  --data_path data/openreview_dataset/rm_val.json \
  --base_model pretrained/Qwen/Qwen3-8B \
  --output eval/reward_model_results.json
```

**评估指标：**
- **Accuracy**: 正确排序的比例 (chosen_reward > rejected_reward)
- **Mean Margin**: 平均奖励差异 (chosen - rejected)
- **Margin Std**: 奖励差异的标准差

## 配置说明

### `configs/reward_model_config.yaml`

```yaml
model:
  base_model_path: pretrained/Qwen/Qwen3-8B
  sft_adapter_path: models/qwen3_8b_qlora_full_context_32k

training:
  max_length: 16384
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5          # 全量微调用更低学习率
  num_epochs: 2
  gradient_checkpointing: true
```

## 关键特性

### 1. 从 SFT 初始化
- 加载 SFT 阶段的 QLoRA adapter
- 合并 adapter 到 base model
- 保留 SFT 学到的知识

### 2. Value Head 架构
- 提取最后一个 token 的 hidden state
- 线性层映射到 scalar reward
- 输出未归一化的 logit（训练时用 sigmoid）

### 3. Pairwise Ranking Loss
```python
loss = -log sigmoid(R(chosen) - R(rejected))
```
- 优化 chosen reward > rejected reward
- 等价于二分类交叉熵

### 4. Full Fine-tuning
- 不使用量化或 LoRA
- 所有参数都可训练
- 需要 ~16GB 显存（BF16 + gradient checkpointing）

## 训练监控

### WandB
训练过程会自动记录到 WandB：
- Train/Loss: 训练损失
- Train/Accuracy: 训练准确率
- Eval/Loss: 验证损失
- Eval/Accuracy: 验证准确率
- Eval/Margin: 平均奖励差异

### 保存策略
- **best_model/**: 验证集 loss 最低的模型
- **final_model/**: 训练结束时的模型
- **checkpoint-epoch-N/**: 每 epoch 保存一次

## 硬件需求

### 推荐配置
- **GPU**: 8× A100 80GB 或 H100 80GB
- **显存**: 每个 GPU ~16GB（BF16 + gradient checkpointing）
- **训练时间**: ~2 epochs, 约 6-10 小时（取决于数据量）

### 最小配置
- **GPU**: 4× A100 80GB（需增加 gradient_accumulation_steps）
- **显存**: 每个 GPU ~20GB

## 与 DPO 的关系

| 特性 | DPO | Reward Model |
|------|-----|--------------|
| 目标 | 直接优化策略 | 学习奖励函数 |
| 模型 | Policy + Reference | 单一 Reward Model |
 Loss | DPO loss (implicit reward) | Pairwise ranking loss |
| 用途 | 模型对齐 | 质量评估、数据过滤 |

**本项目使用场景：**
1. **保留 DPO**: 继续使用 DPO 作为主要对齐方法
2. **添加 RM**: 用于评估 review 质量、过滤低质量数据

## 故障排查

### CUDA OOM
- 减小 `batch_size`
- 减小 `max_length`
- 增加 `gradient_accumulation_steps`

### 训练不稳定
- 降低 `learning_rate`（如 5e-6）
- 增加 `warmup_ratio`（如 0.2）
- 增加 `gradient_accumulation_steps`

### 准确率不高
- 检查数据质量（chosen 确实优于 rejected）
- 增加 `num_epochs`
- 调整 `learning_rate`

## 引用

本实现基于以下工作：
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [Learning to Summarize with Human Feedback](https://arxiv.org/abs/2203.02155)
