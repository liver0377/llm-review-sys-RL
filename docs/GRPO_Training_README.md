# GRPO 强化学习训练系统

基于 swift 框架的 GRPO (Group Relative Policy Optimization) 强化学习训练方案，用于优化论文评审生成模型。

## 目录结构

```
train/
├── code/
│   ├── reward_function.py    # 奖励函数实现
│   ├── rm_plugin.py          # Reward Model 插件
│   ├── plugin.py             # GRPO 奖励函数插件
│   ├── train_reward_model.py # RM 训练入口
│   └── train_grpo.py         # GRPO 训练入口
configs/
├── reward_model_config.yaml  # RM 训练配置
├── grpo_config.yaml          # GRPO 训练配置
└── deepspeed_zero3_config.json # DeepSpeed ZeRO-3 配置
scripts/
├── convert_dpo_to_rm.py      # DPO → RM 数据转换
├── prepare_grpo_data.py      # GRPO 数据准备
├── train_grpo.sh             # 训练脚本（推荐）
└── train_grpo_pipeline.sh    # 完整流水线
```

## 环境准备

```bash
# 安装 ms-swift
pip install ms-swift -U

# 安装其他依赖
pip install transformers accelerate deepspeed vllm wandb

# 登录 WandB（可选）
wandb login
```

## 数据准备

### 数据量配置

| 阶段 | 训练集 | 验证集 | 数据源 |
|------|--------|--------|--------|
| Reward Model | 5,000 条 | 1,052 条 | DPO (vllm as rejected) |
| GRPO | 3,000 条 | 1,061 条 | SFT 数据 |

**数据源选择：**
- **使用 vllm as rejected 数据**：rejected 格式差、质量低，chosen/rejected 差异明显
- RM 更容易学习到"什么是好评审"的偏好
- 格式分计算已兼容 vllm 数据的 Rating 格式（支持有无 `**` 标记）

**配置说明：**
- **Reward Model**: 5,000 条数据足以学习偏好，同时避免过拟合
- **GRPO**: 3,000 条 × 4 generations = 12,000 次 reward 计算，计算成本可控
- **验证集**: 使用全部验证数据，确保评估稳定性

### 1. 转换 DPO 数据为 RM 数据

```bash
# 默认配置: 5000 条训练数据，随机采样，使用 vllm as rejected
python scripts/convert_dpo_to_rm.py \
    --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
    --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
    --max_train_samples 5000 \
    --random_sample \
    --seed 42

# 或使用 base as rejected 数据
python scripts/convert_dpo_to_rm.py \
    --dpo_train data/openreview_dataset/dpo_base_as_rejected_train_cleaned.json \
    --dpo_val data/openreview_dataset/dpo_base_as_rejected_val_cleaned.json \
    --max_train_samples 5000
```

将 DPO 数据转换为 RM 训练格式，默认使用 vllm as rejected 数据（chosen/rejected 差异更明显）。

### 2. 准备 GRPO 数据

```bash
# 默认配置: 3000 条训练数据，随机采样
python scripts/prepare_grpo_data.py

# 自定义数据量
python scripts/prepare_grpo_data.py \
    --max_train_samples 3000 \
    --random_sample \
    --seed 42
```

从 SFT 数据提取 prompt 用于 GRPO 训练。

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_train_samples` | RM: 5000, GRPO: 3000 | 训练数据最大数量 |
| `--random_sample` | True | 随机采样（推荐） |
| `--seed` | 42 | 随机种子，保证可复现 |
| `--dpo_train` | dpo_vllm_as_rejected_train_cleaned.json | DPO 训练数据路径 |
| `--dpo_val` | dpo_vllm_as_rejected_val_cleaned.json | DPO 验证数据路径 |
| `--sft_train` | sft_train.json | SFT 训练数据路径 |

## 训练流程

### 完整训练流水线

```bash
bash scripts/train_grpo.sh
```

这将依次执行：
1. 数据准备
2. 奖励模型训练
3. GRPO 训练

### 分步训练

#### Step 1: 训练奖励模型

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=18579 \
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type rm \
    --model models/qwen3-8b-base \
    --dataset data/openreview_dataset/rm_train.json \
    --val_dataset data/openreview_dataset/rm_val.json \
    --output_dir models/reward_model_qwen3_8b \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --deepspeed configs/deepspeed_zero3_config.json \
    --report_to wandb
```

#### Step 2: GRPO 训练

```bash
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model models/qwen3-8b-sft \
    --dataset data/openreview_dataset/grpo_train.json \
    --output_dir models/grpo_qwen3_8b \
    --tuner_type full \
    --num_generations 4 \
    --temperature 0.7 \
    --beta 0.1 \
    --use_vllm true \
    --vllm_mode colocate \
    --reward_func train.code.plugin:review_format_reward \
    --deepspeed configs/deepspeed_zero3_config.json \
    --report_to wandb
```

## 奖励函数设计

### 格式分计算

评审输出应包含以下结构，满分 4.0 分：

| 检查项 | 正则表达式 | 分值 | 说明 |
|--------|-----------|------|------|
| Overall Quality (1-10) | `\*{0,2}Overall Quality:\*{0,2}` | 1.0 | 支持 **Overall Quality:** 或 Overall Quality: |
| Review Confidence (1-5) | `\*{0,2}Review Confidence:\*{0,2}` | 1.0 | 支持 **Review Confidence:** 或 Review Confidence: |
| ### Key Points | `### Key Points` | 0.5 | 标题存在 |
| ### Strengths and Weaknesses | `### Strengths and Weaknesses` | 0.25 | 标题存在 |
| **Strengths:** | `\*\*Strengths:\*\*` | 0.25 | 必须有加粗标记 |
| **Weaknesses:** | `\*\*Weaknesses:\*\*` | 0.25 | 必须有加粗标记 |
| ### Suggestions for Improvement | `### Suggestions for Improvement` | 0.5 | 完整标题 |
| ### Rating | `### Rating` | 0.25 | 标题存在 |
| **满分** | | **4.0** | |

**格式示例：**

```
### Key Points
[内容]

### Strengths and Weaknesses
**Strengths:**
- [内容]
**Weaknesses:**
- [内容]

### Suggestions for Improvement
[内容]

### Rating
**Overall Quality:** [1-10]
**Review Confidence:** [1-5]
```

### 组合奖励

```python
total_reward = format_score + alpha * rm_score
```

- `format_score`: 格式分 [0, 4.0]
- `rm_score`: 奖励模型分数
- `alpha`: 平衡系数 (默认 1.0)

## 配置参数

### RM 训练参数

- `learning_rate`: 1e-5
- `num_train_epochs`: 2
- `max_length`: 16384
- `deepspeed`: ZeRO-3 + CPU offload

### GRPO 训练参数

- `num_generations`: 4 (每个 prompt 生成 4 个 response)
- `temperature`: 0.7
- `beta`: 0.1 (KL 散度系数)
- `learning_rate`: 5e-7
- `use_vllm`: true (Colocate 模式)
- `vllm_gpu_memory_utilization`: 0.4

## 显存优化

针对 8× A100 80GB 的优化措施：

1. **DeepSpeed ZeRO-3 + CPU Offload**
   - 优化器状态和参数卸载到 CPU
   - 减少 GPU 显存占用

2. **vLLM Colocate 模式**
   - 训练和推理共享 GPU
   - `vllm_gpu_memory_utilization: 0.4`
   - `sleep_level: 1` 训练时释放 vLLM 内存

3. **梯度检查点**
   - `gradient_checkpointing: true`
   - 用计算换内存

## 评估模型

```bash
python eval/eval.py \
    --model_name grpo_qwen3_8b \
    --test_data data/openreview_dataset/test.json
```

## 常见问题

### 1. OOM 问题

- 减小 `vllm_gpu_memory_utilization`
- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`

### 2. 训练速度慢

- 使用 vLLM 加速推理
- 调整 `num_generations` (推荐 4-8)
- 检查 DeepSpeed 配置

### 3. 奖励模型不收敛

- 检查数据质量 (chosen vs rejected 差异)
- 调整 `learning_rate`
- 增加 `num_train_epochs`

## 参考资料

- [GRPO 论文](https://arxiv.org/abs/2402.03300)
- [ms-swift 文档](https://swift.readthedocs.io/)
- [GRPO 训练指南](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html)
