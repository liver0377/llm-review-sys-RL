# 论文评审强化学习系统 - veRL 框架实现

> 基于 veRL 框架的 GRPO 强化学习训练系统，用于优化论文评审生成模型

---

## 系统概述

本系统使用 **veRL (Volcano Engine Reinforcement Learning)** 框架实现 GRPO (Group Relative Policy Optimization) 算法，对论文评审生成模型进行强化学习训练。

### 核心架构

```
┌─────────────────────────────────────────────────────────┐
│                    veRL GRPO 训练系统                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  GPU 0-3: Policy Model (Qwen3-8B-Base)                 │
│  ├─ veRL Actor-Rollout-Ref Model                       │
│  ├─ FSDP 分布式训练                                     │
│  └─ vLLM 推理引擎                                       │
│                                                         │
│  GPU 4-7: Reward Model (Qwen3-8B-Base)                 │
│  ├─ 外部 vLLM 推理服务                                  │
│  ├─ 异步批量 Reward 计算                                │
│  └─ HTTP API 接口                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 环境准备

```bash
# 激活 veRL 环境
conda activate verl

# 验证安装
python -c "import verl; print('veRL version:', verl.__version__)"
```

### 2. 数据准备

```bash
# 转换数据为 veRL 格式（parquet）
bash scripts/prepare_data.sh

# 输出文件：
# - data/openreview_dataset/train.parquet
# - data/openreview_dataset/val.parquet
```

### 3. 启动 Reward Model 服务

```bash
# 在 GPU 4-7 上启动 RM 服务
bash scripts/start_rm_service.sh

# 检查服务状态
curl http://127.0.0.1:8002/health

# 查看日志
tail -f logs/rm_service.log
```

### 4. 启动 GRPO 训练

```bash
# 在 GPU 0-3 上启动训练
bash scripts/train_grpo.sh

# 查看训练日志
tail -f logs/grpo_training_*.log
```

---

## 文件结构

```
llm-review-sys-RL/
├── scripts/
│   ├── prepare_data.sh           # 数据预处理脚本
│   ├── start_rm_service.sh       # 启动 RM 服务
│   ├── stop_rm_service.sh        # 停止 RM 服务
│   ├── train_grpo.sh             # GRPO 训练主脚本
│   ├── run_grpo_verl.sh          # veRL GRPO 配置
│   └── data/
│       └── prepare_openreview_parquet.py  # JSON → Parquet 转换
│
├── train/code/
│   └── reward_function.py        # External RM Reward Function
│
├── data/openreview_dataset/
│   ├── grpo_train.json           # 原始训练数据
│   ├── grpo_val.json             # 原始验证数据
│   ├── train.parquet             # veRL 格式训练数据
│   └── val.parquet               # veRL 格式验证数据
│
├── models/
│   ├── Qwen3-8B-Base/           # 基础模型
│   └── grpo_qwen3_8b_verl/      # GRPO 训练输出
│
└── logs/                         # 训练日志
```

---

## GRPO 算法详解

### 核心思想

GRPO (Group Relative Policy Optimization) 是一种无需 Critic 模型的强化学习算法：

1. **多候选生成**: 对每个 prompt 生成 N 个 response
2. **组内归一化**: 在同组 response 间计算相对优势
3. **策略优化**: 使用相对优势更新策略

### 与 PPO 的区别

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic 模型 | 需要 | **不需要** |
| Value Function | 需要训练 | 无 |
| Advantage 计算 | GAE | 组内归一化 |
| 计算开销 | 高 | **低** |
| 内存占用 | 高 | **低** |

### 训练流程

```
Prompt (论文信息)
    ↓
Policy Model 生成 N 个 Responses
[Response_1, Response_2, ..., Response_N]
    ↓
External RM 计算 Rewards
[R_1, R_2, ..., R_N]
    ↓
组内归一化 Advantage
A_i = (R_i - mean(R)) / std(R)
    ↓
KL-regularized Policy Gradient
Loss = -log_prob * Advantage + β * KL(π || π_ref)
```

---

## veRL 框架特性

### 1. Hybrid Controller Programming Model

veRL 使用混合控制器编程模型，将训练和推理解耦：

```python
# Actor-Rollout-Ref Model
# - Actor: 训练和推理的策略模型
# - Rollout: vLLM 推理引擎
# - Ref: 参考模型（用于 KL 计算）

# 配置示例
actor_rollout_ref.model.path=models/Qwen3-8B-Base
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

### 2. FSDP 分布式训练

使用 PyTorch FSDP (Fully Sharded Data Parallel) 进行训练：

```bash
# FSDP 配置
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

### 3. vLLM 推理集成

使用 vLLM 作为推理引擎，提供高效生成：

```bash
# vLLM 配置
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.n=3  # 每个prompt生成3个response
```

---

## Reward 计算机制

### 奖励组成

```python
Total Reward = Format Score + RM Score

# Format Score (0-4分)
# 检查评审结构：
# - Overall Quality: 1.0
# - Key Points: 0.5
# - Strengths: 0.25
# - Weaknesses: 0.25
# - Suggestions: 0.5
# - Rating: 0.25

# RM Score (0-10分)
# 由外部 RM 服务评估内容质量
```

### External RM Reward Function

```python
# train/code/reward_function.py

async def compute_score_async(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs
):
    """
    veRL 调用的 Reward Function
    
    Args:
        solution_str: 生成的评审内容
        extra_info: 额外信息（包含 prompt）
        
    Returns:
        Dict with 'score' key
    """
    rm = ExternalReviewRM(rm_api_base="http://127.0.0.1:8002/v1")
    result = await rm.compute_reward(prompt, solution_str)
    return result
```

---

## 训练配置详解

### 核心配置参数

```bash
# GRPO 算法配置
algorithm.adv_estimator=grpo              # 使用 GRPO
algorithm.use_kl_in_reward=False          # 不在reward中加KL

# 数据配置
data.train_batch_size=64                  # 全局 batch size
data.max_prompt_length=14336              # 最大输入长度
data.max_response_length=2000             # 最大生成长度

# Actor 配置
actor_rollout_ref.actor.optim.lr=5e-7     # 学习率
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.use_kl_loss=True  # 使用 KL loss
actor_rollout_ref.actor.kl_loss_coef=0.001

# Rollout 配置
actor_rollout_ref.rollout.n=3             # 每个prompt生成3个response
actor_rollout_ref.rollout.tensor_model_parallel_size=4

# 训练配置
trainer.n_gpus_per_node=4
trainer.total_epochs=1
trainer.save_freq=50
trainer.test_freq=50
```

### 参数调优建议

1. **batch_size**: 根据 GPU 内存调整
   - 16GB GPU: `train_batch_size=32, ppo_mini_batch_size=8`
   - 24GB GPU: `train_batch_size=64, ppo_mini_batch_size=16`
   - 40GB GPU: `train_batch_size=128, ppo_mini_batch_size=32`

2. **learning_rate**: 
   - 保守: `1e-7`
   - 标准: `5e-7`
   - 激进: `1e-6`

3. **kl_loss_coef**:
   - 小: `0.0001` (更自由探索)
   - 标准: `0.001`
   - 大: `0.01` (更稳定)

---

## 性能优化

### 1. 内存优化

```bash
# Gradient Checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing=True

# Parameter Offloading
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

### 2. 速度优化

```bash
# Remove Padding (加速训练)
actor_rollout_ref.model.use_remove_padding=True

# Micro Batch Size (平衡显存和速度)
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
```

### 3. vLLM 推理优化

```bash
# GPU Memory Utilization
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# Tensor Parallel
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

---

## 监控与调试

### 训练监控

veRL 支持 WandB、TensorBoard 等日志工具：

```bash
# WandB
trainer.logger='["console","wandb"]'
trainer.project_name='verl_grpo_openreview'
trainer.experiment_name='qwen3_8b_review_grpo'
```

### 关键指标

- `actor/loss`: 策略损失
- `actor/kl_loss`: KL 散度损失
- `actor/entropy`: 策略熵
- `reward/mean`: 平均奖励
- `reward/format_score`: 格式分数
- `reward/rm_score`: RM 质量分数

### 调试技巧

1. **OOM 问题**:
   ```bash
   # 减小 batch size
   data.train_batch_size=32
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
   
   # 启用 offloading
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   ```

2. **训练不稳定**:
   ```bash
   # 降低学习率
   actor_rollout_ref.actor.optim.lr=1e-7
   
   # 增加 KL 系数
   actor_rollout_ref.actor.kl_loss_coef=0.01
   ```

3. **生成质量差**:
   ```bash
   # 调整温度
   actor_rollout_ref.rollout.temperature=0.5
   
   # 增加生成数量
   actor_rollout_ref.rollout.n=4
   ```

---

## 常见问题

### Q: veRL 和 swift 有什么区别？

A: veRL 是专门的 RL 训练框架，特性包括：
- 混合控制器编程模型
- FSDP + vLLM 无缝集成
- 生产级性能优化
- 更灵活的分布式配置

### Q: 为什么使用外部 RM 服务？

A: 优势包括：
- 独立扩展：RM 和 Policy 可以独立部署
- 资源隔离：避免 GPU 资源竞争
- 灵活性：可以使用不同规模的 RM 模型
- 可监控：独立的日志和监控

### Q: 如何调整生成质量？

A: 可以通过以下方式：
1. 调整 temperature 和 top_p
2. 修改 reward function 的评分标准
3. 调整 KL loss 系数
4. 增加训练 epochs

### Q: 训练速度慢怎么办？

A: 优化建议：
1. 增加 `ppo_micro_batch_size_per_gpu`
2. 启用 `use_remove_padding`
3. 调整 `gpu_memory_utilization`
4. 使用更大的 `tensor_model_parallel_size`

---

## 技术栈

- **框架**: veRL 0.8.0
- **模型**: Qwen3-8B-Base
- **训练引擎**: PyTorch FSDP
- **推理引擎**: vLLM
- **算法**: GRPO
- **数据格式**: Parquet

---

## 参考资料

- [veRL Documentation](https://verl.readthedocs.io/)
- [GRPO Paper](https://arxiv.org/pdf/2402.03300)
- [veRL GitHub](https://github.com/volcengine/verl)
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256)

---

## 更新日志

### 2026-03-10
- ✅ 完全迁移到 veRL 框架
- ✅ 实现 External RM Reward Function
- ✅ 优化训练配置
- ✅ 更新文档和示例