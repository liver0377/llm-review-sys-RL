# 使用生成式奖励模型进行 GRPO 训练

本指南介绍如何使用 Qwen3.5-35B-A3B 作为生成式奖励模型进行 GRPO 训练。

## 概述

我们不再单独训练奖励模型，而是直接使用 **Qwen3.5-35B-A3B** 作为生成式奖励模型，在 GRPO 训练过程中对生成的评审进行打分。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│  GRPO 训练流程                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌─────────────────┐                │
│  │   策略模型   │───► │ 生成的          │                │
│  │   (Qwen3-8B) │      │ 评审内容        │                │
│  └──────────────┘      └────────┬────────┘                │
│                                 │                          │
│                         ┌───────┴──────────┐              │
│                         │                  │              │
│                         ▼                  ▼              │
│              ┌──────────────────┐  ┌─────────────────┐   │
│              │  格式分数        │  │ 生成式奖励模型  │   │
│              │  (0-4 分)        │  │ (Qwen3.5-35B)   │   │
│              └────────┬─────────┘  └────────┬────────┘   │
│                       │                     │             │
│                       └──────────┬──────────┘             │
│                                  ▼                        │
│                        ┌─────────────────┐                │
│                        │ 组合奖励        │                │
│                        │ = format_weight │                │
│                        │   * 格式分 +    │                │
│                        │   alpha * RM分  │                │
│                        └─────────────────┘                │
│                                  │                        │
│                                  ▼                        │
│                        ┌─────────────────┐                │
│                        │ 策略更新        │                │
│                        │ (GRPO)          │                │
│                        └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 优势

1. **无需训练 RM**：跳过奖励模型训练步骤
2. **更强的 RM**：Qwen3.5-35B-A3B 比典型训练的 RM 模型大得多
3. **更好的打分**：利用模型强大的评审质量理解能力
4. **灵活性**：可以轻松切换到不同的生成式 RM

## 设置

### 1. 下载 Qwen3.5-35B-A3B 模型

```bash
# 赋予脚本执行权限
chmod +x scripts/download_qwen35_35b_a3b.sh

# 下载模型（约 70GB）
bash scripts/download_qwen35_35b_a3b.sh
```

或使用 Python 手动下载：

```bash
python scripts/download_qwen35_35b_a3b.py
```

### 2. 准备数据

确保 GRPO 数据已准备就绪：

```bash
# 准备 GRPO 训练数据
python scripts/prepare_grpo_data.py
```

### 3. 使用生成式 RM 运行 GRPO 训练

```bash
bash scripts/train_grpo.sh
```

脚本将：
1. 检查 Qwen3.5-35B-A3B 是否已下载
2. 将其用作生成式奖励模型
3. 组合格式分数和 RM 分数
4. 使用 GRPO 训练策略模型

## 配置详情

### 奖励函数

我们使用**组合奖励函数**，结合了：

1. **格式分数**（0-4 分）：评估评审结构
   - 整体质量评分
   - 评审置信度
   - 关键点部分
   - 优点和缺点
   - 改进建议
   - 评分部分

2. **RM 分数**（0-10 分）：来自 Qwen3.5-35B-A3B
   - 生成式 RM 分析评审内容
   - 基于模型的理解提供质量分数

3. **组合公式**：
   ```
   总奖励 = format_weight * 格式分数 + alpha * RM分数
   ```

   默认值：`format_weight=1.0`, `alpha=1.0`

### 训练参数

```yaml
# 策略模型
model: models/qwen3-8b-base

# 奖励模型（生成式）
reward_model: models/qwen3.5-35b-a3b
reward_model_type: qwen

# 组合奖励
reward_func: train.code.reward_function:combined_reward
alpha: 1.0              # RM 分数权重
format_weight: 1.0      # 格式分数权重

# GPU 配置
NPROC_PER_NODE: 8
CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7

# 内存优化（针对 35B RM）
gradient_accumulation_steps: 8
offload_optimizer: true
offload_model: true
vllm_gpu_memory_utilization: 0.3  # 降低以为 35B RM 腾出空间
```

## 自定义配置

### 调整奖励权重

如果你想调整格式分数与 RM 分数的权重：

```bash
# 更多权重在 RM 分数（默认）
--alpha 1.0 --format_weight 1.0

# 更多权重在格式分数
--alpha 0.5 --format_weight 2.0

# 仅使用 RM 分数
--alpha 1.0 --format_weight 0.0

# 仅使用格式分数
--alpha 0.0 --format_weight 1.0
```

### 使用不同的生成式 RM

1. 下载你喜欢的模型
2. 在 `train_grpo.sh` 中更新 `REWARD_MODEL_PATH`：

```bash
REWARD_MODEL_PATH="models/your-generative-rm"
```

3. 相应地更新 `reward_model_type`：

```bash
--reward_model_type qwen  # 或 llama, internlm 等
```

## 修改的文件

1. **scripts/train_grpo.sh**：更新 Step 4 以使用生成式 RM
2. **train/code/plugin.py**：更新 `CombinedReward` 类以正确处理 RM 分数
3. **configs/grpo_generative_rm_config.yaml**：生成式 RM 的新配置文件
4. **scripts/download_qwen35_35b_a3b.py/sh**：新的下载脚本

## 故障排除

### 内存溢出错误

如果遇到 OOM 错误：

1. 降低 `vllm_gpu_memory_utilization`（例如从 0.3 降到 0.2）
2. 增加 `gradient_accumulation_steps`（例如从 8 增到 16）
3. 为策略模型使用更少的 GPU（例如用 4 个而不是 8 个）

### 模型下载失败

1. 检查网络连接
2. 确保有足够的磁盘空间（约 80GB）
3. 如果模型需要认证，尝试使用 `--token` 参数
4. 使用 `--resume-download` 标志（默认已启用）

### 训练速度慢

使用 35B 生成式 RM 的训练比使用小型训练 RM 更慢：

1. 由于模型较大，这是预期的
2. 如果你的评审较短，考虑降低 `max_length`
3. 调整 `num_generations`（默认：4）

## 参考资料

- **MS-Swift RLHF**: [https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- **Qwen3.5-35B-A3B**: [https://huggingface.co/Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- **GRPO 论文**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03100)
