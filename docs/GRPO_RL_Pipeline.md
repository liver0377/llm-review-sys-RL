# GRPO 强化学习训练方案

## 一、方案概述

本文档描述了基于GRPO (Group Relative Policy Optimization) 的强化学习训练方案，用于优化论文评审生成模型。

### 1.1 算法选择对比

| 算法 | 需要奖励模型 | 需要价值模型 | 训练步骤 | 时间成本 |
|-----|------------|------------|---------|---------|
| DPO | 否 | 否 | 1步 | 最短 |
| GRPO | 是 | 否 | 2步 | 中等 |
| PPO | 是 | 是 | 2步 | 最长 |

**选择GRPO的原因**：
- 比PPO省去价值模型训练，降低复杂度
- 比DPO能更好地利用奖励信号进行优化
- 时间成本可控

### 1.2 整体流程

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 训练奖励模型 (约4-6小时)                          │
├─────────────────────────────────────────────────────────┤
│ 输入: DPO pairs (chosen_text, rejected_text)            │
│ 目标: 学习区分高质量/低质量评审                            │
│ 输出: Reward Model                                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: GRPO训练 (约6-10小时)                            │
├─────────────────────────────────────────────────────────┤
│ 对每个prompt:                                           │
│   1. 生成G个response (G=4-8)                            │
│   2. 计算 reward = format_score + α × rm_score          │
│   3. 组内相对比较，更新策略                              │
│ 输出: 优化后的Policy Model                              │
└─────────────────────────────────────────────────────────┘
```

---

## 二、奖励函数设计

### 2.1 组合奖励公式

```python
final_reward = format_score + α × rm_score
```

其中：
- `format_score`: 基于正则的格式分，范围 [0, 4.0]
- `rm_score`: 奖励模型输出的偏好分，范围约 [-2, 2]
- `α`: 平衡系数，建议初始值 1.0

### 2.2 格式分详细设计

评审输出应包含以下结构：

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

#### 格式检查项与分值

| 检查项 | 正则表达式 | 分值 | 说明 |
|-------|-----------|------|------|
| Overall Quality | `\*{0,2}Overall Quality:\*{0,2}\s*([1-9](?:\.[0-9])?\|10(?:\.0)?)` | +1.0 | 支持 **Overall Quality:** 或 Overall Quality: |
| Review Confidence | `\*{0,2}Review Confidence:\*{0,2}\s*([1-5](?:\.[0-9])?)` | +1.0 | 支持 **Review Confidence:** 或 Review Confidence: |
| Key Points | `### Key Points` | +0.5 | 标题存在 |
| S&W Section | `### Strengths and Weaknesses` | +0.25 | 标题存在 |
| **Strengths:** | `\*\*Strengths:\*\*` | +0.25 | 必须有加粗标记 |
| **Weaknesses:** | `\*\*Weaknesses:\*\*` | +0.25 | 必须有加粗标记 |
| Suggestions | `### Suggestions for Improvement` | +0.5 | 完整标题 |
| Rating Section | `### Rating` | +0.25 | 标题存在 |
| **满分** | | **4.0** | |

#### 格式分计算代码

```python
import re

FORMAT_PATTERNS = {
    "overall_quality": r"\*{0,2}Overall Quality:\*{0,2}\s*([1-9](?:\.[0-9])?|10(?:\.0)?)",
    "review_confidence": r"\*{0,2}Review Confidence:\*{0,2}\s*([1-5](?:\.[0-9])?)",
    "key_points": r"### Key Points",
    "strengths_weaknesses": r"### Strengths and Weaknesses",
    "strengths": r"\*\*Strengths:\*\*",
    "weaknesses": r"\*\*Weaknesses:\*\*",
    "suggestions": r"### Suggestions for Improvement",
    "rating_section": r"### Rating",
}

FORMAT_SCORES = {
    "overall_quality": 1.0,
    "review_confidence": 1.0,
    "key_points": 0.5,
    "strengths_weaknesses": 0.25,
    "strengths": 0.25,
    "weaknesses": 0.25,
    "suggestions": 0.5,
    "rating_section": 0.25,
}

def compute_format_score(response: str) -> float:
    """计算格式分，范围 [0, 4.0]"""
    total_score = 0.0
    for key, pattern in FORMAT_PATTERNS.items():
        if re.search(pattern, response, re.IGNORECASE):
            total_score += FORMAT_SCORES[key]
    return total_score
```

### 2.3 奖励模型架构

```python
class RewardModel(nn.Module):
    """
    基于SFT模型初始化的奖励模型
    
    结构:
        - Base Model (Qwen3-8B)
        - Value Head (Linear: hidden_size -> 1)
    
    输出:
        - 标量奖励值，表示对response的偏好程度
    """
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        # 获取最后一个token的hidden state
        hidden_states = self.base_model(...).last_hidden_state
        last_hidden = hidden_states[:, -1, :]
        
        # 通过value head得到奖励
        reward = self.value_head(last_hidden).squeeze(-1)
        return reward
```

### 2.4 组合奖励计算

```python
def compute_total_reward(
    response: str,
    rm_model: RewardModel,
    tokenizer,
    alpha: float = 1.0,
) -> float:
    """
    计算组合奖励
    
    Args:
        response: 生成的评审文本
        rm_model: 奖励模型
        tokenizer: 分词器
        alpha: RM分数权重
    
    Returns:
        total_reward: 组合奖励值
    """
    # 1. 格式分 (确定性计算)
    format_score = compute_format_score(response)
    
    # 2. 奖励模型分数
    with torch.no_grad():
        inputs = tokenizer(
            response, 
            return_tensors="pt",
            truncation=True,
            max_length=16384
        ).to(rm_model.device)
        rm_score = rm_model(**inputs).item()
    
    # 3. 组合
    total_reward = format_score + alpha * rm_score
    
    return total_reward
```

---

## 三、训练配置

### 3.1 奖励模型训练配置

```yaml
# configs/reward_model_v2_config.yaml

model:
  base_model_path: pretrained/Qwen/Qwen3-8B
  sft_model_path: models/qwen3_8b_full_sft_16k  # SFT后的模型
  trust_remote_code: true

training:
  max_length: 16384
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  num_train_epochs: 2
  gradient_checkpointing: true
  bf16: true

dataset:
  train_data: data/openreview_dataset/rm_train.json
  val_data: data/openreview_dataset/rm_val.json

output:
  output_dir: models/reward_model_qwen3_8b_v2

wandb:
  project: reward_model_grpo
  run_name: rm_from_sft_v2
```

### 3.2 GRPO训练配置

```yaml
# configs/grpo_config.yaml

model:
  policy_model_path: models/qwen3_8b_full_sft_16k
  ref_model_path: models/qwen3_8b_full_sft_16k  # 参考模型(frozen)
  rm_model_path: models/reward_model_qwen3_8b_v2
  trust_remote_code: true

training:
  max_length: 16384
  max_new_tokens: 2000
  
  # GRPO特定参数
  num_generations: 4  # G: 每个prompt生成的response数量
  temperature: 0.7
  top_p: 0.9
  
  # GRPO优化参数
  grpo_beta: 0.1  # KL散度系数
  alpha: 1.0      # 格式分与RM分数的平衡系数
  
  # 训练参数
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-7  # GRPO学习率要小
  weight_decay: 0.01
  warmup_ratio: 0.05
  num_train_epochs: 1
  gradient_checkpointing: true
  bf16: true
  
  # 评估与保存
  eval_strategy: steps
  eval_steps: 500
  save_steps: 500
  save_total_limit: 2

dataset:
  train_data: data/openreview_dataset/grpo_train.json
  val_data: data/openreview_dataset/grpo_val.json

output:
  output_dir: models/grpo_qwen3_8b

wandb:
  project: grpo_training
  run_name: grpo_reviewer_v1
```

---

## 四、数据格式

### 4.1 奖励模型训练数据

从DPO数据转换而来：

```json
[
  {
    "chosen_text": "### Key Points\n...\n### Rating\n**Overall Quality:** 5.0\n**Review Confidence:** 4.0",
    "rejected_text": "### Key Points\n...\n### Rating\n**Overall Quality:** 9\n**Review Confidence:** 5"
  }
]
```

### 4.2 GRPO训练数据

只需prompt，无需chosen/rejected：

```json
[
  {
    "prompt": "You are an academic paper reviewer. Please write a structured review...\n\nPaper Details:\n- Title: ...\n- Content: ..."
  }
]
```

### 4.3 数据转换脚本

```python
# scripts/convert_dpo_to_rm_data.py

import json

def convert_dpo_to_rm(dpo_data_path: str, output_path: str):
    """将DPO数据转换为RM训练数据"""
    with open(dpo_data_path, 'r') as f:
        dpo_data = json.load(f)
    
    rm_data = []
    for item in dpo_data:
        rm_data.append({
            "chosen_text": item["chosen"],
            "rejected_text": item["rejected"]
        })
    
    with open(output_path, 'w') as f:
        json.dump(rm_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(rm_data)} samples to {output_path}")

if __name__ == "__main__":
    convert_dpo_to_rm(
        "data/openreview_dataset/dpo_base_as_rejected_train_cleaned.json",
        "data/openreview_dataset/rm_train.json"
    )
    convert_dpo_to_rm(
        "data/openreview_dataset/dpo_base_as_rejected_val_cleaned.json",
        "data/openreview_dataset/rm_val.json"
    )
```

---

## 五、GRPO算法详解

### 5.1 算法核心思想

GRPO的核心是**组内相对比较**：

对于每个prompt，生成G个response，计算每个response的奖励，然后在组内进行相对比较，而非绝对评分。

### 5.2 损失函数

```
L_GRPO = -E[∑_i (r_i - r_mean) / r_std × log π_θ(y_i|x)]

其中:
- r_i = format_score_i + α × rm_score_i (组合奖励)
- r_mean, r_std = 组内奖励的均值和标准差
- π_θ(y_i|x) = 策略模型生成response y_i的概率
```

### 5.3 训练伪代码

```python
def grpo_train_step(policy_model, ref_model, rm_model, prompts, tokenizer, config):
    """GRPO单步训练"""
    
    all_rewards = []
    all_log_probs = []
    
    for prompt in prompts:
        # 1. 生成G个response
        responses = []
        log_probs = []
        
        for _ in range(config.num_generations):
            output = policy_model.generate(
                prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens
            )
            responses.append(output.text)
            log_probs.append(output.log_prob)
        
        # 2. 计算每个response的奖励
        rewards = []
        for response in responses:
            reward = compute_total_reward(
                response, rm_model, tokenizer, 
                alpha=config.alpha
            )
            rewards.append(reward)
        
        # 3. 组内标准化
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        all_rewards.extend(rewards)
        all_log_probs.extend(log_probs)
    
    # 4. 计算GRPO损失
    loss = -torch.sum(
        torch.tensor(all_rewards) * torch.stack(all_log_probs)
    )
    
    # 5. 可选: 添加KL散度约束
    kl_div = compute_kl_divergence(policy_model, ref_model, prompts)
    loss = loss + config.grpo_beta * kl_div
    
    return loss
```

---

## 六、文件结构

```
llm-review-sys/
├── train/
│   └── code/
│       ├── reward_function.py     # 奖励函数实现
│       ├── rm_plugin.py           # Reward Model 插件
│       ├── plugin.py              # GRPO 奖励函数插件
│       ├── train_reward_model.py  # 奖励模型训练脚本
│       └── train_grpo.py          # GRPO训练脚本
├── configs/
│   ├── reward_model_config.yaml   # RM配置
│   ├── grpo_config.yaml           # GRPO配置
│   └── deepspeed_zero3_config.json # DeepSpeed ZeRO-3 配置
├── scripts/
│   ├── convert_dpo_to_rm.py       # 数据转换 (DPO → RM)
│   ├── prepare_grpo_data.py       # GRPO数据准备
│   ├── train_grpo.sh              # 完整训练脚本
│   └── train_grpo_pipeline.sh     # 训练流水线
├── docs/
│   ├── GRPO_RL_Pipeline.md        # 本文档
│   └── GRPO_Training_README.md    # 训练说明
└── data/
    └── openreview_dataset/
        ├── rm_train.json          # RM训练数据 (5000条)
        ├── rm_val.json            # RM验证数据
        ├── grpo_train.json        # GRPO训练数据 (3000条)
        └── grpo_val.json          # GRPO验证数据
```

---

## 七、时间成本预估

### 7.1 硬件配置

- GPU: 8 × NVIDIA A100 80GB
- 序列长度: 16384 tokens

### 7.2 数据量配置

| 阶段 | 训练集 | 验证集 | 说明 |
|------|--------|--------|------|
| Reward Model | 5,000 条 | 1,061 条 | 从 DPO 数据随机采样 |
| GRPO | 3,000 条 | 1,061 条 | 从 SFT 数据随机采样 |

**数据量选择理由：**
1. **RM: 5,000 条** - 足以学习偏好差异，同时避免过拟合
2. **GRPO: 3,000 条** - 每个样本生成 4 个 response，计算成本可控
3. **验证集全量** - 确保评估稳定性

### 7.3 时间预估

| 阶段 | 步骤 | 时间 | 累计 |
|-----|------|------|------|
| 数据准备 | DPO → RM数据转换 | 0.5h | 0.5h |
| Step 1 | RM训练 (5000条, 2 epochs) | 3-4h | 4.5h |
| Step 2 | GRPO训练 (3000条, 1 epoch, G=4) | 4-6h | 10.5h |
| **总计** | | **~10h** | |

### 7.4 显存预估

| 阶段 | 模型 | 单卡显存 | 8卡总量 |
|-----|------|---------|--------|
| RM训练 | RM (8B) | ~35GB | 280GB |
| GRPO | Policy + Ref + RM | ~50GB | 400GB |

使用DeepSpeed ZeRO-3 + CPU offload后，80GB显存足够。

---

## 八、执行步骤

### Step 0: 环境准备

```bash
# 安装依赖
pip install ms-swift -U
pip install transformers accelerate deepspeed vllm wandb

# 登录WandB
wandb login
```

### Step 1: 数据准备

```bash
# 转换DPO数据为RM训练数据 (默认5000条)
python scripts/convert_dpo_to_rm.py \
    --max_train_samples 5000 \
    --random_sample \
    --seed 42

# 准备GRPO训练数据 (默认3000条)
python scripts/prepare_grpo_data.py \
    --max_train_samples 3000 \
    --random_sample \
    --seed 42
```

### Step 2: 训练奖励模型

```bash
# 完整训练流水线
bash scripts/train_grpo.sh

# 或单独训练RM
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type rm \
    --model models/qwen3-8b-sft \
    --dataset data/openreview_dataset/rm_train.json \
    --val_dataset data/openreview_dataset/rm_val.json \
    --output_dir models/reward_model_qwen3_8b \
    --tuner_type full \
    --deepspeed configs/deepspeed_zero3_config.json \
    --report_to wandb
```

### Step 3: GRPO训练

```bash
# GRPO训练
bash scripts/train_grpo.sh

# 或使用完整流水线 (RM + GRPO)
bash scripts/train_grpo_pipeline.sh
```

### Step 4: 评估

```bash
# 评估GRPO后的模型
python eval/eval.py \
    --model_name grpo_qwen3_8b \
    --test_data data/openreview_dataset/sft_test.json
```

---

## 九、参数调优建议

### 9.1 奖励函数参数

| 参数 | 默认值 | 调优建议 |
|-----|-------|---------|
| α (alpha) | 1.0 | 若格式问题严重，增大α；若偏好学习更重要，减小α |
| 格式分权重 | 见2.2节 | 可根据任务需求调整各项分值 |

### 9.2 GRPO参数

| 参数 | 默认值 | 调优建议 |
|-----|-------|---------|
| num_generations (G) | 4 | 增大更稳定但更慢，推荐4-8 |
| grpo_beta | 0.1 | 防止偏离参考模型太远 |
| learning_rate | 5e-7 | GRPO需要较小学习率 |
| temperature | 0.7 | 控制生成多样性 |
| alpha | 1.0 | 格式分与RM分数的平衡 |

### 9.3 训练监控指标

- `train/reward_mean`: 组内平均奖励
- `train/reward_std`: 组内奖励标准差
- `train/format_score_mean`: 平均格式分
- `train/rm_score_mean`: 平均RM分数
- `train/kl_divergence`: KL散度约束项
- `train/loss`: GRPO总损失

---

## 十、常见问题

### Q1: RM训练时chosen和rejected的奖励差距很小？

**原因**: 数据质量问题或学习率过大
**解决**: 
- 检查chosen/rejected质量差异
- 降低学习率
- 增加训练epochs

### Q2: GRPO训练后格式变差？

**原因**: RM分数主导了奖励
**解决**:
- 增大α权重
- 调整格式分计算逻辑
- 增加格式分的权重

### Q3: 生成多样性下降？

**原因**: KL约束不足
**解决**:
- 减小grpo_beta
- 提高temperature
- 增加num_generations

### Q4: 显存不足？

**原因**: 长序列 + 多模型
**解决**:
- 启用gradient_checkpointing
- 使用ZeRO-3 + CPU offload
- 减小batch_size或num_generations

---

## 十一、参考资料

1. GRPO原论文: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

2. TRL文档: https://huggingface.co/docs/trl

3. DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/

---

*文档版本: v2.0*
*最后更新: 2026-03-06*
*基于 ms-swift 框架实现*
