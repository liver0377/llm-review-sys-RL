# GRPO训练技术文档

**版本**: 1.0
**更新日期**: 2026-03-09
**适用框架**: ms-swift (Swift)
**目标读者**: 开发者/研究员

---

## 目录

1. [系统架构](#1-系统架构)
2. [核心概念](#2-核心概念)
3. [训练方案](#3-训练方案)
4. [技术实现详解](#4-技术实现详解)
5. [配置参考](#5-配置参考)
6. [性能优化](#6-性能优化)
7. [故障排查](#7-故障排查)
8. [参考资料](#8-参考资料)

---

## 1. 系统架构

### 1.1 整体架构

本系统基于 **ms-swift** 框架实现 GRPO (Group Relative Policy Optimization) 强化学习训练，用于优化论文评审生成模型。

```
┌──────────────────────────────────────────────────────────────────┐
│                     GRPO 训练系统架构                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐      ┌─────────────────┐                  │
│  │   策略模型       │      │  vLLM 推理引擎  │                  │
│  │   (Qwen3-8B)    │◄────►│   (Colocate)    │                  │
│  │   Policy Model  │      │                 │                  │
│  └────────┬─────────┘      └────────┬────────┘                  │
│           │                         │                            │
│           │ 生成 G 个 responses     │                            │
│           ▼                         │                            │
│  ┌──────────────────────────────────┴───────────┐                │
│  │          奖励计算 (Reward Computation)       │                │
│  ├──────────────────────┬───────────────────────┤                │
│  │                      │                       │                │
│  │  ┌──────────────┐   │   ┌──────────────────┐ │                │
│  │  │  格式分数     │   │   │  奖励模型        │ │                │
│  │  │  (0-4 分)    │   │   │  (生成式/传统)   │ │                │
│  │  │  正则匹配    │   │   │  模型推理        │ │                │
│  │  └──────────────┘   │   └────────┬─────────┘ │                │
│  │                      │             │           │                │
│  │                      │             │ (0-10 分) │                │
│  └──────────────────────┴─────────────┴───────────┘                │
│                                    │                             │
│                                    ▼                             │
│                    ┌───────────────────────────┐                  │
│                    │   组合奖励                │                  │
│                    │   = format_weight × 格式  │                  │
│                    │     + alpha × RM 分数     │                  │
│                    └───────────┬───────────────┘                  │
│                                │                                 │
│                                ▼                                 │
│                    ┌───────────────────────────┐                  │
│                    │   GRPO 优势函数计算       │                  │
│                    │   - 组内归一化            │                  │
│                    │   - 优势估计              │                  │
│                    └───────────┬───────────────┘                  │
│                                │                                 │
│                                ▼                                 │
│                    ┌───────────────────────────┐                  │
│                    │   策略更新                │                  │
│                    │   - PPO Clip              │                  │
│                    │   - 梯度累积              │                  │
│                    └───────────────────────────┘                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流

```
训练数据 (grpo_train.json)
    │
    ├─> prompts [N 个]
    │       │
    │       └─> 策略模型 (每个 prompt 生成 G 个 responses)
    │               │
    │               └─> responses [N × G 个]
    │                       │
    │                       ├─> 格式分数计算 (正则匹配)
    │                       │       └─> format_scores [N × G]
    │                       │
    │                       ├─> 奖励模型评分
    │                       │       ├─> 方案A: 生成式RM (Qwen3.5-35B)
    │                       │       └─> 方案B: 传统RM (训练的8B模型)
    │                       │       └─> rm_scores [N × G]
    │                       │
    │                       └─> 组合奖励
    │                               └─> rewards = format_weight × format_scores + alpha × rm_scores
    │
    └─> GRPO 训练
            ├─> 组内归一化 (每个 prompt 的 G 个 responses)
            ├─> 优势函数计算
            ├─> 策略损失计算
            └─> 参数更新
```

### 1.3 组件交互关系

```
┌─────────────────────────────────────────────────────────────┐
│                    Swift RLHF Trainer                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐    ┌─────────────────────────────────┐   │
│  │  GRPOTrainer  │◄───│  RewardModelPlugin              │   │
│  │               │    │  - __call__(inputs)             │   │
│  └───────┬───────┘    │  - _get_rm_score()              │   │
│          │            │  - _extract_rating()            │   │
│          │            └─────────────┬───────────────────┘   │
│          │                          │                       │
│          │            ┌─────────────▼───────────────────┐   │
│          │            │  TransformersEngine            │   │
│          │            │  - infer()                     │   │
│          │            │  - RequestConfig               │   │
│          │            └─────────────┬───────────────────┘   │
│          │                          │                       │
│          │            ┌─────────────▼───────────────────┐   │
│          └───────────►│  Reward Function               │   │
│                       │  - FormatRewardFunction        │   │
│                       │  - CombinedRewardFunction      │   │
│                       └─────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心概念

### 2.1 GRPO 算法原理

GRPO (Group Relative Policy Optimization) 是一种强化学习算法，专门用于大语言模型的对齐训练。

#### 核心思想

对于每个 prompt，生成 **G 个 responses**（通常 G=4-8），然后：

1. **计算奖励**：为每个 response 计算奖励分数
2. **组内归一化**：在同一个 prompt 的 G 个 responses 之间进行归一化
3. **优势估计**：使用归一化后的奖励作为优势估计
4. **策略更新**：使用 PPO-style clipping 更新策略

#### 数学表达

给定 prompt $x$，生成 responses $\{y_1, y_2, ..., y_G\}$：

**奖励计算**：
$$R_i = \text{format\_weight} \times \text{format\_score}(y_i) + \alpha \times \text{rm\_score}(y_i)$$

**组内归一化**：
$$\hat{R}_i = \frac{R_i - \mu}{\sigma}$$
其中 $\mu = \frac{1}{G}\sum_{j=1}^{G} R_j$，$\sigma = \sqrt{\frac{1}{G}\sum_{j=1}^{G} (R_j - \mu)^2}$

**优势函数**：
$$A_i = \hat{R}_i$$

**目标函数**：
$$L = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} A_i, \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) A_i\right)\right]$$

### 2.2 奖励机制设计

#### 2.2.1 格式分数计算 (0-4 分)

格式分数基于**正则表达式匹配**，检查评审是否包含必需的结构元素。

**评分标准**：

| 检查项 | 正则表达式 | 分值 | 说明 |
|--------|-----------|------|------|
| Overall Quality (1-10) | `\*{0,2}Overall Quality:\*{0,2}\s*([1-9](?:\.[0-9])?\|10(?:\.0)?)` | 1.0 | 支持 **Overall Quality:** 或 Overall Quality: |
| Review Confidence (1-5) | `\*{0,2}Review Confidence:\*{0,2}\s*([1-5](?:\.[0-9])?)` | 1.0 | 支持 **Review Confidence:** 或 Review Confidence: |
| ### Key Points | `### Key Points` | 0.5 | 标题存在 |
| ### Strengths and Weaknesses | `### Strengths and Weaknesses` | 0.25 | 标题存在 |
| **Strengths:** | `\*\*Strengths:\*\*` | 0.25 | 必须有加粗标记 |
| **Weaknesses:** | `\*\*Weaknesses:\*\*` | 0.25 | 必须有加粗标记 |
| ### Suggestions for Improvement | `### Suggestions for Improvement` | 0.5 | 完整标题 |
| ### Rating | `### Rating` | 0.25 | 标题存在 |
| **满分** | | **4.0** | |

**标准格式示例**：

```markdown
### Key Points
- The paper presents a novel approach to X
- The methodology is sound but lacks Y

### Strengths and Weaknesses
**Strengths:**
- Clear problem formulation
- Strong experimental results

**Weaknesses:**
- Limited discussion of related work
- Missing ablation study

### Suggestions for Improvement
- Add comparison with method Z
- Include more diverse test cases

### Rating
**Overall Quality:** 8.5
**Review Confidence:** 4.0
```

**代码实现**：`train/code/reward_function.py:28-33`

```python
def compute_format_score(response: str) -> float:
    total_score = 0.0
    for key, pattern in FORMAT_PATTERNS.items():
        if re.search(pattern, response, re.IGNORECASE):
            total_score += FORMAT_SCORES[key]
    return total_score
```

#### 2.2.2 奖励模型分数 (0-10 分)

奖励模型分数来自两种方案：

**方案A：生成式 RM (Qwen3.5-35B-A3B)**
- 大语言模型直接生成评审和评分
- 评分范围：0-10
- 质量高，但推理速度慢

**方案B：传统训练的 RM (8B)**
- 带有分类头的模型
- 直接输出标量分数
- 推理快，但需要训练

#### 2.2.3 组合奖励公式

**最终奖励**：
$$\text{total\_reward} = \text{format\_weight} \times \text{format\_score} + \alpha \times \text{rm\_score}$$

**参数说明**：
- `format_weight`: 格式分数权重（默认 1.0）
- `alpha`: RM 分数权重（默认 1.0）
- `format_score`: 格式分数 [0, 4.0]
- `rm_score`: RM 分数 [0, 10.0]

**推荐配置**：
```bash
--format_weight 1.0  # 格式分权重
--alpha 1.0          # RM 分数权重
```

**调整建议**：
- 如果评审结构完整但质量差：降低 `format_weight`，提高 `alpha`
- 如果评审质量高但结构不完整：提高 `format_weight`，降低 `alpha`

### 2.3 Swift 框架的 Plugin 系统

Swift 框架通过 **Plugin 系统** 实现灵活的奖励模型集成。

#### Plugin 架构

```
DefaultRMPlugin (基类)
    │
    ├─── GenRMPlugin (内置)
    │       └─> 生成式 RM，使用 TransformersEngine
    │
    └─── CustomPlugin (自定义)
            └─> 用户自定义的奖励逻辑
```

#### Plugin 接口

**必需实现的方法**：

```python
class DefaultRMPlugin:
    def __init__(self, model, template):
        """
        初始化插件

        Args:
            model: 奖励模型
            template: 模板 (用于模型输入格式化)
        """
        self.model = model
        self.template = template

    def __call__(self, inputs: List[Dict], **kwargs) -> List[float]:
        """
        计算奖励分数

        Args:
            inputs: 输入列表，每个元素包含：
                - messages: 对话消息列表
                - 其他数据集字段

        Returns:
            List[float]: 奖励分数列表
        """
        pass
```

**输入格式**：
```python
inputs = [
    {
        'messages': [
            {'role': 'user', 'content': 'Please review...'},
            {'role': 'assistant', 'content': 'Generated review...'}
        ],
        'solution': 'abc',  # 其他数据集字段
    },
    # ... 更多样本
]
```

#### Plugin 注册方式

**方式1：使用 `--reward_model_plugin`**
```bash
swift rlhf \
    --reward_model_plugin train.code.genrm_plugin:get_review_genrm_plugin \
    --external_plugins train/code/genrm_plugin.py
```

**方式2：使用 `--reward_func`**
```bash
swift rlhf \
    --reward_func train.code.reward_function:combined_reward
```

**区别**：
- `--reward_model_plugin`: 用于加载完整的奖励模型（带模型实例）
- `--reward_func`: 用于简单的奖励函数（不需要模型实例）

---

## 3. 训练方案

本系统提供 **两种 GRPO 训练方案**：

### 3.1 方案概览对比表

| 特性 | 方案A：生成式 RM | 方案B：传统训练 RM |
|------|-----------------|-------------------|
| **推荐程度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **奖励模型** | Qwen3.5-35B-A3B (预训练) | 训练的 8B RM |
| **需要训练 RM** | ❌ 不需要 | ✅ 需要 (4-6小时) |
| **RM 质量** | 高 (35B 强大理解能力) | 中 (8B 专门训练) |
| **推理速度** | 慢 (生成文本) | 快 (直接输出分数) |
| **显存占用** | 高 (35B + 8B = ~120GB) | 低 (8B + 8B = ~80GB) |
| **总训练时间** | 短 (无 RM 训练) | 长 (含 RM 训练) |
| **适用场景** | 追求最佳质量，显存充足 | 显存受限，需要快速迭代 |
| **实现方式** | 内部插件 / 外部部署 | 传统 RM 训练 |
| **脚本名称** | `train_grpo_GRM.sh` | `train_grpo_pipeline.sh` |

**推荐选择**：
- ✅ **首选方案A**：质量优先，显存充足
- ⚠️ **备选方案B**：显存受限，或需要快速迭代

---

### 3.2 方案A：生成式 RM（推荐）

#### 3.2.1 架构设计

生成式奖励模型直接使用强大的大语言模型（Qwen3.5-35B-A3B）对生成的评审进行质量评估。

**核心思想**：
- 利用大模型的强大理解和推理能力
- 通过精心设计的 prompt 引导模型生成评审和评分
- 从生成的文本中提取评分作为奖励信号

**流程图**：

```
┌─────────────────────────────────────────────────────────────┐
│                   生成式 RM 打分流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  输入：生成的评审文本 (review_text)                            │
│          │                                                    │
│          ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Step 1: 构造 Prompt                                 │     │
│  │  - 原始任务 (论文信息)                               │     │
│  │  - 生成的评审内容                                    │     │
│  │  - System Prompt (评分标准和格式要求)                │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Step 2: 调用生成式 RM (Qwen3.5-35B-A3B)            │     │
│  │  - 输入：构造的 Prompt                               │     │
│  │  - 输出：评审分析和评分 (文本格式)                    │     │
│  │  示例输出：                                          │     │
│  │    "这篇评审整体质量良好...                           │     │
│  │     Rating: 8.5"                                     │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Step 3: 提取评分                                    │     │
            │  │  - 正则表达式匹配 "Overall Quality: X.X"              │     │
│  │  - 支持多种格式 (评分：X.X, 分数：X.X)                │     │
│  │  - 如果提取失败，返回默认值 5.0                       │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │                                    │
│                           ▼                                    │
│                     输出：RM 分数 (0-10)                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**System Prompt 示例**：

```python
system_prompt = """
You are an academic paper reviewer evaluating a generated review. Your task is to assess the quality of the given review based on its clarity, completeness, accuracy, and constructiveness.

Provide a brief assessment (2-3 sentences) explaining your evaluation, then give a score.

Your response should follow this format:
<Your brief assessment here>

**Overall Quality:** X.X

Scoring guidelines:
- 9.0-10.0: Excellent - comprehensive, insightful, well-structured
- 7.0-8.9: Good - solid analysis with minor gaps
- 5.0-6.9: Average - adequate but lacks depth or clarity
- 3.0-4.9: Poor - significant issues or incomplete
- 0.0-2.9: Very Poor - fails to meet basic requirements

**Overall Quality:** (1–10, where 10 is top-tier)
"""
```

#### 3.2.2 内部插件方式实现

**优点**：
- ✅ 集成简单，无需额外服务
- ✅ 易于调试和修改
- ✅ 所有代码在一个进程中

**缺点**：
- ❌ 推理速度慢（35B 模型生成文本）
- ❌ 显存占用高（同时加载 8B + 35B）
- ⚠️ Swift 文档警告：大模型会很慢

**适用场景**：
- 显存充足（8× A100 80GB）
- 训练时间不敏感
- 需要频繁调试和修改

**完整代码实现**：

**文件：`train/code/genrm_plugin.py`**

```python
"""
生成式奖励模型插件
用于使用 Qwen3.5-35B-A3B 对评审质量进行打分
"""

import re
import textwrap
import torch
from copy import deepcopy
from typing import List, Dict, Optional

from swift.plugin import DefaultRMPlugin
from swift.infer_engine import TransformersEngine, RequestConfig
from swift.utils import get_logger

logger = get_logger()


class ReviewGenRMPlugin(DefaultRMPlugin):
    """
    生成式奖励模型插件 - 评审质量打分

    使用大语言模型（Qwen3.5-35B-A3B）对生成的评审进行质量评估，
    并组合格式分数和 RM 分数作为最终奖励。

    Args:
        model: 奖励模型
        template: 模板
        alpha (float): RM 分数的权重 (默认: 1.0)
        format_weight (float): 格式分数的权重 (默认: 1.0)
        max_tokens (int): 生成的最大 token 数 (默认: 512)
        temperature (float): 生成温度 (默认: 0.1，更确定性)
    """

    def __init__(
        self,
        model,
        template,
        alpha: float = 1.0,
        format_weight: float = 1.0,
        max_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(model, template, **kwargs)
        self.alpha = alpha
        self.format_weight = format_weight
        self.max_tokens = max_tokens
        self.temperature = temperature

        # 初始化推理引擎
        self.engine = TransformersEngine(
            self.model,
            template=self.template,
            max_batch_size=0  # 0 表示无限制
        )

        # 配置生成参数
        self.request_config = RequestConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )

        # 导入格式评分函数
        self._import_modules()

        # 设置评审打分的 system prompt
        self.system_prompt = textwrap.dedent("""
            You are an academic paper reviewer evaluating a generated review. Your task is to assess the quality of the given review based on its clarity, completeness, accuracy, and constructiveness.

            Provide a brief assessment (2-3 sentences) explaining your evaluation, then give a score.

            Your response should follow this format:
            <Your brief assessment here>

            **Overall Quality:** X.X

            Scoring guidelines:
            - 9.0-10.0: Excellent - comprehensive, insightful, well-structured
            - 7.0-8.9: Good - solid analysis with minor gaps
            - 5.0-6.9: Average - adequate but lacks depth or clarity
            - 3.0-4.9: Poor - significant issues or incomplete
            - 0.0-2.9: Very Poor - fails to meet basic requirements

            **Overall Quality:** (1–10, where 10 is top-tier)
        """).strip()

    def _import_modules(self):
        """动态导入依赖模块"""
        import sys
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        from reward_function import compute_format_score
        self.compute_format_score = compute_format_score

    def __call__(self, inputs: List[Dict], **kwargs) -> List[float]:
        """
        计算奖励分数

        Args:
            inputs: 输入列表，每个元素包含：
                - messages: 对话消息列表
                - 其他数据集字段

        Returns:
            List[float]: 奖励分数列表
        """
        rewards = []

        for item in inputs:
            # 提取生成的评审内容
            messages = item.get("messages", [])
            completion = self._extract_completion(messages)

            # 计算格式分数 (0-4)
            format_score = self.compute_format_score(completion)

            # 使用生成式 RM 获取质量分数 (0-10)
            rm_score = self._get_genrm_score(messages, completion)

            # 组合奖励
            total_reward = (
                self.format_weight * format_score +
                self.alpha * rm_score
            )

            rewards.append(total_reward)

            logger.debug(
                f"Format: {format_score:.2f}, RM: {rm_score:.2f}, "
                f"Total: {total_reward:.2f}"
            )

        return rewards

    def _extract_completion(self, messages: List[Dict]) -> str:
        """从消息中提取模型生成的评审内容"""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _get_genrm_score(self, messages: List[Dict], completion: str) -> float:
        """
        使用生成式奖励模型获取质量分数

        Args:
            messages: 原始对话消息
            completion: 生成的评审内容

        Returns:
            float: 质量分数 (0-10)
        """
        # 构造评审打分的 prompt
        rm_messages = self._prepare_rm_prompt(messages, completion)

        try:
            # 调用生成式 RM
            results = self.engine.infer(
                [rm_messages],
                self.request_config,
                use_tqdm=False
            )

            if results and len(results) > 0:
                response = results[0].choices[0].message.content.strip()

                # 从响应中提取评分
                score = self._extract_rating(response)

                if score is not None:
                    logger.debug(f"GenRM response: {response[:100]}...")
                    logger.debug(f"Extracted score: {score}")
                    return score
                else:
                    logger.warning(
                        f"Failed to extract rating from response: {response[:200]}"
                    )
                    return 5.0  # 默认中等分数
            else:
                logger.warning("Empty response from GenRM")
                return 5.0

        except Exception as e:
            logger.error(f"Error calling GenRM: {e}")
            return 5.0

    def _prepare_rm_prompt(self, original_messages: List[Dict], completion: str) -> List[Dict]:
        """
        准备给生成式 RM 的 prompt

        Args:
            original_messages: 原始对话消息
            completion: 生成的评审内容

        Returns:
            List[Dict]: 给 RM 的消息列表
        """
        # 提取原始任务（论文信息）
        query = ""
        for msg in original_messages:
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        # 构造评审内容描述
        review_content = textwrap.dedent(f"""
            [生成的评审内容]
            {completion}
            [/生成的评审内容]
        """).strip()

        # 组合成完整的 prompt
        user_prompt = textwrap.dedent(f"""
            [原始任务]
            {query}
            [/原始任务]

            {review_content}

            请根据上述评审内容给出质量评分。
        """).strip()

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _extract_rating(self, response: str) -> Optional[float]:
        """
        从模型响应中提取评分

        支持多种格式：
        - Rating: 8.5
        - rating: 8.5
        - 评分：8.5
        - 8.5
        """
        # 模式 1: "Rating: X.X" 或 "rating: X.X"
        pattern1 = r'[Rr]ating:\s*([0-9](?:\.[0-9])?|10(?:\.0)?)'
        match = re.search(pattern1, response)
        if match:
            return float(match.group(1))

        # 模式 2: "评分：X.X" 或 "分数：X.X"
        pattern2 = r'[评分分数][：:]\s*([0-9](?:\.[0-9])?|10(?:\.0)?)'
        match = re.search(pattern2, response)
        if match:
            return float(match.group(1))

        # 模式 3: 直接提取最后出现的 0-10 之间的数字
        pattern3 = r'(?:^|[\s\n])([0-9](?:\.[0-9])?|10(?:\.0)?)(?:[\s\n]|$)'
        matches = re.findall(pattern3, response)
        if matches:
            # 取最后一个匹配
            last_score = float(matches[-1])
            if 0 <= last_score <= 10:
                return last_score

        # 如果都没找到，尝试在最后一行查找
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # 尝试提取数字
            numbers = re.findall(r'[0-9]+\.?[0-9]*', last_line)
            if numbers:
                score = float(numbers[-1])
                if 0 <= score <= 10:
                    return score

        return None


def get_review_genrm_plugin(model=None, template=None, **kwargs):
    """获取插件实例的函数"""
    return ReviewGenRMPlugin(model=model, template=template, **kwargs)
```

**训练脚本**：

**文件：`scripts/train_grpo_GRM.sh`**

```bash
#!/bin/bash
set -e

echo "========================================"
echo "GRPO Training with Generative Reward Model"
echo "Using Qwen3.5-35B-A3B as Generative RM"
echo "========================================"

cd /data/wudy/RL/llm-review-sys-RL

echo ""
echo "[Step 1] Converting DPO data to RM format..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/rm_train.json" ]; then
    python scripts/convert_dpo_to_rm.py \
        --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
        --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
        --max_train_samples 5000 \
        --random_sample \
        --seed 42
else
    echo "RM data exists, skipping..."
fi

echo ""
echo "[Step 2] Preparing GRPO data..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/grpo_train.json" ]; then
    python scripts/prepare_grpo_data.py \
        --max_train_samples 3000 \
        --random_sample \
        --seed 42
else
    echo "GRPO data exists, skipping..."
fi

echo ""
echo "[Step 3] Checking Generative Reward Model..."
echo "----------------------------------------"

REWARD_MODEL_PATH="models/qwen3.5-35b-a3b"

if [ ! -d "${REWARD_MODEL_PATH}" ]; then
    echo "Error: Generative Reward Model not found at ${REWARD_MODEL_PATH}"
    echo ""
    echo "Please download it first using:"
    echo "  bash scripts/download_qwen35_35b_a3b.sh"
    echo ""
    echo "Or manually:"
    echo "  python scripts/download_qwen35_35b_a3b.py"
    exit 1
fi

echo "✓ Generative Reward Model found: ${REWARD_MODEL_PATH}"
echo "  Model: Qwen3.5-35B-A3B"
echo "  Type: Generative Reward Model"
echo ""

echo ""
echo "[Step 4] Starting GRPO Training..."
echo "----------------------------------------"

echo "Configuration:"
echo "  Policy Model:        models/qwen3-8b-base"
echo "  Generative RM:       ${REWARD_MODEL_PATH}"
echo "  Reward Function:     Combined (RM + Format Score)"
echo "  Alpha (RM weight):   1.0"
echo "  Format Weight:       1.0"
echo "  GPUs:                8 (CUDA:0-7)"
echo "  Batch Size:          1 x 8 = 8 (accumulation)"
echo "  Max Length:          16384"
echo ""

WANDB_PROJECT=grpo_training \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=18579 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model models/qwen3-8b-base \
    --reward_model ${REWARD_MODEL_PATH} \
    --reward_model_plugin train.code.genrm_plugin:get_review_genrm_plugin \
    --external_plugins train/code/genrm_plugin.py \
    --alpha 1.0 \
    --format_weight 1.0 \
    --dataset data/openreview_dataset/grpo_train.json \
    --val_dataset data/openreview_dataset/grpo_val.json \
    --output_dir models/grpo_qwen3_8b_grm \
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
    --num_generations 4 \
    --temperature 0.7 \
    --top_p 0.9 \
    --beta 0.1 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 16384 \
    --vllm_enforce_eager true \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --report_to wandb \
    --run_name grpo_qwen3_8b_grm_v1

echo ""
echo "========================================"
echo "GRPO Training Completed!"
echo "========================================"
echo ""
echo "Models:"
echo "  ✓ Policy Model:      models/grpo_qwen3_8b_grm"
echo "  ✓ Generative RM:     ${REWARD_MODEL_PATH} (unchanged)"
echo ""
echo "Reward Configuration:"
echo "  - Type:              Generative Reward Model (Qwen3.5-35B-A3B)"
echo "  - Format Score:      0-4 points"
echo "  - RM Score:          0-10 points"
echo "  - Combined Formula:  format_weight * format + alpha * rm_score"
echo "  - Weights:           format_weight=1.0, alpha=1.0"
echo ""
echo "To evaluate the model:"
echo "  python eval/eval.py --model_name grpo_qwen3_8b_grm"
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|---|------|
| `--reward_model` | `models/qwen3.5-35b-a3b` | 生成式 RM 路径 |
| `--reward_model_plugin` | `train.code.genrm_plugin:get_review_genrm_plugin` | Plugin 获取函数 |
| `--external_plugins` | `train/code/genrm_plugin.py` | Plugin 文件路径 |
| `--alpha` | 1.0 | RM 分数权重 |
| `--format_weight` | 1.0 | 格式分数权重 |
| `--num_generations` | 4 | 每个 prompt 生成 4 个 responses |
| `--temperature` | 0.7 | 生成温度 |
| `--beta` | 0.1 | KL 散度系数 |

#### 3.2.3 外部部署方式实现

**优点**：
- ✅ 推理速度快（vLLM 加速）
- ✅ 可扩展性强（多机部署）
- ✅ 不占用训练 GPU

**缺点**：
- ❌ 需要额外部署服务
- ❌ 网络通信开销
- ❌ 资源需求更高（额外 GPU）

**适用场景**：
- 需要快速推理
- 多机环境
- 训练资源紧张

**部署步骤**：

**Step 1: 启动 vLLM 服务**

```bash
# 在单独的 GPU 上启动服务（例如 GPU 0-3）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
    --model models/qwen3.5-35b-a3b \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

**Step 2: 修改插件使用 OpenAI 客户端**

```python
from openai import OpenAI

class ReviewGenRMPluginExternal(DefaultRMPlugin):
    def __init__(self, model=None, template=None, **kwargs):
        super().__init__(model, template, **kwargs)
        self.client = OpenAI(
            api_key='EMPTY',
            base_url='http://127.0.0.1:8000/v1',
        )

    def _get_genrm_score(self, messages, completion):
        rm_messages = self._prepare_rm_prompt(messages, completion)

        try:
            response = self.client.chat.completions.create(
                model='qwen3.5-35b-a3b',
                messages=rm_messages,
                max_tokens=512,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()
            score = self._extract_rating(content)
            return score if score is not None else 5.0

        except Exception as e:
            logger.error(f"Error calling external GenRM: {e}")
            return 5.0
```

**Step 3: 训练命令**

```bash
swift rlhf \
    --rlhf_type grpo \
    --model models/qwen3-8b-base \
    --reward_model_plugin train.code.genrm_plugin_external:get_review_genrm_plugin \
    --external_plugins train/code/genrm_plugin_external.py \
    # ... 其他参数
```

#### 3.2.4 性能对比

| 指标 | 内部插件 | 外部部署 |
|------|---------|---------|
| 单次推理时间 | ~5-10 秒 | ~1-2 秒 |
| 显存占用 | 高 (同进程) | 低 (独立进程) |
| 实现复杂度 | 低 | 中 |
| 调试难度 | 低 | 中 |
| 推荐度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

### 3.3 方案B：传统训练 RM

#### 3.3.1 训练流程

传统方案需要先训练一个带分类头的奖励模型（8B），然后用于 GRPO 训练。

**流程图**：

```
┌─────────────────────────────────────────────────────────────┐
│                  传统 RM 训练流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Phase 1: Reward Model Training (4-6 小时)          │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │                                                       │     │
│  │  输入：DPO 数据 (chosen, rejected pairs)              │     │
│  │          │                                            │     │
│  │          ▼                                            │     │
│  │  ┌─────────────────────────────────────────────┐     │     │
│  │  │  训练目标：学习区分高质量和低质量评审          │     │     │
│  │  │  Loss: -log(sigmoid(rm_score(chosen) -      │     │     │
│  │  │                    rm_score(rejected)))      │     │     │
│  │  └─────────────────────────────────────────────┘     │     │
│  │                                                       │     │
│  │  输出：训练好的 RM (8B, 带分类头)                    │     │
│  └─────────────────────────────────────────────────────┘     │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Phase 2: GRPO Training (6-10 小时)                 │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │                                                       │     │
│  │  使用训练好的 RM 为生成的评审打分                     │     │
│  │  RM 直接输出标量分数 (无需生成文本)                    │     │
│  │                                                       │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**完整代码实现**：

**文件：`scripts/train_grpo_pipeline.sh`**

```bash
#!/bin/bash
set -e

echo "========================================"
echo "GRPO Training Pipeline"
echo "Traditional Reward Model + GRPO"
echo "========================================"

cd /data/wudy/RL/llm-review-sys-RL

echo ""
echo "[Step 1] Converting DPO data to RM format..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/rm_train.json" ]; then
    python scripts/convert_dpo_to_rm.py \
        --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
        --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
        --max_train_samples 5000 \
        --random_sample \
        --seed 42
else
    echo "RM data exists, skipping..."
fi

echo ""
echo "[Step 2] Preparing GRPO data..."
echo "----------------------------------------"

if [ ! -f "data/openreview_dataset/grpo_train.json" ]; then
    python scripts/prepare_grpo_data.py \
        --max_train_samples 3000 \
        --random_sample \
        --seed 42
else
    echo "GRPO data exists, skipping..."
fi

echo ""
echo "[Step 3] Training Reward Model..."
echo "----------------------------------------"

if [ ! -d "models/reward_model_qwen3_8b" ]; then
    WANDB_PROJECT=reward_model_grpo MASTER_ADDR=127.0.0.1 MASTER_PORT=18579 NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    swift rlhf \
        --rlhf_type rm \
        --model models/qwen3-8b-base \
        --dataset data/openreview_dataset/rm_train.json \
        --val_dataset data/openreview_dataset/rm_val.json \
        --custom_dataset_info configs/custom_dataset_info.json \
        --output_dir models/reward_model_qwen3_8b \
        --tuner_type full \
        --torch_dtype bfloat16 \
        --max_length 4096 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --num_train_epochs 2 \
        --gradient_checkpointing true \
        --bf16 true \
        --eval_strategy steps \
        --eval_steps 100 \
        --save_steps 200 \
        --save_total_limit 2 \
        --logging_steps 10 \
        --deepspeed configs/deepspeed_zero2_config.json \
        --beta 0.1 \
        --report_to wandb \
        --run_name rm_qwen3_8b_v1

    echo "Reward Model training completed!"
else
    echo "Reward Model exists, skipping training..."
fi

echo ""
echo "[Step 4] Starting GRPO Training..."
echo "----------------------------------------"

echo "Configuration:"
echo "  Policy Model:        models/qwen3-8b-sft"
echo "  Reward Model:        models/reward_model_qwen3_8b"
echo "  Reward Function:     Combined (RM + Format Score)"
echo "  Alpha (RM weight):   1.0"
echo "  Format Weight:       1.0"
echo "  GPUs:                8 (CUDA:0-7)"
echo "  Batch Size:          1 x 8 = 8 (accumulation)"
echo "  Max Length:          16384"
echo ""

WANDB_PROJECT=grpo_training \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=18579 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model models/qwen3-8b-sft \
    --reward_model models/reward_model_qwen3_8b \
    --reward_model_type default \
    --dataset data/openreview_dataset/grpo_train.json \
    --val_dataset data/openreview_dataset/grpo_val.json \
    --output_dir models/grpo_qwen3_8b \
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
    --num_generations 4 \
    --temperature 0.7 \
    --top_p 0.9 \
    --beta 0.1 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 16384 \
    --vllm_enforce_eager true \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --reward_func train.code.reward_function:combined_reward \
    --alpha 1.0 \
    --format_weight 1.0 \
    --report_to wandb \
    --run_name grpo_qwen3_8b_v1

echo ""
echo "========================================"
echo "GRPO Training Completed!"
echo "========================================"
echo ""
echo "Models:"
echo "  ✓ Reward Model:       models/reward_model_qwen3_8b"
echo "  ✓ Policy Model:       models/grpo_qwen3_8b"
echo ""
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|---|------|
| `--rlhf_type` | `rm` | 训练奖励模型 |
| `--reward_model_type` | `default` | 使用默认的 ORM 处理逻辑 |
| `--max_length` | 4096 (RM) / 16384 (GRPO) | RM 使用更短的上下文 |
| `--learning_rate` | 1e-5 (RM) / 5e-7 (GRPO) | GRPO 使用更小的学习率 |
| `--num_train_epochs` | 2 (RM) / 1 (GRPO) | RM 训练 2 个 epoch |

---

## 4. 技术实现详解

### 4.1 插件系统架构

Swift 的 Plugin 系统基于 `DefaultRMPlugin` 基类，提供了灵活的奖励模型集成方式。

**类继承结构**：

```
DefaultRMPlugin (swift.plugin.DefaultRMPlugin)
    │
    ├─── GenRMPlugin (swift 内置)
    │       └─> 生成式 RM 实现
    │
    └─── ReviewGenRMPlugin (自定义)
            └─> 本项目的生成式 RM 实现
```

**关键方法**：

| 方法 | 功能 | 必须实现 |
|------|------|---------|
| `__init__()` | 初始化插件，加载模型和引擎 | ✅ |
| `__call__()` | 计算奖励分数 | ✅ |
| `_prepare_rm_prompt()` | 构造 prompt | ❌ (可选) |
| `_extract_rating()` | 提取评分 | ❌ (可选) |

### 4.2 RewardModelPlugin 实现

**初始化流程**：

```python
def __init__(self, model, template, alpha=1.0, format_weight=1.0, **kwargs):
    # 1. 调用父类初始化
    super().__init__(model, template, **kwargs)

    # 2. 保存参数
    self.alpha = alpha
    self.format_weight = format_weight

    # 3. 初始化推理引擎
    self.engine = TransformersEngine(
        self.model,
        template=self.template,
        max_batch_size=0
    )

    # 4. 配置生成参数
    self.request_config = RequestConfig(
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
    )

    # 5. 导入依赖模块
    self._import_modules()
```

**调用流程**：

```python
def __call__(self, inputs: List[Dict], **kwargs) -> List[float]:
    """
    Swift 训练时的调用流程：

    1. GRPOTrainer 生成 responses
    2. 构造 inputs = [
           {
               'messages': [
                   {'role': 'user', 'content': 'prompt...'},
                   {'role': 'assistant', 'content': 'response...'}
               ]
           },
           # ...
       ]
    3. 调用 plugin(inputs)
    4. 返回 rewards = [r1, r2, ...]
    5. GRPOTrainer 使用 rewards 计算优势函数
    """
    rewards = []

    for item in inputs:
        # Step 1: 提取生成的评审
        messages = item.get("messages", [])
        completion = self._extract_completion(messages)

        # Step 2: 计算格式分数
        format_score = self.compute_format_score(completion)

        # Step 3: 调用生成式 RM
        rm_score = self._get_genrm_score(messages, completion)

        # Step 4: 组合奖励
        total_reward = (
            self.format_weight * format_score +
            self.alpha * rm_score
        )

        rewards.append(total_reward)

    return rewards
```

### 4.3 数据准备流程

**步骤1：转换 DPO 数据为 RM 数据**

**文件：`scripts/convert_dpo_to_rm.py`**

```python
def convert_dpo_to_rm(
    dpo_data_path: str,
    output_path: str,
    max_samples: int = None,
    random_sample: bool = True,
    seed: int = 42,
):
    """
    将 DPO 格式转换为 RM 训练格式

    DPO 格式：
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }

    RM 格式：
    {
        "query": "...",
        "chosen": "...",
        "rejected": "..."
    }
    """
    with open(dpo_data_path, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)

    rm_data = []
    for item in dpo_data:
        # 只提取 instruction 作为 query
        prompt = item.get("prompt", "")
        query = extract_instruction_only(prompt)

        rm_item = {
            "query": query,
            "chosen": item.get("chosen", ""),
            "rejected": item.get("rejected", ""),
        }
        if rm_item["query"] and rm_item["chosen"] and rm_item["rejected"]:
            rm_data.append(rm_item)

    # 随机采样
    if max_samples and max_samples > 0 and max_samples < len(rm_data):
        if random_sample:
            random.seed(seed)
            rm_data = random.sample(rm_data, max_samples)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rm_data, f, ensure_ascii=False, indent=2)
```

**步骤2：准备 GRPO 数据**

**文件：`scripts/prepare_grpo_data.py`**

```python
def prepare_grpo_data(
    sft_data_path: str,
    output_path: str,
    max_samples: int = None,
    random_sample: bool = True,
    seed: int = 42,
):
    """
    从 SFT 数据提取 prompts 用于 GRPO 训练

    SFT 格式：
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }

    GRPO 格式：
    {
        "messages": [
            {"role": "user", "content": "prompt"}
        ],
        "reference": "..."  # 可选
    }
    """
    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    grpo_data = []
    for item in sft_data:
        if isinstance(item, dict):
            # 构造 prompt
            if "input" in item and item["input"]:
                prompt = f"{item.get('instruction', '')}\n\n{item.get('input', '')}"
            else:
                prompt = item.get("instruction", "") or item.get("prompt", "")

            if prompt:
                grpo_item = {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                if "output" in item:
                    grpo_item["reference"] = item["output"]
                grpo_data.append(grpo_item)

    # 随机采样
    if max_samples and max_samples > 0 and max_samples < len(grpo_data):
        if random_sample:
            random.seed(seed)
            grpo_data = random.sample(grpo_data, max_samples)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grpo_data, f, ensure_ascii=False, indent=2)
```

### 4.4 训练配置详解

**GRPO 训练的关键参数**：

| 参数类别 | 参数名 | 默认值 | 说明 |
|---------|--------|--------|------|
| **生成参数** | `num_generations` | 4 | 每个 prompt 生成的 response 数量 |
| | `temperature` | 0.7 | 生成温度 |
| | `top_p` | 0.9 | nucleus sampling 参数 |
| **RLHF 参数** | `beta` | 0.1 | KL 散度系数 |
| | `alpha` | 1.0 | RM 分数权重 |
| | `format_weight` | 1.0 | 格式分数权重 |
| **训练参数** | `learning_rate` | 5e-7 | 学习率（很小） |
| | `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| | `per_device_train_batch_size` | 1 | 每设备 batch size |
| **vLLM 参数** | `use_vllm` | true | 使用 vLLM 加速 |
| | `vllm_mode` | colocate | 与训练同进程 |
| | `vllm_gpu_memory_utilization` | 0.3 | vLLM 显存占用比例 |
| | `sleep_level` | 1 | 训练时释放 vLLM 内存 |

**参数调优建议**：

1. **生成数量 (`num_generations`)**
   - 增加到 8：提高多样性，但计算量翻倍
   - 减少到 2：加快训练，但可能收敛慢

2. **学习率 (`learning_rate`)**
   - 太大：策略不稳定
   - 太小：收敛慢
   - 推荐：5e-7 到 1e-6

3. **KL 系数 (`beta`)**
   - 太大：策略变化过小
   - 太小：策略崩溃
   - 推荐：0.05 - 0.2

4. **权重平衡 (`alpha` vs `format_weight`)**
   - 如果格式完整但质量差：降低 `format_weight`
   - 如果质量高但格式差：提高 `format_weight`

---

## 5. 配置参考

### 5.1 Shell 脚本配置

**完整训练命令**：

```bash
#!/bin/bash

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=18579
export WANDB_PROJECT=grpo_training

# 训练命令
swift rlhf \
    # === 基础配置 ===
    --rlhf_type grpo \
    --model models/qwen3-8b-base \
    --reward_model models/qwen3.5-35b-a3b \
    --reward_model_plugin train.code.genrm_plugin:get_review_genrm_plugin \
    --external_plugins train/code/genrm_plugin.py \
    \
    # === 数据配置 ===
    --dataset data/openreview_dataset/grpo_train.json \
    --val_dataset data/openreview_dataset/grpo_val.json \
    --max_length 16384 \
    --max_new_tokens 2000 \
    \
    # === 输出配置 ===
    --output_dir models/grpo_qwen3_8b_grm \
    --tuner_type full \
    --torch_dtype bfloat16 \
    \
    # === 训练配置 ===
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --bf16 true \
    \
    # === 评估和保存 ===
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    \
    # === DeepSpeed 配置 ===
    --deepspeed configs/deepspeed_zero3_config.json \
    --offload_optimizer true \
    --offload_model true \
    \
    # === GRPO 配置 ===
    --num_generations 4 \
    --temperature 0.7 \
    --top_p 0.9 \
    --beta 0.1 \
    \
    # === 奖励配置 ===
    --alpha 1.0 \
    --format_weight 1.0 \
    \
    # === vLLM 配置 ===
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 16384 \
    --vllm_enforce_eager true \
    --sleep_level 1 \
    \
    # === 日志配置 ===
    --report_to wandb \
    --run_name grpo_qwen3_8b_grm_v1
```

### 5.2 YAML 配置文件

**文件：`configs/grpo_grm_config.yaml`**

```yaml
# === 模型配置 ===
model:
  model_id_or_path: models/qwen3-8b-base
  trust_remote_code: true

# === 奖励模型配置 ===
reward_model:
  model_id_or_path: models/qwen3.5-35b-a3b
  model_type: qwen  # 用于模板选择
  trust_remote_code: true

# === 数据集配置 ===
dataset:
  train_dataset: data/openreview_dataset/grpo_train.json
  val_dataset: data/openreview_dataset/grpo_val.json
  max_length: 16384
  max_new_tokens: 2000
  train_dataset_sample: -1  # -1 表示使用全部数据

# === 训练配置 ===
training:
  output_dir: models/grpo_qwen3_8b_grm
  tuner_type: full
  torch_dtype: bfloat16

  # Batch 和梯度累积
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8

  # 优化器参数
  learning_rate: 5.0e-7
  weight_decay: 0.01
  warmup_ratio: 0.05

  # 训练轮数
  num_train_epochs: 1

  # 显存优化
  gradient_checkpointing: true
  bf16: true

  # 评估和保存
  eval_strategy: steps
  eval_steps: 50
  save_steps: 100
  save_total_limit: 2
  logging_steps: 5

  # DeepSpeed 配置
  deepspeed: configs/deepspeed_zero3_config.json

# === GRPO 配置 ===
rlhf:
  rlhf_type: grpo
  num_generations: 4
  temperature: 0.7
  top_p: 0.9
  beta: 0.1

# === 奖励配置 ===
reward:
  reward_func: train.code.reward_function:combined_reward
  reward_model_plugin: train.code.genrm_plugin:get_review_genrm_plugin
  alpha: 1.0              # RM 分数权重
  format_weight: 1.0      # 格式分数权重

# === vLLM 配置 ===
vllm:
  use_vllm: true
  vllm_mode: colocate
  vllm_gpu_memory_utilization: 0.3
  vllm_max_model_len: 16384
  vllm_enforce_eager: true

# === 优化配置 ===
optimization:
  offload_optimizer: true
  offload_model: true
  sleep_level: 1

# === WandB 配置 ===
wandb:
  project: grpo_training
  run_name: grpo_qwen3_8b_grm_v1
```

### 5.3 环境变量

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 可见的 GPU | `0,1,2,3,4,5,6,7` |
| `MASTER_ADDR` | 主节点地址 | `127.0.0.1` |
| `MASTER_PORT` | 主节点端口 | `18579` |
| `NPROC_PER_NODE` | 每节点进程数 | `8` |
| `WANDB_PROJECT` | WandB 项目名 | `grpo_training` |
| `WANDB_API_KEY` | WandB API Key | `your_key_here` |

### 5.4 超参数调优

**学习率调优**：

```bash
# 保守配置（稳定性优先）
--learning_rate 3e-7 --num_train_epochs 2

# 标准配置（推荐）
--learning_rate 5e-7 --num_train_epochs 1

# 激进配置（速度优先）
--learning_rate 1e-6 --num_train_epochs 0.5
```

**生成数量调优**：

```bash
# 高质量（慢）
--num_generations 8

# 标准（推荐）
--num_generations 4

# 快速（低质量）
--num_generations 2
```

**权重平衡调优**：

```bash
# 重视质量
--alpha 2.0 --format_weight 0.5

# 平衡（推荐）
--alpha 1.0 --format_weight 1.0

# 重视格式
--alpha 0.5 --format_weight 2.0
```

---

## 6. 性能优化

### 6.1 显存优化策略

**策略1：DeepSpeed ZeRO-3 + CPU Offload**

```bash
--deepspeed configs/deepspeed_zero3_config.json \
--offload_optimizer true \
--offload_model true
```

**效果**：
- 优化器状态卸载到 CPU：节省 ~10-20GB
- 模型参数卸载到 CPU：节省 ~20-30GB
- 总节省：~30-50GB

**策略2：梯度检查点**

```bash
--gradient_checkpointing true
```

**效果**：
- 用计算换内存
- 节省 ~20-30GB
- 增加约 20% 训练时间

**策略3：降低 vLLM 显存占用**

```bash
--vllm_gpu_memory_utilization 0.2  # 从 0.3 降到 0.2
```

**效果**：
- 节省 ~8-16GB (8× GPU)
- 可能略增加推理时间

**策略4：减小 batch size 和增加梯度累积**

```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16  # 从 8 增加到 16
```

**效果**：
- 减少 batch size 显存
- 保持有效 batch size 不变

### 6.2 计算效率优化

**优化1：使用 vLLM 加速**

```bash
--use_vllm true \
--vllm_mode colocate
```

**效果**：
- 生成速度提升 3-5×
- 显存需要共享

**优化2：调整生成参数**

```bash
--max_new_tokens 1500  # 从 2000 降到 1500
```

**效果**：
- 每次生成快 ~25%
- 可能略影响评审完整性

**优化3：减少生成数量**

```bash
--num_generations 2  # 从 4 降到 2
```

**效果**：
- 训练速度提升 2×
- 可能影响收敛性

**优化4：使用混合精度训练**

```bash
--bf16 true \
--torch_dtype bfloat16
```

**效果**：
- 显存减半
- 速度提升 ~30%
- 精度损失可忽略

### 6.3 性能基准测试

**硬件配置**：
- 8× NVIDIA A100 80GB
- CPU: Intel Xeon / AMD EPYC
- 内存: 512GB+ DDR4

**测试结果**：

| 方案 | 单步时间 | Epoch 时间 | 显存占用 | 总时间 |
|------|---------|-----------|---------|--------|
| 方案A (生成式 RM) | 30-60s | 8-16h | ~120GB | 8-16h |
| 方案B (传统 RM) | 15-30s | 4-8h | ~80GB | 10-16h (含 RM 训练) |

**性能优化建议**：

1. **使用方案A (生成式 RM)**：如果显存充足
2. **使用 vLLM 加速**：必须启用
3. **ZeRO-3 + CPU Offload**：如果显存紧张
4. **降低 `num_generations`**：如果时间紧张

---

## 7. 故障排查

### 7.1 常见错误

#### 错误1：OOM (Out of Memory)

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解决方案**：

```bash
# 方案1：降低 vLLM 显存占用
--vllm_gpu_memory_utilization 0.2

# 方案2：启用 offload
--offload_optimizer true --offload_model true

# 方案3：减小 batch size
--per_device_train_batch_size 1

# 方案4：增加梯度累积
--gradient_accumulation_steps 16

# 方案5：降低 max_length
--max_length 8192
```

#### 错误2：插件导入失败

**症状**：
```
ModuleNotFoundError: No module named 'train.code.genrm_plugin'
```

**解决方案**：

```bash
# 确保 plugin 文件存在
ls -la train/code/genrm_plugin.py

# 确保路径正确
--external_plugins train/code/genrm_plugin.py

# 确保函数名正确
--reward_model_plugin train.code.genrm_plugin:get_review_genrm_plugin
```

#### 错误3：评分提取失败

**症状**：
```
Warning: Failed to extract rating from response: ...
```

**解决方案**：

1. 检查 system prompt 是否正确
2. 增加更多正则表达式模式
3. 提供默认值：

```python
def _extract_rating(self, response: str) -> Optional[float]:
    score = self._try_extract(response)
    if score is None:
        logger.warning("Using default score 5.0")
        return 5.0  # 默认中等分数
    return score
```

#### 错误4：生成式 RM 推理太慢

**症状**：
- 每个 step 需要 >60 秒
- 训练速度明显慢于预期

**解决方案**：

```bash
# 方案1：使用外部部署
# 启动 vLLM 服务
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model models/qwen3.5-35b-a3b \
    --tensor-parallel-size 4

# 修改插件使用 OpenAI 客户端

# 方案2：降低生成长度
--max_tokens 256  # 从 512 降到 256

# 方案3：减小生成数量
--num_generations 2  # 从 4 降到 2
```

#### 错误5：奖励分数异常

**症状**：
- 所有奖励分数都相同
- 奖励分数为 0 或 NaN

**解决方案**：

1. 检查格式分数计算：

```python
# 测试
python -c "
from train.code.reward_function import compute_format_score
test_review = '''
### Key Points
- Test
### Rating
**Overall Quality:** 8.0
'''
print(compute_format_score(test_review))
"
```

2. 检查 RM 分数提取：

```python
# 测试评分提取
plugin = ReviewGenRMPlugin(...)
test_response = "Rating: 8.5"
print(plugin._extract_rating(test_response))
```

3. 调整权重：

```bash
--alpha 1.0 --format_weight 1.0
```

### 7.2 调试技巧

**技巧1：启用详细日志**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**技巧2：测试单个样本**

```python
from train.code.genrm_plugin import ReviewGenRMPlugin

# 创建插件
plugin = ReviewGenRMPlugin(...)

# 测试单个样本
test_input = {
    'messages': [
        {'role': 'user', 'content': 'Please review...'},
        {'role': 'assistant', 'content': 'Generated review...'}
    ]
}

rewards = plugin([test_input])
print(f"Reward: {rewards[0]}")
```

**技巧3：监控 GPU 使用**

```bash
# 在训练时监控
watch -n 1 nvidia-smi
```

**技巧4：检查中间输出**

```python
# 在 __call__ 中添加日志
def __call__(self, inputs, **kwargs):
    for i, item in enumerate(inputs):
        messages = item.get("messages", [])
        completion = self._extract_completion(messages)

        # 打印调试信息
        logger.info(f"Sample {i}:")
        logger.info(f"  Completion: {completion[:100]}...")

        format_score = self.compute_format_score(completion)
        logger.info(f"  Format score: {format_score}")

        rm_score = self._get_genrm_score(messages, completion)
        logger.info(f"  RM score: {rm_score}")

        total_reward = self.format_weight * format_score + self.alpha * rm_score
        logger.info(f"  Total reward: {total_reward}")

        rewards.append(total_reward)

    return rewards
```

### 7.3 日志分析

**正常训练日志**：

```
Step 1: loss=2.345, reward=6.78, learning_rate=5e-7
Step 2: loss=2.123, reward=6.89, learning_rate=5e-7
Step 3: loss=1.987, reward=7.01, learning_rate=5e-7
...
```

**异常情况1：Loss 不下降**

```
Step 1: loss=2.345, reward=6.78
Step 2: loss=2.345, reward=6.78
Step 3: loss=2.345, reward=6.78
```

**可能原因**：
- 学习率太小
- 奖励信号太弱
- 策略崩溃

**解决方案**：
```bash
# 增加学习率
--learning_rate 1e-6

# 调整奖励权重
--alpha 2.0 --format_weight 0.5

# 降低 beta
--beta 0.05
```

**异常情况2：Reward 异常**

```
Step 1: reward=0.00
Step 2: reward=0.00
Step 3: reward=0.00
```

**可能原因**：
- 格式分数为 0
- RM 分数为 0
- 提取失败

**解决方案**：
- 检查格式分数计算
- 检查 RM 评分提取
- 启用详细日志

**异常情况3：OOM 频繁**

```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 降低 vLLM 显存占用
- 启用 CPU offload
- 减小 batch size

---

## 8. 参考资料

### 8.1 相关论文

1. **GRPO**: Group Relative Policy Optimization
   - 论文链接：https://arxiv.org/abs/2402.03100
   - 核心思想：组内相对策略优化，无需价值模型

2. **PPO**: Proximal Policy Optimization
   - 论文链接：https://arxiv.org/abs/1707.06347
   - GRPO 的基础算法

3. **DPO**: Direct Preference Optimization
   - 论文链接：https://arxiv.org/abs/2305.18290
   - 本项目使用 DPO 数据训练 RM

### 8.2 框架文档

1. **ms-swift (Swift)**
   - GitHub: https://github.com/modelscope/ms-swift
   - 文档: https://swift.readthedocs.io/
   - GRPO 教程: https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html

2. **vLLM**
   - GitHub: https://github.com/vllm-project/vllm
   - 文档: https://docs.vllm.ai/

3. **DeepSpeed**
   - GitHub: https://github.com/microsoft/DeepSpeed
   - 文档: https://www.deepspeed.ai/

### 8.3 代码索引

**核心文件**：

| 文件 | 功能 | 关键类/函数 |
|------|------|-----------|
| `train/code/genrm_plugin.py` | 生成式 RM 插件 | `ReviewGenRMPlugin` |
| `train/code/rm_plugin.py` | 传统 RM 插件 | `ReviewRewardModelPlugin` |
| `train/code/reward_function.py` | 奖励函数 | `compute_format_score`, `CombinedRewardFunction` |
| `train/code/plugin.py` | GRPO 插件 | `review_format_reward`, `combined_reward` |
| `scripts/train_grpo_GRM.sh` | 生成式 RM 训练脚本 | - |
| `scripts/train_grpo_pipeline.sh` | 传统 RM 训练脚本 | - |
| `scripts/convert_dpo_to_rm.py` | 数据转换 | `convert_dpo_to_rm()` |
| `scripts/prepare_grpo_data.py` | 数据准备 | `prepare_grpo_data()` |
| `configs/grpo_grm_config.yaml` | GRPO 配置 | - |
| `configs/deepspeed_zero3_config.json` | DeepSpeed 配置 | - |

**关键代码位置**：

| 功能 | 文件 | 行号 |
|------|------|------|
| 格式分数计算 | `train/code/reward_function.py` | 28-33 |
| 组合奖励 | `train/code/reward_function.py` | 69-84 |
| 生成式 RM 插件 | `train/code/genrm_plugin.py` | 全文 |
| 评分提取 | `train/code/genrm_plugin.py` | 233-280 |
| 训练命令 | `scripts/train_grpo_GRM.sh` | 76-124 |

### 8.4 数据格式

**GRPO 训练数据格式**：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Please review the following paper..."
    }
  ]
}
```

**RM 训练数据格式**：

```json
{
  "query": "Please review the following paper...",
  "chosen": "### Key Points\n- The paper presents...",
  "rejected": "This paper is good."
}
```

### 8.5 模型下载

**Qwen3.5-35B-A3B (生成式 RM)**：
```bash
# 使用脚本下载
bash scripts/download_qwen35_35b_a3b.sh

# 或使用 Python
python scripts/download_qwen35_35b_a3b.py
```

**Qwen3-8B-Base (策略模型)**：
```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen3-8B-Base \
  --local-dir models/qwen3-8b-base
```

### 8.6 社区资源

1. **ms-swift GitHub Issues**: https://github.com/modelscope/ms-swift/issues
2. **Swift Discord**: https://discord.gg/modelscope
3. **Stack Overflow**: 标签 `ms-swift`, `grpo`, `rlhf`

---

## 附录

### A. 完整示例：从零开始训练

```bash
# === 1. 准备环境 ===
pip install ms-swift transformers accelerate deepspeed vllm wandb

# === 2. 准备数据 ===
python scripts/convert_dpo_to_rm.py
python scripts/prepare_grpo_data.py

# === 3. 下载生成式 RM ===
bash scripts/download_qwen35_35b_a3b.sh

# === 4. 启动训练 ===
bash scripts/train_grpo_GRM.sh

# === 5. 评估模型 ===
python eval/eval.py --model_name grpo_qwen3_8b_grm
```

### B. 参数快速参考

```bash
# 快速开始（推荐配置）
bash scripts/train_grpo_GRM.sh

# 高质量（慢）
--num_generations 8 --alpha 2.0 --format_weight 0.5

# 快速训练（低质量）
--num_generations 2 --alpha 0.5 --format_weight 2.0

# 显存优化
--vllm_gpu_memory_utilization 0.2 \
--offload_optimizer true \
--offload_model true \
--gradient_accumulation_steps 16

# 速度优化
--max_new_tokens 1500 \
--num_generations 2 \
--max_length 8192
```

### C. 故障排查检查清单

- [ ] 检查 GPU 显存是否充足
- [ ] 检查模型文件是否完整
- [ ] 检查数据格式是否正确
- [ ] 检查插件路径和函数名
- [ ] 检查 DeepSpeed 配置
- [ ] 检查 vLLM 版本兼容性
- [ ] 检查网络连接（WandB, 模型下载）
- [ ] 检查磁盘空间（>200GB）

---

**文档版本**: 1.0
**最后更新**: 2026-03-09
**维护者**: OpenReview 项目组
