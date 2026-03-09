# 基于 LLM 的自动化论文评审系统 - GRPO 训练

## 项目简介

本项目使用 **GRPO (Group Relative Policy Optimization)** 强化学习技术训练大语言模型进行自动化论文评审。通过结合生成式奖励模型（Qwen3.5-35B-A3B）和格式分数评估，实现了高质量的论文评审生成。

---

## 📚 文档导航

### 🚀 快速开始
- **[GRPO 快速开始](./GRPO_Quickstart.md)** - 5 分钟上手 GRPO 训练

### 📖 技术文档
- **[GRPO 训练技术指南](./RL/GRPO_Training_Technical_Guide.md)** ⭐ - 完整的技术文档（2267 行）
  - 系统架构和数据流
  - 核心概念详解（GRPO 算法、奖励机制）
  - 两种训练方案完整代码
  - 性能优化和故障排查
- **[GRPO 快速参考](./RL/GRPO_Quick_Reference.md)** - 常用命令和参数速查
- **[RL 目录索引](./RL/README.md)** - RL 文档导航

### 🔧 框架文档
- **[Swift GenRM 文档](./swift/swift-GenRM.md)** - ms-swift 生成式 RM 指南

---

## 🎯 训练方案对比

| 方案 | 脚本 | 奖励模型 | 显存需求 | 训练时间 | 质量 | 推荐度 |
|------|------|---------|---------|---------|------|--------|
| **生成式 RM** | `train_grpo_GRM.sh` | Qwen3.5-35B-A3B | ~120GB | 8-16h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **传统 RM** | `train_grpo_pipeline.sh` | 训练的 8B RM | ~80GB | 10-16h | ⭐⭐⭐ | ⭐⭐⭐ |

**推荐选择**：
- ✅ **生成式 RM**：质量优先，显存充足（8× A100 80GB）
- ⚠️ **传统 RM**：显存受限，或需要快速迭代

---

## 🚀 快速开始

### 方案A：生成式 RM（推荐）

```bash
# 1. 下载生成式奖励模型（约 70GB）
bash scripts/download_qwen35_35b_a3b.sh

# 2. 准备数据
python scripts/convert_dpo_to_rm.py
python scripts/prepare_grpo_data.py

# 3. 启动训练
bash scripts/train_grpo_GRM.sh
```

### 方案B：传统 RM

```bash
# 1. 准备数据
python scripts/convert_dpo_to_rm.py
python scripts/prepare_grpo_data.py

# 2. 训练奖励模型 + GRPO
bash scripts/train_grpo_pipeline.sh
```

---

## 📊 奖励机制

### 组合奖励公式

```
total_reward = format_weight × format_score + alpha × rm_score
```

- **格式分数** (0-4 分)：基于正则表达式检查评审结构
  - Overall Quality (1-10)
  - Review Confidence (1-5)
  - Key Points, Strengths, Weaknesses, Suggestions, Rating

- **RM 分数** (0-10 分)：来自奖励模型的质量评估
  - 生成式 RM：Qwen3.5-35B-A3B 直接评分
  - 传统 RM：训练的 8B 奖励模型评分

### 权重调整

```bash
# 重视质量
--alpha 2.0 --format_weight 0.5

# 平衡（推荐）
--alpha 1.0 --format_weight 1.0

# 重视格式
--alpha 0.5 --format_weight 2.0
```

---

## 🎨 标准评审格式

```markdown
### Key Points
- The paper presents...

### Strengths and Weaknesses
**Strengths:**
- Clear methodology

**Weaknesses:**
- Limited discussion

### Suggestions for Improvement
- Add comparison...

### Rating
**Overall Quality:** 8.5
**Review Confidence:** 4.0
```

---

## 📁 项目结构

```
.
├── scripts/               # 训练脚本
│   ├── train_grpo_GRM.sh          # 生成式 RM 训练
│   ├── train_grpo_pipeline.sh     # 传统 RM 训练
│   ├── prepare_grpo_data.py       # GRPO 数据准备
│   └── convert_dpo_to_rm.py       # DPO → RM 数据转换
│
├── train/code/            # 训练代码
│   ├── genrm_plugin.py             # 生成式 RM 插件
│   ├── reward_function.py          # 奖励函数
│   └── plugin.py                  # GRPO 插件
│
├── configs/               # 配置文件
│   ├── grpo_grm_config.yaml        # GRPO 配置
│   └── deepspeed_zero3_config.json # DeepSpeed 配置
│
└── models/               # 模型输出
    ├── grpo_qwen3_8b_grm/          # GRPO 策略模型
    └── qwen3.5-35b-a3b/            # 生成式 RM
```

---

## ⚙️ 核心参数

### 生成参数

```bash
--num_generations 4     # 每个 prompt 生成 4 个 responses
--temperature 0.7       # 生成温度
--max_new_tokens 2000   # 最大生成长度
```

### RLHF 参数

```bash
--beta 0.1              # KL 散度系数
--alpha 1.0             # RM 分数权重
--format_weight 1.0     # 格式分数权重
```

### 训练参数

```bash
--learning_rate 5e-7    # 学习率（很小）
--gradient_accumulation_steps 8  # 梯度累积
--per_device_train_batch_size 1  # 每 GPU batch size
```

### 显存优化

```bash
--vllm_gpu_memory_utilization 0.3  # vLLM 显存占用
--offload_optimizer true           # 优化器卸载到 CPU
--offload_model true               # 模型参数卸载到 CPU
--gradient_checkpointing true      # 梯度检查点
```

---

## 🐛 常见问题

### Q1: OOM (Out of Memory)

```bash
# 降低 vLLM 显存占用
--vllm_gpu_memory_utilization 0.2

# 启用 CPU offload
--offload_optimizer true --offload_model true

# 减小 batch size
--per_device_train_batch_size 1
```

### Q2: 训练速度慢

```bash
# 使用外部部署（参考技术文档）
# 或降低生成长度
--max_new_tokens 1500

# 或减少生成数量
--num_generations 2
```

### Q3: 评分提取失败

检查：
1. System prompt 是否正确
2. 正则表达式模式是否匹配
3. 模型输出格式是否符合预期

详见：[GRPO 技术指南 - 故障排查](./RL/GRPO_Training_Technical_Guide.md#7-故障排查)

---

## 📊 性能基准

**硬件配置**：8× NVIDIA A100 80GB

| 指标 | 生成式 RM | 传统 RM |
|------|-----------|---------|
| 单步时间 | 30-60s | 15-30s |
| Epoch 时间 | 8-16h | 4-8h (GRPO 部分) |
| 显存占用 | ~120GB | ~80GB |
| 总时间 | 8-16h | 10-16h (含 RM 训练) |

---

## 🔗 相关资源

### 数据集
- [OpenReview 原始数据](https://huggingface.co/datasets/guochenmeinian/openreview)
- [OpenReview 训练数据](https://huggingface.co/datasets/guochenmeinian/openreview_dataset)

### 模型
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) - 生成式奖励模型
- [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) - 策略模型基础

### 框架
- [ms-swift](https://github.com/modelscope/ms-swift) - Swift 框架
- [vLLM](https://github.com/vllm-project/vllm) - 高效推理引擎
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 分布式训练

### 论文
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03100)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

---

## 📂 归档文档

以下文档已归档至 `archive/` 目录，供参考：

- **[SFT 微调](./archive/sft/)** - 监督式微调相关文档
- **[DPO 偏好优化](./archive/dpo/)** - DPO 训练文档
- **[传统 RM](./archive/reward_model/)** - 奖励模型训练文档
- **[数据处理](./archive/data/)** - 数据爬取与处理
- **[评估](./archive/evaluation/)** - 模型评估相关

---

## 📞 获取帮助

- 📖 完整文档：[GRPO 训练技术指南](./RL/GRPO_Training_Technical_Guide.md)
- 🐛 问题排查：[故障排查章节](./RL/GRPO_Training_Technical_Guide.md#7-故障排查)
- 💬 社区支持：[Swift Discord](https://discord.gg/modelscope)
- 📋 Issues：[ms-swift GitHub](https://github.com/modelscope/ms-swift/issues)

---

## 📄 许可证

本项目遵循相关开源许可证。

---

**最后更新**: 2026-03-09
**维护者**: OpenReview 项目组
