# GRPO 训练快速参考卡片

## 🎯 快速选择指南

| 方案 | 脚本 | 显存 | 时间 | 质量 | 推荐度 |
|------|------|------|------|------|--------|
| **生成式 RM** | `train_grpo_GRM.sh` | ~120GB | 8-16h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **传统 RM** | `train_grpo_pipeline.sh` | ~80GB | 10-16h | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 🚀 一键启动

```bash
# 生成式 RM（推荐）
bash scripts/train_grpo_GRM.sh

# 传统 RM
bash scripts/train_grpo_pipeline.sh
```

---

## ⚙️ 核心参数

### 奖励权重

```bash
# 重视质量
--alpha 2.0 --format_weight 0.5

# 平衡（推荐）
--alpha 1.0 --format_weight 1.0

# 重视格式
--alpha 0.5 --format_weight 2.0
```

### 生成参数

```bash
# 高质量（慢）
--num_generations 8 --temperature 0.7

# 标准（推荐）
--num_generations 4 --temperature 0.7

# 快速（低质量）
--num_generations 2 --temperature 0.9
```

### 优化参数

```bash
# 显存优化
--vllm_gpu_memory_utilization 0.2 \
--offload_optimizer true \
--offload_model true

# 速度优化
--max_new_tokens 1500 \
--num_generations 2
```

---

## 📊 奖励计算

```
total_reward = format_weight × format_score + alpha × rm_score

format_score: [0, 4.0]   # 格式分数
rm_score:     [0, 10.0]  # RM 分数
```

**格式评分标准**：

| 检查项 | 分值 |
|--------|------|
| Overall Quality (1-10) | 1.0 |
| Review Confidence (1-5) | 1.0 |
| ### Key Points | 0.5 |
| ### Strengths and Weaknesses | 0.25 |
| **Strengths:** / **Weaknesses:** | 0.5 |
| ### Suggestions for Improvement | 0.5 |
| ### Rating | 0.25 |
| **满分** | **4.0** |

---

## 🔧 常见问题

### OOM (Out of Memory)

```bash
# 降低 vLLM 显存
--vllm_gpu_memory_utilization 0.2

# 启用 offload
--offload_optimizer true --offload_model true

# 减小 batch size
--per_device_train_batch_size 1
```

### 训练太慢

```bash
# 方案1：使用外部部署（参考技术文档）
# 方案2：降低生成长度
--max_new_tokens 1500

# 方案3：减少生成数量
--num_generations 2
```

### 奖励异常

```bash
# 检查格式分数
python -c "
from train.code.reward_function import compute_format_score
test = '### Key Points\n- Test\n### Rating\n**Overall Quality:** 8.0'
print(compute_format_score(test))
"

# 检查插件
python scripts/test_genrm_plugin.py
```

---

## 📝 数据格式

**GRPO 训练数据**：
```json
{
  "messages": [
    {"role": "user", "content": "Please review..."}
  ]
}
```

**RM 训练数据**：
```json
{
  "query": "Please review...",
  "chosen": "### Key Points\n- Good...",
  "rejected": "Bad review."
}
```

---

## 🔗 关键文件

| 文件 | 功能 |
|------|------|
| `train/code/genrm_plugin.py` | 生成式 RM 插件 |
| `train/code/reward_function.py` | 奖励函数 |
| `scripts/train_grpo_GRM.sh` | 训练脚本 |
| `configs/grpo_grm_config.yaml` | 配置文件 |

---

## 📖 完整文档

详见：[GRPO_Training_Technical_Guide.md](./GRPO_Training_Technical_Guide.md)

---

**更新**: 2026-03-09
