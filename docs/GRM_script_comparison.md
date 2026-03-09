# GRPO 训练脚本对比

本项目提供了三个GRPO训练脚本，以下是它们的详细对比：

## 脚本对比表

| 特性 | train_grpo_GRM.sh | train_grpo.sh | train_grpo_pipeline.sh |
|------|------------------|---------------|----------------------|
| **推荐程度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **奖励模型类型** | 生成式RM | 生成式RM | 传统训练RM |
| **奖励模型** | Qwen3.5-35B-A3B | Qwen3.5-35B-A3B | 训练的8B RM |
| **需要训练RM** | ❌ 不需要 | ⚠️ 需要（但不使用） | ✅ 需要 |
| **训练时间** | 短 | 长（含无用RM训练） | 长 |
| **RM质量** | 高（35B） | 高（35B） | 中（8B） |
| **显存需求** | 高（35B RM） | 高（35B RM） | 低（8B RM） |

## 快速选择指南

### 使用 `train_grpo_GRM.sh` ✅ 推荐

**适用场景：**
- ✅ 想使用最强大的奖励模型
- ✅ 有足够显存（8× GPU）
- ✅ 不想花时间训练RM
- ✅ 追求最佳性能

**命令：**
```bash
# 1. 下载生成式RM
bash scripts/download_qwen35_35b_a3b.sh

# 2. 训练
bash scripts/train_grpo_GRM.sh
```

**输出：**
- `models/grpo_qwen3_8b_grm/`

---

### 使用 `train_grpo.sh` ⚠️ 不推荐

**适用场景：**
- ⚠️ 想使用生成式RM，但脚本有冗余步骤
- ⚠️ 会浪费时间训练一个不用的8B RM

**问题：**
- Step 3 训练8B RM，但 Step 4 不使用它
- 浪费时间和计算资源

**建议：** 使用 `train_grpo_GRM.sh` 替代

---

### 使用 `train_grpo_pipeline.sh` 📊 传统方法

**适用场景：**
- ✅ 显存受限，无法加载35B模型
- ✅ 想使用自己训练的小型RM
- ✅ 需要完全控制RM训练过程

**命令：**
```bash
bash scripts/train_grpo_pipeline.sh
```

**输出：**
- `models/reward_model_qwen3_8b/` (训练的RM)
- `models/grpo_qwen3_8b/` (策略模型)

## 详细流程对比

### train_grpo_GRM.sh (推荐)

```
┌─────────────────────────────────────────┐
│  Step 1: 准备RM数据                     │
│  Step 2: 准备GRPO数据                   │
│  Step 3: 检查生成式RM (Qwen3.5-35B)     │
│  Step 4: GRPO训练 (使用生成式RM)        │
└─────────────────────────────────────────┘
    时间: ~2小时 (不含数据准备)
    显存: 8× GPU (35B RM + 8B Policy)
```

### train_grpo.sh (不推荐)

```
┌─────────────────────────────────────────┐
│  Step 1: 准备RM数据                     │
│  Step 2: 准备GRPO数据                   │
│  Step 3: 训练8B RM ❌ (不会被使用)       │
│  Step 4: GRPO训练 (使用生成式RM)        │
└─────────────────────────────────────────┘
    时间: ~6小时 (含无用RM训练)
    显存: 8× GPU (35B RM + 8B Policy)
```

### train_grpo_pipeline.sh (传统)

```
┌─────────────────────────────────────────┐
│  Step 1: 准备RM数据                     │
│  Step 2: 准备GRPO数据                   │
│  Step 3: 训练8B RM                      │
│  Step 4: GRPO训练 (使用训练的RM)        │
└─────────────────────────────────────────┘
    时间: ~6小时
    显存: 8× GPU (8B RM + 8B Policy)
```

## 奖励函数对比

### 生成式RM (Qwen3.5-35B-A3B)

**优点：**
- ✅ 强大的语言理解能力
- ✅ 无需训练，直接使用
- ✅ 更准确的打分
- ✅ 可以直接生成评审意见

**缺点：**
- ❌ 模型大（~70GB）
- ❌ 推理速度慢
- ❌ 显存占用高

**适用：** 追求最佳质量

### 训练的8B RM

**优点：**
- ✅ 模型小（~16GB）
- ✅ 推理速度快
- ✅ 显存占用低
- ✅ 可以针对特定任务微调

**缺点：**
- ❌ 需要训练时间
- ❌ 需要训练数据
- ❌ 能力上限较低

**适用：** 资源受限或需要定制化

## 性能对比（预估）

| 指标 | 生成式RM (35B) | 训练RM (8B) |
|------|---------------|-------------|
| 打分质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 训练速度 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 显存占用 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 总体推荐 | ✅ | ⚠️ |

## 迁移指南

### 从 train_grpo.sh 迁移到 train_grpo_GRM.sh

**好消息：** 几乎无需修改！

```bash
# 旧脚本
bash scripts/train_grpo.sh

# 新脚本（完全相同的效果，但更快）
bash scripts/train_grpo_GRM.sh
```

**主要区别：**
- ✅ 跳过无用的Step 3（训练8B RM）
- ✅ 节省 ~4小时训练时间
- ✅ 输出目录：`grpo_qwen3_8b_grm` vs `grpo_qwen3_8b_generative_rm`

### 从 train_grpo_pipeline.sh 迁移到 train_grpo_GRM.sh

**步骤：**

1. **下载生成式RM：**
   ```bash
   bash scripts/download_qwen35_35b_a3b.sh
   ```

2. **使用新脚本：**
   ```bash
   # 旧脚本（训练RM + GRPO）
   bash scripts/train_grpo_pipeline.sh

   # 新脚本（直接使用生成式RM）
   bash scripts/train_grpo_GRM.sh
   ```

3. **调整权重（可选）：**
   - 生成式RM的分数范围可能不同
   - 根据验证集性能调整 `--alpha` 和 `--format_weight`

## 常见问题

### Q: 哪个脚本效果最好？

**A:** `train_grpo_GRM.sh` 使用35B生成式RM，理论效果最好。

### Q: 哪个脚本最快？

**A:** `train_grpo_GRM.sh`（跳过RM训练），但如果数据已准备好，`train_grpo_pipeline.sh` 的GRPO步骤可能更快（8B RM vs 35B RM）。

### Q: 我应该使用哪个？

**A:** 推荐顺序：
1. **首选：** `train_grpo_GRM.sh` - 最佳效果，最简单
2. **备选：** `train_grpo_pipeline.sh` - 如果显存不足
3. **不推荐：** `train_grpo.sh` - 有冗余步骤

### Q: 生成式RM需要训练吗？

**A:** 不需要！Qwen3.5-35B-A3B 是预训练模型，直接用作打分器。

### Q: 能用其他生成式模型吗？

**A:** 可以！修改脚本中的 `REWARD_MODEL_PATH` 和 `--reward_model_type` 即可。

## 总结

**推荐配置：**

```bash
# 最佳选择（推荐）
bash scripts/train_grpo_GRM.sh

# 如果显存不足
bash scripts/train_grpo_pipeline.sh
```

**不推荐：** `train_grpo.sh`（会被 `train_grpo_GRM.sh` 取代）
