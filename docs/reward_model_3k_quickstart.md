# Reward Model 训练 - 3000 样本版本

使用 3000 条 DPO 数据训练 Reward Model，从 SFT 权重初始化。

## 快速开始

### 一键训练（推荐）
```bash
./scripts/train_reward_model_3k.sh
```

### 手动执行

**1. 转换数据（3000 条）**
```bash
python train/code/convert_dpo_to_rm.py \
  --input data/openreview_dataset/dpo_base_as_rejected_train_cleaned.json \
  --output data/openreview_dataset/rm_train_3k.json \
  --max_samples 3000 \
  --shuffle \
  --seed 42
```

**2. 训练**
```bash
accelerate launch --config_file configs/accelerate_config_8gpu.yaml \
  train/code/train_reward_model.py \
  --config configs/reward_model_config.yaml \
  --train_data data/openreview_dataset/rm_train_3k.json \
  --val_data data/openreview_dataset/rm_val.json \
  --output_dir models/reward_model_qwen3_8b_3k
```

**3. 评估**
```bash
python eval/eval_reward_model.py \
  --model_path models/reward_model_qwen3_8b_3k/best_model \
  --data_path data/openreview_dataset/rm_val.json \
  --output eval/reward_model_3k_results.json
```

## 预期性能

使用 3000 条高质量样本：
- **准确率**: > 75%
- **平均奖励差异**: > 1.0
- **训练时间**: ~2-4 小时（8× GPU）

## 数据采样参数

`--max_samples`: 限制样本数量（推荐 3000-5000）
`--shuffle`: 打乱数据（确保随机性）
`--seed`: 随机种子（确保可重复性）

## 性能对比

| 样本数 | 准确率 | 训练时间 | 推荐场景 |
|--------|--------|----------|----------|
| 3,000  | ~75-80% | 2-4h  | 快速验证、原型开发 |
| 5,000  | ~80-85% | 3-6h  | 平衡性能与成本 |
| 9,000  | ~85-90% | 6-10h | 最佳性能 |

## 调整样本数量

修改 `--max_samples` 参数：
```bash
# 使用 5000 条样本
python train/code/convert_dpo_to_rm.py \
  --input data/openreview_dataset/dpo_base_as_rejected_train_cleaned.json \
  --output data/openreview_dataset/rm_train_5k.json \
  --max_samples 5000 \
  --shuffle \
  --seed 42
```

## 性能不足时的解决方案

如果准确率 < 70%：
1. **增加样本数**: 提升到 5000-10000
2. **增加训练轮数**: 修改 `num_epochs: 3-5`
3. **检查数据质量**: 确保 chosen 确实优于 rejected
4. **调整学习率**: 尝试 `learning_rate: 5e-6`

## 完整文档

详见: `docs/reward_model_pipeline.md`
