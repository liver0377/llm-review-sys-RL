# 使用生成式奖励模型 (GRM) 进行 GRPO 训练

快速指南：使用 Qwen3.5-35B-A3B 作为生成式奖励模型。

## 快速开始

```bash
# 1. 下载生成式RM模型 (~70GB)
bash scripts/download_qwen35_35b_a3b.sh

# 2. 运行GRPO训练
bash scripts/train_grpo_GRM.sh
```

## 脚本说明

### `train_grpo_GRM.sh` - 使用生成式RM的GRPO训练脚本

**特点：**
- ✅ 直接使用 Qwen3.5-35B-A3B 作为生成式RM
- ✅ 无需训练传统的8B奖励模型
- ✅ 组合格式分数和RM分数
- ✅ 优化的显存配置（8 GPUs）

**训练流程：**
```
Step 1: 准备RM数据
Step 2: 准备GRPO数据
Step 3: 检查生成式RM (Qwen3.5-35B-A3B)
Step 4: GRPO训练 (使用生成式RM)
```

**输出模型：**
- 策略模型：`models/grpo_qwen3_8b_grm`
- 生成式RM：`models/qwen3.5-35b-a3b`（预训练，不修改）

## 与其他脚本的区别

| 脚本 | 奖励模型 | 是否需要训练RM | 推荐场景 |
|------|---------|---------------|---------|
| `train_grpo_GRM.sh` | Qwen3.5-35B-A3B (生成式) | ❌ 不需要 | **推荐** - 使用强大的生成式RM |
| `train_grpo.sh` | Qwen3.5-35B-A3B (生成式) | ⚠️ 仍训练8B RM | 包含冗余的RM训练步骤 |
| `train_grpo_pipeline.sh` | 训练的8B RM | ✅ 需要 | 传统方法，训练小型RM |

## 配置

### 奖励函数

```
总奖励 = format_weight × 格式分数 + alpha × RM分数
```

- **格式分数** (0-4分)：评估评审结构完整性
- **RM分数** (0-10分)：Qwen3.5-35B-A3B生成的质量分数
- **默认权重**：`format_weight=1.0`, `alpha=1.0`

### 调整权重

编辑 `train_grpo_GRM.sh` 中的参数：

```bash
--alpha 1.0              # RM分数权重
--format_weight 1.0      # 格式分数权重
```

**示例：**
```bash
# 强调RM分数
--alpha 2.0 --format_weight 0.5

# 强调格式分数
--alpha 0.5 --format_weight 2.0

# 只使用RM分数
--alpha 1.0 --format_weight 0.0
```

## 系统要求

### 硬件
- **GPU**: 8× NVIDIA GPU (推荐 A100 40GB/80GB)
- **内存**: 足够加载 35B 模型
- **磁盘**: ~150GB（模型 + 训练数据 + 输出）

### 软件
- Python 3.8+
- ms-swift
- PyTorch + CUDA
- vLLM

## 故障排除

### OOM (内存溢出)

如果遇到显存不足：

1. **降低 vLLM 显存占用**：
   ```bash
   --vllm_gpu_memory_utilization 0.2  # 从0.3降到0.2
   ```

2. **增加梯度累积**：
   ```bash
   --gradient_accumulation_steps 16  # 从8增到16
   ```

3. **减少GPU数量**：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU
   NPROC_PER_NODE=4
   ```

### 模型未找到

```
Error: Generative Reward Model not found
```

**解决方法：**
```bash
bash scripts/download_qwen35_35b_a3b.sh
```

### 训练速度慢

这是正常的，因为：
- 35B生成式RM比小型RM慢
- 每个batch需要RM打分

**优化建议：**
- 减少 `max_length`（如果评审较短）
- 减少 `num_generations`（默认4）
- 使用更快的GPU

## 配置文件

使用配置文件运行：

```bash
python train/code/train_grpo.py --config configs/grpo_grm_config.yaml
```

## 监控训练

### Wandb
训练会自动上传到 Wandb：
- 项目：`grpo_training`
- 运行名称：`grpo_qwen3_8b_grm_v1`

### 日志
查看训练日志：
```bash
# 查看实时日志
tail -f models/grpo_qwen3_8b_grm/trainer_log.jsonl

# 查看Wandb链接（在终端输出中）
```

## 评估

训练完成后评估模型：

```bash
python eval/eval.py --model_name grpo_qwen3_8b_grm
```

## 参考资料

- [完整指南](docs/generative_rm_guide.md)
- [MS-Swift RLHF](https://github.com/modelscope/ms-swift)
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
