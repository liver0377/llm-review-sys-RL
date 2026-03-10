# veRL GRPO训练指南

## 环境配置

### 依赖环境
- conda环境: `verl`
- veRL版本: 0.8.0.dev
- PyTorch: 2.10.0+cu130
- Transformers: 4.57.6

### GPU分配
```
GPU 0-2: 训练进程
GPU 4-7: Reward Model服务 (vLLM, 端口8002)
```

## 关键修复

### 问题1: Weight Bucket大小不足
**错误信息**:
```
AssertionError: Weight model.embed_tokens.weight is too large to fit in the bucket
```

**解决方案**:
增加`rollout.update_weights_bucket_megabytes`配置：
```yaml
rollout:
  update_weights_bucket_megabytes: 8192  # 8GB
```

### 问题2: Tensor Parallel配置
根据GPU数量配置：
- 2卡: `tensor_parallel_size=2`
- 4卡: `tensor_parallel_size=4`

## 训练脚本

### 方式1: SPMD模式（推荐）
```bash
conda activate verl
CUDA_VISIBLE_DEVICES=0,1,2 \
python train/code/train_grpo_verl_spmd.py
```

### 方式2: 自定义训练
```bash
conda activate verl
CUDA_VISIBLE_DEVICES=0,1,2 \
python train/code/train_grpo_verl_native.py
```

## 配置文件

**路径**: `configs/verl_grpo_config.yaml`

**关键参数**:
```yaml
model:
  model_path: models/Qwen3-8B-Base
  
training:
  batch_size: 4
  learning_rate: 5e-7
  num_generations: 3
  
rollout:
  update_weights_bucket_megabytes: 8192  # 必须足够大
  tensor_parallel_size: 2
```

## RM服务

### 启动RM服务
```bash
bash scripts/start_rm_service.sh
```

### 验证RM服务
```bash
curl http://127.0.0.1:8002/v1/models
```

## 监控训练

### 实时日志
```bash
tail -f logs/grpo_verl_*.log
```

### GPU监控
```bash
watch -n 1 nvidia-smi
```

### Wandb
- 自动启用，查看终端输出的URL

## 常见问题

### Q: Weight bucket不足
**A**: 增加`update_weights_bucket_megabytes`到8192或更高

### Q: OOM错误
**A**: 降低`batch_size`或`micro_batch_size`

### Q: RM服务连接失败
**A**: 检查端口8002是否启动：`curl http://127.0.0.1:8002/v1/models`

## 性能优化

### 已实现
- ✅ vLLM加速生成
- ✅ 异步批量RM调用
- ✅ Tensor Parallel
- ✅ Gradient Checkpointing

### 建议
- 根据GPU内存调整`batch_size`
- 使用`micro_batch_size`控制内存峰值
- 监控GPU内存避免OOM

## 输出

### 模型保存
- Checkpoint: `models/grpo_qwen3_8b_verl/checkpoint-{step}`
- Final: `models/grpo_qwen3_8b_verl/final`

### 日志
- 训练日志: `logs/grpo_verl_*.log`
- Wandb日志: 自动保存

## 评估

```bash
python eval/eval.py --model_path models/grpo_qwen3_8b_verl/final
```
