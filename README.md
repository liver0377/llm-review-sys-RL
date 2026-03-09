# 论文评审强化学习训练系统

> 基于 Swift 框架的 GRPO 强化学习训练系统，用于优化论文评审生成模型

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 检查 GPU
nvidia-smi
```

### 2. 数据准备

```bash
# 转换 RM 训练数据
python scripts/convert_dpo_to_rm_swift_format.py \
    --dpo_train data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json \
    --dpo_val data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json \
    --max_train_samples 5000 \
    --random_sample
```

### 3. 启动训练

```bash
# Reward Model 训练
bash scripts/train_rm_full_content_7gpu.sh

# 或后台运行
nohup bash scripts/train_rm_full_content_7gpu.sh > logs/rm_training.log 2>&1 &
```

---

## 文档导航

### 📖 核心文档

- **[Reward Model 训练指南](docs/Reward_Model_Training_Guide.md)** ⭐ **推荐**
  - 数据合成流程
  - 训练脚本使用
  - 参数调优
  - 常见问题

- **[GRPO 训练 README](docs/GRPO_Training_README.md)**
  - GRPO 训练流程
  - 奖励函数设计
  - VLLM 集成

- **[GRPO RL Pipeline](docs/GRPO_RL_Pipeline.md)**
  - 完整训练流程
  - RM + GRPO 两阶段训练

### 📊 数据分析

- **[Query 策略最终方案](docs/RM_QUERY_STRATEGY_FINAL.md)**
  - Token 分布分析
  - 为什么选择完整论文
  - max_length 选择依据

- **[Token 分布统计](docs/rm_token_stats.txt)**
  - 详细的 token 长度统计

- **[完整 Token 分析](docs/rm_query_token_distribution_comprehensive.txt)**
  - 不同策略的对比

### 📈 其他文档

- [评估指南](docs/评估.md)
- [训练曲线](docs/训练曲线.md)
- [数据爬取与处理](docs/数据爬取与处理.md)
- [DPO 偏好优化](docs/dpo偏好优化.md)

---

## 项目结构

```
llm-review-sys-RL/
├── configs/                      # 配置文件
│   ├── deepspeed_zero2_config.json
│   ├── deepspeed_zero3_config.json
│   └── custom_dataset_info.json
│
├── scripts/                      # 训练脚本
│   ├── train_rm_full_content_7gpu.sh    # RM 训练（推荐）
│   ├── train_grpo.sh                    # GRPO 训练
│   ├── convert_dpo_to_rm_swift_format.py
│   ├── analyze_rm_token_distribution.py
│   └── ...
│
├── train/code/                   # 训练代码
│   ├── reward_function.py        # 格式奖励函数
│   ├── plugin.py                 # GRPO 插件
│   └── rm_plugin.py              # RM 插件
│
├── data/openreview_dataset/      # 数据集
│   ├── rm_train.json             # RM 训练数据
│   ├── rm_val.json               # RM 验证数据
│   └── ...
│
├── docs/                         # 文档
│   ├── Reward_Model_Training_Guide.md  ⭐
│   ├── GRPO_Training_README.md
│   └── ...
│
├── logs/                         # 训练日志
└── models/                       # 模型输出
    ├── qwen3-8b-base/           # 基础模型
    └── reward_model_qwen3_8b/   # RM 输出
```

---

## 训练进度

### 当前状态

**Reward Model 训练**：
- ✅ 数据准备完成（5000 samples）
- ✅ 训练脚本配置完成
- ⏳ 训练进行中

**配置**：
- 模型: Qwen3-8B
- max_length: 16384
- batch_size: 2
- GPU: 7 (跳过 GPU 4)
- 显存使用: ~77 GiB (96% 利用率)

**监控**：
- WandB: https://wandb.ai/liverspecial-dlut-edu-cn/reward_model_grpo
- 日志: `logs/rm_training_*.log`

### 下一步

- [ ] 完成 RM 训练（预计 8 小时）
- [ ] 评估 RM 模型
- [ ] 准备 GRPO 数据
- [ ] 启动 GRPO 训练

---

## 快速命令

### 检查训练状态

```bash
# 查看 GPU
nvidia-smi

# 查看日志
tail -f logs/rm_training_*.log

# 检查进程
ps aux | grep "swift rlhf"

# 检查 checkpoint
ls -lh models/reward_model_qwen3_8b/v0-*/checkpoint-*/
```

### 停止训练

```bash
# 查找进程
ps aux | grep "swift rlhf" | grep -v grep

# 停止进程
pkill -f "swift/cli/rlhf.py"
```

### 重新开始

```bash
# 清理旧模型
rm -rf models/reward_model_qwen3_8b

# 重新训练
bash scripts/train_rm_full_content_7gpu.sh
```

---

## 常见问题

### Q: 如何选择 max_length？

**A**: 参考文档 [RM_QUERY_STRATEGY_FINAL.md](docs/RM_QUERY_STRATEGY_FINAL.md)

- 16384: 覆盖 99.4% 样本（推荐）
- 12288: 覆盖 95% 样本（显存紧张时）
- 20000: 覆盖 99.5% 样本（显存充足时）

### Q: 显存不足怎么办？

**A**: 参考 [Reward_Model_Training_Guide.md](docs/Reward_Model_Training_Guide.md#常见问题)

```bash
# 方案 1: 减小 batch size
--per_device_train_batch_size 1

# 方案 2: 减小 max_length
--max_length 12288

# 方案 3: 使用 ZeRO-3
--deepspeed configs/deepspeed_zero3_config.json
```

### Q: 训练速度慢怎么办？

**A**: 

```bash
# 增大 batch size（如果显存允许）
--per_device_train_batch_size 4
--gradient_accumulation_steps 4

# 关闭 gradient checkpointing（如果显存允许）
--gradient_checkpointing false
```

### Q: Loss 不下降怎么办？

**A**: 

```bash
# 调整学习率
--learning_rate 2e-5  # 尝试 5e-6 ~ 5e-5

# 检查数据质量
python scripts/analyze_rm_token_distribution.py
```

---

## 许可证

MIT License

---

## 更新日志

### 2026-03-06
- ✅ 创建完整 RM 训练指南
- ✅ 优化显存利用率（96%）
- ✅ 更新文档结构
- ✅ 添加快速开始指南
