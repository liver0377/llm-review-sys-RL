# 论文评审强化学习训练系统

> 基于 veRL 框架的 GRPO 强化学习训练系统，用于优化论文评审生成模型

---

## 快速开始

### 1. 环境准备

```bash
# 激活 veRL 环境
conda activate verl

# 验证安装
python -c "import verl; print('veRL version:', verl.__version__)"
```

### 2. 数据准备

```bash
# 转换数据为 veRL 格式（parquet）
bash scripts/prepare_data.sh
```

### 3. 启动 Reward Model 服务

训练前需要先启动 Reward Model 的 vLLM 服务：

```bash
# 在 GPU 4-7 上启动 RM 服务
bash scripts/start_rm_service.sh

# 检查服务状态
curl http://127.0.0.1:8002/health
```

**服务配置：**
- 模型: Qwen3-8B-Base
- GPU: 4,5,6,7 (Tensor Parallel = 4)
- 端口: 8002

### 4. 启动 GRPO 训练

```bash
# 在 GPU 0-3 上启动训练
bash scripts/train_grpo.sh

# 查看训练日志
tail -f logs/grpo_training_*.log
```

**训练配置：**
- 框架: veRL 0.8.0
- 算法: GRPO
- 策略模型: Qwen3-8B-Base
- GPU: 0,1,2,3 (FSDP + vLLM)
- 数据: data/openreview_dataset/train.parquet
- 输出: models/grpo_qwen3_8b_verl/

---

## 项目结构

```
llm-review-sys-RL/
├── scripts/                      # 训练脚本
│   ├── prepare_data.sh           # 数据预处理
│   ├── start_rm_service.sh       # 启动 RM 服务
│   ├── stop_rm_service.sh        # 停止 RM 服务
│   ├── train_grpo.sh             # 训练主脚本
│   ├── run_grpo_verl.sh          # veRL 配置
│   └── data/
│       └── prepare_openreview_parquet.py
│
├── train/code/                   # 训练代码
│   └── reward_function.py        # External RM Reward Function
│
├── data/openreview_dataset/      # 数据集
│   ├── grpo_train.json           # 原始训练数据
│   ├── train.parquet             # veRL 格式数据
│   └── ...
│
├── docs/                         # 文档
│   └── RL_SYSTEM_GUIDE.md        # 系统详细说明
│
├── logs/                         # 训练日志
│
└── models/                       # 模型输出
    ├── Qwen3-8B-Base/           # 基础模型
    └── grpo_qwen3_8b_verl/      # GRPO 训练输出
```

---

## 核心特性

### 1. veRL 框架

使用 ByteDance 开发的 veRL 框架：
- **Hybrid Controller**: 混合控制器编程模型
- **FSDP**: PyTorch Fully Sharded Data Parallel
- **vLLM**: 高效推理引擎集成
- **Production-Ready**: 生产级性能优化

### 2. GRPO 算法

Group Relative Policy Optimization：
- **无需 Critic**: 不需要训练 Value Network
- **组内归一化**: 在同组 response 间计算优势
- **KL 正则化**: 稳定的策略优化

### 3. External RM Service

分离式 Reward Model 架构：
- **独立部署**: RM 和 Policy 在不同 GPU 组
- **异步批量**: 高效的并发 Reward 计算
- **灵活扩展**: 可独立扩展和监控

---

## 训练流程

### 完整流程

```bash
# Step 1: 准备数据
bash scripts/prepare_data.sh

# Step 2: 启动 RM 服务（GPU 4-7）
bash scripts/start_rm_service.sh

# Step 3: 等待 RM 服务就绪
curl http://127.0.0.1:8002/health

# Step 4: 启动训练（GPU 0-3）
bash scripts/train_grpo.sh

# Step 5: 监控训练
tail -f logs/grpo_training_*.log
```

### 并行策略

```
GPU 0-3: Policy Model
├─ FSDP 分布式训练
├─ vLLM 推理引擎
└─ Tensor Parallel = 4

GPU 4-7: Reward Model
├─ vLLM 推理服务
├─ Tensor Parallel = 4
└─ 异步批量处理
```

---

## 详细文档

完整的系统说明、算法详解、配置调优等内容，请查看：

**[系统详细说明文档](docs/RL_SYSTEM_GUIDE.md)**

包含内容：
- GRPO 算法原理与实现
- veRL 框架特性详解
- Reward 计算机制
- 训练配置详解
- 性能优化指南
- 监控与调试
- 常见问题解答

---

## 常见问题

### Q: veRL 和 swift 有什么区别？

A: veRL 是专门的 RL 训练框架，特性包括：
- 混合控制器编程模型
- FSDP + vLLM 无缝集成
- 生产级性能优化
- 更灵活的分布式配置

### Q: 训练时显存不足？

调整配置参数：
```bash
# 在 scripts/run_grpo_verl.sh 中修改
data.train_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

### Q: RM 服务无响应？

检查服务状态：
```bash
curl http://127.0.0.1:8002/health
tail -f logs/rm_service.log
```

---

## 技术栈

- **框架**: veRL 0.8.0 (ByteDance)
- **模型**: Qwen3-8B-Base
- **训练引擎**: PyTorch FSDP
- **推理引擎**: vLLM
- **算法**: GRPO
- **数据格式**: Parquet

---

## 参考资料

- [veRL Documentation](https://verl.readthedocs.io/)
- [GRPO Paper](https://arxiv.org/pdf/2402.03300)
- [veRL GitHub](https://github.com/volcengine/verl)

---

## 更新日志

### 2026-03-10
- ✅ 完全迁移到 veRL 框架
- ✅ 实现 External RM Reward Function
- ✅ 优化训练配置
- ✅ 创建完整的系统说明文档

### 2026-03-09
- ✅ 删除所有 swift 相关代码
- ✅ 清理项目结构