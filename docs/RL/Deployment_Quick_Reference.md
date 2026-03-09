# GRPO 训练：部署方式快速参考

**更新时间**: 2026-03-09

---

## 🎯 两种部署方式

### 方案对比总览

| 特性 | 内部部署 (Colocate) | 外部部署 (Launch) |
|------|---------------------|-----------------|
| **脚本** | `train_grpo_GRM.sh` | `train_grpo_GRM_external.sh` |
| **启动方式** | 一键启动 | 两步启动 |
| **GPU 需求** | 8× GPU | 8× GPU (4+4 分配) |
| **进程数** | 1 个 | 2 个 |
| **通信** | 内存 (TransformersEngine) | HTTP (OpenAI Client) |
| **训练速度** | ~47.6s/step | ~42s/step |
| **RM 评估** | ~7.6s/评估 | ~2s/评估 |
| **复杂度** | 简单 | 中等 |
| **推荐度** | ⭐⭐⭐⭐ 快速原型 | ⭐⭐⭐⭐⭐ 生产训练 |

---

## 🚀 快速启动指南

### 内部部署（一键启动）

```bash
# 直接启动训练
bash scripts/train_grpo_GRM.sh
```

**特点**：
- ✅ 简单：一行命令
- ✅ 自动管理所有组件
- ✅ 适合快速开发

### 外部部署（两步启动）

```bash
# Step 1: 启动 RM 服务
bash scripts/start_rm_service.sh

# Step 2: 启动训练
bash scripts/train_grpo_GRM_external.sh
```

**特点**：
- ✅ 快 12% 训练速度
- ✅ RM 评估快 3.8×
- ✅ 显存分配更清晰

---

## 📊 关键差异

### 1. GPU 分配

**内部部署**：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 全部 8 张卡共享
```

**外部部署**：
```bash
# 训练进程
CUDA_VISIBLE_DEVICES=0,1,2,3

# RM 服务
CUDA_VISIBLE_DEVICES=4,5,6,7
```

### 2. 关键参数

| 参数 | 内部部署 | 外部部署 |
|------|---------|---------|
| `--vllm_mode` | `colocate` | `launch` |
| `--reward_model_plugin` | `genrm_plugin` | `genrm_plugin_external` |
| `--external_plugins` | `genrm_plugin.py` | `genrm_plugin_external.py` |
| `--vllm_gpu_memory_utilization` | `0.3` | `0.4` |
| `--reward_model_api_base` | (无) | `http://127.0.0.1:8000/v1` |

### 3. 服务管理

**内部部署**：
```bash
# 训练和服务在一起，无需额外管理
# 训练结束即停止
```

**外部部署**：
```bash
# 需要手动管理 RM 服务
# 启动: bash scripts/start_rm_service.sh
# 停止: bash scripts/stop_rm_service.sh
```

---

## 🎛️ 故障排查

### 内部部署问题

**Q: 训练速度慢**
```
A: 这是正常的（47.6s/step）
   如需更快，切换到外部部署
```

**Q: OOM 错误**
```bash
--vllm_gpu_memory_utilization 0.2  # 降低到 0.2
--gradient_accumulation_steps 16     # 增加累积
```

### 外部部署问题

**Q: RM 服务启动失败**
```bash
# 检查日志
tail -100 logs/rm_service.log

# 检查端口
lsof -i :8000

# 重启服务
bash scripts/stop_rm_service.sh
bash scripts/start_rm_service.sh
```

**Q: 训练无法连接 RM**
```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 检查服务状态
ps aux | grep vllm
```

---

## 📈 性能对比

### 时间分解

| 阶段 | 内部 | 外部 | 差异 |
|------|------|------|------|
| 生成阶段 | 30s | 30s | - |
| **RM 评估** | **7.6s** | **2s** | **↑3.8×** |
| 优势计算 | 1s | 1s | - |
| 损失计算 | 1s | 1s | - |
| 反向传播 | 8s | 8s | - |
| **总计** | **47.6s** | **42s** | **↑12%** |

### 显存占用

| 组件 | 内部 | 外部 | 说明 |
|------|------|------|------|
| Policy Model | 16GB | 16GB | - |
| vLLM KV Cache | 24GB | 32GB | 外部需要更多缓存 |
| Reward Model | 70GB | 70GB | - |
| **总计** | **110GB (共享)** | **118GB (分离)** | 差异 8GB |

---

## 🎯 选择建议

### 使用内部部署，如果：

- ✅ 快速原型开发
- ✅ 单机 8× GPU
- ✅ 不想管理多个服务
- ✅ 追求简单易用

### 使用外部部署，如果：

- ✅ 追求训练速度（快 12%）
- ✅ 有多个 GPU 集群
- ✅ 生产环境训练
- ✅ 需要 RM 评估加速

---

## 🔄 切换方案

### 从内部 → 外部

```bash
# 1. 停止内部训练（如果在运行）
# Ctrl+C

# 2. 启动 RM 服务
bash scripts/start_rm_service.sh

# 3. 测试外部部署
python scripts/test_external_deployment.py

# 4. 启动外部训练
bash scripts/train_grpo_GRM_external.sh
```

### 从外部 → 内部

```bash
# 1. 停止训练（如果在运行）
# Ctrl+C

# 2. 停止 RM 服务
bash scripts/stop_rm_service.sh

# 3. 启动内部训练
bash scripts/train_grpo_GRM.sh
```

---

## 📚 相关文档

- **对比文档**: [docs/RL/Internal_vs_External_Deployment.md](../docs/RL/Internal_vs_External_Deployment.md)
- **内部脚本**: [scripts/train_grpo_GRM.sh](../scripts/train_grpo_GRM.sh)
- **外部脚本**: [scripts/train_grpo_GRM_external.sh](../scripts/train_grpo_GRM_external.sh)
- **测试脚本**: [scripts/test_external_deployment.py](../scripts/test_external_deployment.py)

---

**更新完成**: 2026-03-09
**脚本状态**: ✅ 已创建并可执行
