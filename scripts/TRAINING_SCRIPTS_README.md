# GRPO Training Scripts - 使用说明

本项目提供了两个训练脚本，都使用外部部署的生成式奖励模型（Generative RM）。

## 脚本对比

| 特性 | 使用 vLLM | 不使用 vLLM |
|------|-----------|-------------|
| 脚本名称 | `train_grpo_GRM_external.sh` | `train_grpo_GRM_external_novllm.sh` |
| 数据生成速度 | ⚡ 快 (~12% 加速) | 🐌 慢 (原生生成) |
| 显存占用 | 💾 需要额外显存 | 💾 仅训练模型 |
| 初始化 | 🔄 需要启动 vLLM 服务 | ✅ 直接开始 |
| GPU 配置 | GPU 0-3: 训练<br>GPU 4-7: RM<br>GPU 1-3: vLLM生成 | GPU 1-3: 训练<br>GPU 4-7: RM |
| 推荐场景 | 大规模训练 | 快速测试/调试 |

## 详细说明

### 1. 使用 vLLM 生成 (`train_grpo_GRM_external.sh`)

**优点：**
- 数据生成速度快约 12%
- 适合大规模训练

**缺点：**
- 需要额外启动 vLLM 服务（端口 8001）
- 占用更多显存（GPU 1-3 需要加载 vLLM 服务）
- 初始化时间更长
- 如果 vLLM 服务启动失败，训练会卡住

**资源需求：**
- GPU 0: 可能被其他进程占用
- GPU 1-3: 训练 + vLLM 生成服务
- GPU 4-7: RM 服务

**使用方法：**
```bash
bash scripts/train_grpo_GRM_external.sh
```

**注意事项：**
- 确保 GPU 1,2,3 有足够显存（每个至少 ~40GB 可用）
- 如果脚本卡住，检查 vLLM 服务是否启动：`lsof -i :8001`

---

### 2. 不使用 vLLM 生成 (`train_grpo_GRM_external_novllm.sh`) ⭐ 推荐

**优点：**
- 无需额外配置，直接开始训练
- 显存占用少
- 初始化简单
- 适合快速测试和调试

**缺点：**
- 数据生成速度较慢（但 RM 评估仍然很快）
- 总训练时间可能增加 10-15%

**资源需求：**
- GPU 1-3: 训练
- GPU 4-7: RM 服务

**使用方法：**
```bash
bash scripts/train_grpo_GRM_external_novllm.sh
```

**推荐理由：**
- 更简单，更不容易出错
- RM 服务（外部部署）已经提供了主要的速度提升
- 对于大多数场景，性能差异可接受

---

## 故障排除

### 问题 1: 训练卡住，提示 "Servers not reachable"

**原因：** 使用 vLLM 脚本时，vLLM 生成服务（端口 8001）未启动

**解决方法：**
1. 检查端口占用：`lsof -i :8001`
2. 尝试使用不使用 vLLM 的脚本：`bash scripts/train_grpo_GRM_external_novllm.sh`

### 问题 2: RM 服务未启动

**解决方法：**
```bash
# 手动启动 RM 服务
bash scripts/start_rm_service.sh

# 检查 RM 服务状态
curl http://127.0.0.1:8002/health
```

### 问题 3: 显存不足

**解决方法：**
- 使用不使用 vLLM 的脚本（节省显存）
- 或者减少 `per_device_train_batch_size` 和 `gradient_accumulation_steps`

---

## 当前状态

根据你的环境：
- ✅ RM 服务（端口 8002）已启动
- ❌ vLLM 生成服务（端口 8001）未启动
- GPU 1,2,3: 空闲（可用于训练）

**建议：先使用不使用 vLLM 的脚本测试**
```bash
bash scripts/train_grpo_GRM_external_novllm.sh
```

如果效果满意，后续可以考虑使用 vLLM 版本加速训练。

---

## 性能对比参考

| 版本 | RM 评估时间 | 生成时间 | 总训练时间 |
|------|------------|---------|-----------|
| 内部部署（不使用外部 RM） | ~7.6s | 慢 | 基线 |
| 外部 RM（无 vLLM 生成） | ~2s | 慢 | ~快 40% |
| 外部 RM（有 vLLM 生成） | ~2s | 快 | ~快 50% |

从内部部署到外部部署（无 vLLM）已经有 **~40% 的加速**，使用 vLLM 生成再额外提升 **~10%**。
