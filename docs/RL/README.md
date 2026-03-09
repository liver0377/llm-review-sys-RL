# RL 训练文档目录

本目录包含强化学习（RL）训练相关的技术文档。

## 📚 文档列表

### [GRPO训练技术文档](./GRPO_Training_Technical_Guide.md) ⭐ 推荐

**完整的 GRPO 训练技术指南，包含：**

- 系统架构和数据流
- 核心概念（GRPO算法、奖励机制、Plugin系统）
- 两种训练方案详解：
  - **方案A：生成式 RM**（推荐）- 使用 Qwen3.5-35B-A3B
  - **方案B：传统训练 RM** - 训练 8B 奖励模型
- 完整代码实现和配置参考
- 性能优化策略
- 故障排查指南

**适合读者**：开发者、研究员

---

## 🚀 快速开始

### 使用生成式 RM（推荐）

```bash
# 1. 下载生成式 RM
bash scripts/download_qwen35_35b_a3b.sh

# 2. 启动训练
bash scripts/train_grpo_GRM.sh
```

### 使用传统 RM

```bash
# 1. 训练奖励模型 + GRPO
bash scripts/train_grpo_pipeline.sh
```

---

## 📖 文档结构

```
docs/RL/
├── README.md                          # 本文件
└── GRPO_Training_Technical_Guide.md   # GRPO 训练技术文档
```

---

## 🔗 相关文档

- [../GRPO_Training_README.md](../GRPO_Training_README.md) - GRPO 基础指南
- [../GRPO_RL_Pipeline.md](../GRPO_RL_Pipeline.md) - GRPO 流水线说明
- [../generative_rm_guide.md](../generative_rm_guide.md) - 生成式 RM 使用指南
- [../GRM_script_comparison.md](../GRM_script_comparison.md) - 脚本对比
- [../swift/swift-GenRM.md](../swift/swift-GenRM.md) - Swift GenRM 文档

---

## 💡 常见问题

### Q: 应该选择哪种训练方案？

**A:** 推荐使用 **方案A（生成式 RM）**，原因：
- ✅ 无需训练 RM，节省时间
- ✅ 使用 35B 大模型，质量更高
- ✅ 实现简单，易于调试

如果显存不足（<8× A100 80GB），则使用 **方案B（传统 RM）**。

### Q: 训练需要多长时间？

**A:**
- **方案A**（生成式 RM）：约 8-16 小时
- **方案B**（传统 RM）：约 10-16 小时（含 RM 训练 4-6 小时）

### Q: 需要多少显存？

**A:**
- **方案A**：~120GB（建议 8× A100 80GB）
- **方案B**：~80GB（可用 8× A100 40GB）

### Q: 如何调整奖励权重？

**A:** 使用 `--alpha` 和 `--format_weight` 参数：

```bash
# 重视质量
--alpha 2.0 --format_weight 0.5

# 平衡（推荐）
--alpha 1.0 --format_weight 1.0

# 重视格式
--alpha 0.5 --format_weight 2.0
```

---

## 📞 获取帮助

如果遇到问题：

1. 查看 [GRPO_Training_Technical_Guide.md](./GRPO_Training_Technical_Guide.md) 的故障排查章节
2. 检查 [GitHub Issues](https://github.com/modelscope/ms-swift/issues)
3. 加入 [Swift Discord](https://discord.gg/modelscope)

---

**最后更新**: 2026-03-09
