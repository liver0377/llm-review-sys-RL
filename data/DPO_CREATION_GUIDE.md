# DPO 数据集创建指南

## 快速开始

### 方式 1：快速测试（不调用 API）

用于测试流程，`rejected` 字段使用占位符。

```bash
cd /data/wudy/RL/llm-review-sys/data
conda activate openreview
python create_dpo_dataset.py
```

**输出**：
- `dpo_train.json` - 训练集（rejected 为占位符）
- `dpo_val.json` - 验证集（rejected 为占位符）
- `dpo_test.json` - 测试集（rejected 为占位符）

---

### 方式 2：一步生成完整数据集（推荐）

使用 DashScope API 生成完整的 `rejected` 字段。

```bash
python create_dpo_dataset.py --generate-rejected
```

**特点**：
- ✅ 一次性生成包含 `chosen` 和 `rejected` 的完整数据集
- ✅ 支持断点续传（Ctrl+C 中断后可继续）
- ✅ 实时保存进度（每个样本都保存）
- ✅ 自动重试失败的样本

**预计耗时**：约 30-60 分钟（取决于数据量）

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--generate-rejected` | 使用 DashScope API 生成 rejected | `False` |
| `--no-rejected` | 明确不生成 rejected（快速测试）| `False` |
| `--help` | 显示帮助信息 | - |

---

## 配置要求

### 1. 环境变量

确保项目根目录的 `.env` 文件包含：

```bash
DASHSCOPE_API_KEY="sk-25587b057d5242428bb940d44035b5fd"
DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 2. Python 环境

```bash
conda activate openreview
pip install openai python-dotenv tqdm
```

### 3. 数据准备

确保已运行 `pdf_parser_marker.py` 生成 Markdown 解析结果。

---

## 断点续传

### 自动恢复

如果中途中断（Ctrl+C 或 API 错误），重新运行相同命令即可自动恢复：

```bash
python create_dpo_dataset.py --generate-rejected
```

脚本会自动：
- 读取进度文件 `dpo_generation_progress.json`
- 跳过已处理的样本
- 继续处理剩余样本

### 进度文件位置

```
data/openreview_dataset/dpo_generation_progress.json
```

### 重置进度

如需重新开始，删除进度文件：

```bash
rm data/openreview_dataset/dpo_generation_progress.json
python create_dpo_dataset.py --generate-rejected
```

---

## 输出格式

### 数据集结构

每条数据包含三个字段：

```json
{
  "prompt": "审稿指令 + 论文内容",
  "chosen": "高质量评审（聚合多个评审）+ 评分",
  "rejected": "低质量评审（类似 chosen 但缺少关键审稿点）"
}
```

### rejected 生成策略

DashScope API 会生成一个：
- **风格类似** chosen 的评审
- **专业、完整**（看起来像真实评审）
- **但故意省略** 1-2 个关键审稿点，例如：
  - 实验设计不充分
  - 缺乏对比方法
  - 理论假设不成立
  - 结论超出实验支持范围
  - 数据集设置有偏差

---

## 成本估算

### DashScope API 定价

- **模型**：qwen-turbo
- **输入**：按 tokens 计费
- **输出**：按 tokens 计费

### 预估成本

假设：
- 数据集大小：500 条
- 平均输入 tokens：3000
- 平均输出 tokens：1500
- qwen-turbo 价格：¥0.57/1M tokens（输入）+ ¥2/1M tokens（输出）

**计算**：
```
输入成本 = 500 × 3000 × ¥0.57 / 1,000,000 ≈ ¥0.86
输出成本 = 500 × 1500 × ¥2 / 1,000,000 ≈ ¥1.50
总成本 ≈ ¥2.36
```

**实际成本可能更低**，因为 rejected 通常比 chosen 短。

---

## 使用示例

### 示例 1：快速测试流程

```bash
# 1. 快速生成占位符版本（1 分钟）
python create_dpo_dataset.py

# 2. 检查输出
ls -lh data/openreview_dataset/dpo_*.json

# 3. 查看样本
head -100 data/openreview_dataset/dpo_train.json | python -m json.tool
```

### 示例 2：生成完整数据集

```bash
# 1. 一步生成完整数据集（30-60 分钟）
python create_dpo_dataset.py --generate-rejected

# 2. 查看进度
tail -f data/openreview_dataset/dpo_generation_progress.json

# 3. 验证 rejected 质量
python -c "
import json
with open('data/openreview_dataset/dpo_train.json', 'r') as f:
    data = json.load(f)
    print('=== Chosen ===')
    print(data[0]['chosen'][:500])
    print('\n=== Rejected ===')
    print(data[0]['rejected'][:500])
"
```

### 示例 3：从断点恢复

```bash
# 运行过程中断（Ctrl+C）
# ...

# 重新运行（自动恢复）
python create_dpo_dataset.py --generate-rejected

# 输出：
# 🔄 恢复进度: 已处理 234/531 条
```

---

## 故障排查

### 问题 1：API 连接失败

**错误信息**：
```
❌ API 初始化失败: DASHSCOPE_API_KEY not found in environment
```

**解决方案**：
```bash
# 检查 .env 文件
cat /data/wudy/RL/llm-review-sys/.env | grep DASHSCOPE

# 确保 API key 配置正确
# 文件位置：项目根目录的 .env 文件
```

### 问题 2：生成失败率高

**原因**：API 速率限制或网络问题

**解决方案**：
1. 检查网络连接
2. 增加延迟时间（修改 `time.sleep(0.5)` 为 `time.sleep(1.0)`）
3. 失败的样本会自动使用占位符，不影响流程

### 问题 3：进度文件损坏

**解决方案**：
```bash
# 删除进度文件重新开始
rm data/openreview_dataset/dpo_generation_progress.json
python create_dpo_dataset.py --generate-rejected
```

---

## 与原始流程的对比

### 原始流程（两步）

```bash
# Step 1: 创建占位符版本
python create_dpo_dataset.py

# Step 2: 使用 dpo_rejected_modifier.py 生成 rejected
python dpo_rejected_modifier.py
```

**缺点**：
- 需要运行两个脚本
- 需要手动管理中间文件
- 断点续传需要单独处理

### 新流程（一步）

```bash
# 一步完成
python create_dpo_dataset.py --generate-rejected
```

**优点**：
- ✅ 一个命令完成所有操作
- ✅ 自动支持断点续传
- ✅ 实时保存进度
- ✅ 更简洁、更易用

---

## 保留的独立脚本

`dpo_rejected_modifier.py` 仍然保留，可用于：

1. 单独修改已存在的 DPO 数据集
2. 批量处理多个数据集
3. 自定义修改逻辑

使用方式：
```bash
python dpo_rejected_modifier.py
```

---

## 版本历史

- **2026-02-22**: 集成 DashScope API，支持一步生成完整 DPO 数据集
- **原始版本**: 分两步，先创建占位符，再使用 DeepSeek API 修改

---

## 联系方式

如有问题或建议，请联系项目维护者。
