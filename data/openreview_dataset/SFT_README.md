# OpenReview SFT Dataset

## 概述

本数据集用于学术论文评审生成任务的监督微调（SFT）。

## 数据集格式

每条数据包含三个字段：
```json
{
  "instruction": "审稿人指令模板（固定）",
  "input": "论文内容（标题+会议+完整正文）",
  "output": "聚合评审+评分"
}
```

## 数据集划分

- **训练集** (sft_train.json): 85%
- **验证集** (sft_val.json): 10%
- **测试集** (sft_test.json): 5%

## 使用方法

```python
from datasets import load_dataset

train_dataset = load_dataset('openreview_dataset', data_files='sft_train.json')['train']
val_dataset = load_dataset('openreview_dataset', data_files='sft_val.json')['train']
test_dataset = load_dataset('openreview_dataset', data_files='sft_test.json')['train']
```

## 数据来源

- **论文内容**: 使用 Marker PDF 解析工具从 PDF 提取
- **评审内容**: 从 OpenReview API 获取并使用 GPT-4 聚合

## 字段说明

### instruction
固定的审稿人指令模板，包含：
- Key Points（主要贡献）
- Strengths and Weaknesses（优缺点）
- Suggestions for Improvement（改进建议）
- Rating（评分）

### input
论文的详细信息：
- 标题
- 会议和年份
- 完整的论文内容（从 Marker PDF 提取）

### output
结构化的评审内容：
- 聚合后的评审文本
- Overall Quality（总体质量，1-10分）
- Review Confidence（评审置信度，1-5分）

## 训练建议

1. **学习率**: 建议使用 2e-5 到 5e-5
2. **Batch Size**: 根据 GPU 内存调整，建议 1-4
3. **Epochs**: 建议 2-3 个 epoch
4. **最大长度**: 根据模型上下文窗口，建议 4096-8192
5. **LoRA**: 推荐使用 LoRA 进行参数高效微调

## 许可证

Apache 2.0
