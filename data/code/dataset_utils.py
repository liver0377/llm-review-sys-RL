#!/usr/bin/env python3
"""
数据集构建工具模块

提供共享的数据加载、匹配、划分和保存功能。
供 create_sft_dataset.py 和 create_dpo_dataset.py 使用。
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


# ============== 配置类 ==============


class DatasetConfig:
    """数据集构建配置"""

    # 输入路径
    BASE_DIR = Path(__file__).parent.parent
    AGGREGATED_REVIEWS_PATH = (
        BASE_DIR / "data" / "aggregated_reviews" / "aggregated_reviews.json"
    )
    PARSED_TEXTS_DIR = BASE_DIR / "data" / "filtered_pdf_texts_marker"
    OUTPUT_DIR = BASE_DIR / "data" / "openreview_dataset"

    # 数据集划分比例
    TRAIN_RATIO = 0.85
    VAL_RATIO = 0.10
    TEST_RATIO = 0.05

    # 过滤选项
    MIN_RATING_COUNT = 1  # 最少评审数量
    MIN_AVG_RATING = 1.0  # 最低平均评分
    MAX_AVG_RATING = 10.0  # 最高平均评分

    # 随机种子
    RANDOM_SEED = 42


# ============== 数据加载函数 ==============


def load_aggregated_reviews(file_path: Path = None) -> List[Dict]:
    """
    加载聚合评审数据

    Args:
        file_path: aggregated_reviews.json 文件路径
                   如果为 None，使用配置中的默认路径

    Returns:
        评审数据列表
    """
    if file_path is None:
        file_path = DatasetConfig.AGGREGATED_REVIEWS_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"❌ 聚合评审文件不存在: {file_path}")

    print(f"📄 加载聚合评审数据: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    print(f"✅ 加载了 {len(reviews)} 条评审数据")
    return reviews


def load_marker_md_files(parsed_texts_dir: Path = None) -> Dict[str, Dict]:
    """
    加载 Marker 解析的 Markdown 文件

    Args:
        parsed_texts_dir: parsed_texts_marker 目录路径
                         如果为 None，使用配置中的默认路径

    Returns:
        {paper_id: {title, content, file_path}, ...}
    """
    if parsed_texts_dir is None:
        parsed_texts_dir = DatasetConfig.PARSED_TEXTS_DIR

    if not parsed_texts_dir.exists():
        raise FileNotFoundError(f"❌ Marker 输出目录不存在: {parsed_texts_dir}")

    print(f"\n📂 扫描 Marker 输出目录: {parsed_texts_dir}")

    marker_papers = {}
    total_files = 0

    # 遍历所有会议目录
    for conf_dir in sorted(parsed_texts_dir.iterdir()):
        if not conf_dir.is_dir():
            continue

        print(f"  📁 处理会议: {conf_dir.name}")

        # 查找所有 .md 文件（递归搜索子目录）
        for md_file in conf_dir.rglob("*.md"):
            total_files += 1
            paper_id = md_file.stem  # 文件名不含扩展名

            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # 提取标题（第一个 # 标题）
                title = ""
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

                marker_papers[paper_id] = {
                    "id": paper_id,
                    "title": title,
                    "content": content,
                    "file_path": str(md_file),
                    "conference": conf_dir.name,
                }

            except Exception as e:
                print(f"  ⚠️  读取文件失败 {md_file}: {e}")

    print(
        f"\n✅ 成功加载 {len(marker_papers)} 个 Marker 解析文件（总共扫描 {total_files} 个文件）"
    )
    return marker_papers


# ============== 数据处理函数 ==============


def match_reviews_and_papers(
    reviews: List[Dict], marker_papers: Dict[str, Dict]
) -> List[Dict]:
    """
    匹配聚合评审和 Marker 解析的论文

    Args:
        reviews: 聚合评审数据列表
        marker_papers: Marker 解析的论文字典

    Returns:
        匹配后的数据列表
    """
    print(f"\n🔗 匹配评审和论文...")

    matched_data = []
    unmatched_reviews = []
    unmatched_papers = set(marker_papers.keys())

    for review in tqdm(reviews, desc="匹配数据"):
        paper_id = review.get("id")

        if not paper_id:
            unmatched_reviews.append(paper_id)
            continue

        if paper_id in marker_papers:
            paper = marker_papers[paper_id]

            # 计算平均评分和置信度
            original_ratings = review.get("original_ratings", [])
            original_confidences = review.get("original_confidences", [])

            valid_ratings = [r for r in original_ratings if r != -1]
            valid_confidences = [c for c in original_confidences if c != -1]

            if not valid_ratings or not valid_confidences:
                unmatched_reviews.append(paper_id)
                continue

            avg_rating = sum(valid_ratings) / len(valid_ratings)
            avg_confidence = sum(valid_confidences) / len(valid_confidences)

            # 构建合并后的数据
            matched_item = {
                "id": paper_id,
                "title": review.get("title", paper.get("title", "")),
                "conference": review.get("conference", paper.get("conference", "")),
                "year": review.get("year", ""),
                "paper_content": paper["content"],
                "aggregated_review": review.get("aggregated_review", ""),
                "original_ratings": original_ratings,
                "original_confidences": original_confidences,
                "avg_rating": round(avg_rating, 2),
                "avg_confidence": round(avg_confidence, 2),
                "reviews_count": review.get("reviews_count", len(original_ratings)),
            }

            matched_data.append(matched_item)
            unmatched_papers.discard(paper_id)
        else:
            unmatched_reviews.append(paper_id)

    print(f"\n✅ 匹配结果:")
    print(f"  成功匹配: {len(matched_data)} 条")
    print(f"  未匹配的评审: {len(unmatched_reviews)} 条")
    print(f"  未匹配的论文: {len(unmatched_papers)} 篇")

    if len(unmatched_reviews) > 0:
        print(f"\n⚠️  前10个未匹配的 paper_id: {unmatched_reviews[:10]}")

    return matched_data


def split_dataset(
    dataset: List[Dict],
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    random_seed: int = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    划分数据集为 train/val/test

    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        (train_data, val_data, test_data)
    """
    # 使用配置中的默认值
    if train_ratio is None:
        train_ratio = DatasetConfig.TRAIN_RATIO
    if val_ratio is None:
        val_ratio = DatasetConfig.VAL_RATIO
    if test_ratio is None:
        test_ratio = DatasetConfig.TEST_RATIO
    if random_seed is None:
        random_seed = DatasetConfig.RANDOM_SEED

    print(
        f"\n🎲 划分数据集 (train:{train_ratio}, val:{val_ratio}, test:{test_ratio})..."
    )
    print(f"   随机种子: {random_seed}")

    # 设置随机种子
    random.seed(random_seed)

    # 随机打乱数据
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # 剩余的作为测试集

    train_data = shuffled[:train_size]
    val_data = shuffled[train_size : train_size + val_size]
    test_data = shuffled[train_size + val_size :]

    print(f"✅ 划分完成:")
    print(f"  训练集: {len(train_data)} 条 ({len(train_data) / total * 100:.1f}%)")
    print(f"  验证集: {len(val_data)} 条 ({len(val_data) / total * 100:.1f}%)")
    print(f"  测试集: {len(test_data)} 条 ({len(test_data) / total * 100:.1f}%)")

    return train_data, val_data, test_data


# ============== 数据保存函数 ==============


def save_json(data: List[Dict], file_path: Path):
    """
    保存 JSON 文件

    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_sft_datasets(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: Path = None,
):
    """
    保存 SFT 数据集

    Args:
        train_data: 训练集
        val_data: 验证集
        test_data: 测试集
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = DatasetConfig.OUTPUT_DIR

    print(f"\n💾 保存 SFT 数据集到: {output_dir}")

    datasets = {
        "sft_train.json": train_data,
        "sft_val.json": val_data,
        "sft_test.json": test_data,
    }

    for filename, data in datasets.items():
        file_path = output_dir / filename
        save_json(data, file_path)
        print(f"  ✅ {filename}: {len(data)} 条记录")


def save_dpo_datasets(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: Path = None,
):
    """
    保存 DPO 数据集

    Args:
        train_data: 训练集
        val_data: 验证集
        test_data: 测试集
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = DatasetConfig.OUTPUT_DIR

    print(f"\n💾 保存 DPO 数据集到: {output_dir}")

    datasets = {
        "dpo_train.json": train_data,
        "dpo_val.json": val_data,
        "dpo_test.json": test_data,
    }

    for filename, data in datasets.items():
        file_path = output_dir / filename
        save_json(data, file_path)
        print(f"  ✅ {filename}: {len(data)} 条记录")


def generate_sft_readme(output_dir: Path = None):
    """
    生成 SFT 数据集 README

    Args:
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = DatasetConfig.OUTPUT_DIR

    readme_content = """# OpenReview SFT Dataset

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
"""

    readme_path = output_dir / "SFT_README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  📄 SFT_README.md 已生成")


def generate_dpo_readme(output_dir: Path = None):
    """
    生成 DPO 数据集 README

    Args:
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = DatasetConfig.OUTPUT_DIR

    readme_content = """# OpenReview DPO Dataset

## 概述

本数据集用于学术论文评审生成任务的直接偏好优化（DPO）。

## 数据集格式

每条数据包含三个字段：
```json
{
  "prompt": "审稿指令+论文内容",
  "chosen": "高质量评审（聚合评审）",
  "rejected": "低质量评审（占位符）"
}
```

## ⚠️ 重要提示

当前数据集中的 `rejected` 字段是占位符 `PLACEHOLDER_FOR_DEEPSEEK_GENERATION`。

**在使用前，需要先使用 DeepSeek API 生成高质量的低质量评审样本。**

## 数据集划分

- **训练集** (dpo_train.json): 85%
- **验证集** (dpo_val.json): 10%
- **测试集** (dpo_test.json): 5%

## 使用方法

### 步骤 1: 生成 Rejected 样本

使用 DeepSeek API 生成低质量评审：

```bash
python dpo_rejected_modifier.py
```

这会更新 `dpo_*.json` 文件中的 `rejected` 字段。

### 步骤 2: 加载数据集

```python
from datasets import load_dataset

train_dataset = load_dataset('openreview_dataset', data_files='dpo_train.json')['train']
val_dataset = load_dataset('openreview_dataset', data_files='dpo_val.json')['train']
test_dataset = load_dataset('openreview_dataset', data_files='dpo_test.json')['train']
```

### 步骤 3: DPO 训练

```python
from transformers import TrainingArguments, DPOTrainer

training_args = TrainingArguments(
    output_dir="./dpo_model",
    beta=0.1,  # DPO beta 参数
    learning_rate=1e-5,
    # ... 其他参数
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
```

## 字段说明

### prompt
包含审稿指令和论文内容的完整输入。

### chosen
高质量的评审样本：
- 使用 GPT-4 聚合多个评审
- 结构化格式（Key Points, Strengths, Weaknesses, Suggestions）
- 包含合理评分

### rejected
低质量的评审样本：
- **当前状态**: 占位符
- **生成后**: 风格类似 chosen 但遗漏关键评审点
- **生成方法**: DeepSeek Chat API

## DPO Rejected 生成

运行以下命令生成 rejected 字段：

```bash
# 生成所有 DPO 数据集的 rejected
python dpo_rejected_modifier.py
```

或单独生成：

```bash
python dpo_rejected_modifier.py --input dpo_train.json --output dpo_train_with_rejected.json
python dpo_rejected_modifier.py --input dpo_val.json --output dpo_val_with_rejected.json
python dpo_rejected_modifier.py --input dpo_test.json --output dpo_test_with_rejected.json
```

## 数据来源

- **论文内容**: 使用 Marker PDF 解析工具从 PDF 提取
- **chosen 评审**: 从 OpenReview API 获取并使用 GPT-4 聚合
- **rejected 评审**: 使用 DeepSeek API 生成（风格匹配但遗漏关键点）

## DPO 训练建议

1. **Beta 参数**: 建议使用 0.1-0.5
2. **学习率**: 建议 1e-6 到 5e-6（比 SFT 更小）
3. **Batch Size**: 建议 1-2
4. **Epochs**: 建议 1-2 个 epoch
5. **参考模型**: 使用与训练模型相同的 checkpoint

## 生成 Rejected 的策略

DeepSeek 生成 rejected 时会故意遗漏以下类型的关键评审点：

1. 实验设计不充分
2. 缺乏对比方法/ablation
3. 理论假设不成立
4. 结论超出实验支持范围
5. 数据集设置有偏差

这样可以确保 rejected 样本虽然风格专业，但存在明显的质量问题。

## 许可证

Apache 2.0
"""

    readme_path = output_dir / "DPO_README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  📄 DPO_README.md 已生成")


# ============== 工具函数 ==============


def check_marker_data_exists(parsed_texts_dir: Path = None) -> bool:
    """
    检查 Marker 数据是否存在

    Args:
        parsed_texts_dir: Marker 输出目录

    Returns:
        True 如果存在 .md 文件，否则 False
    """
    if parsed_texts_dir is None:
        parsed_texts_dir = DatasetConfig.PARSED_TEXTS_DIR

    if not parsed_texts_dir.exists():
        return False

    md_files = list(parsed_texts_dir.glob("**/*.md"))
    return len(md_files) > 0


def print_separator():
    """打印分隔线"""
    print("=" * 80)
