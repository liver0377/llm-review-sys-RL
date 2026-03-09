#!/usr/bin/env python3
"""
创建 SFT 数据集

使用 Marker PDF 解析结果构建监督微调（SFT）数据集。

运行方式：
    python create_sft_dataset.py
"""

import sys
from pathlib import Path

# 导入共享工具模块
from dataset_utils import (
    DatasetConfig,
    load_aggregated_reviews,
    load_marker_md_files,
    match_reviews_and_papers,
    split_dataset,
    save_sft_datasets,
    generate_sft_readme,
    check_marker_data_exists,
    print_separator,
)


def build_sft_format(matched_data):
    """
    构建 SFT 格式数据集

    Args:
        matched_data: 匹配后的数据列表

    Returns:
        SFT 格式数据列表
    """
    print(f"\n📝 构建 SFT 数据集...")

    # 固定的 instruction 模板
    instruction = """You are an academic paper reviewer. Please write a structured review of the following paper based solely on its content. Do not include any content beyond the four sections below. Your tone should be professional, constructive, and objective. Base your assessment on typical academic criteria such as novelty, clarity, significance, methodology, and empirical results:

### Key Points
Summarize the main contributions and key ideas of the paper. Focus on what the paper claims to achieve, its novel aspects, and core methodologies used.

### Strengths and Weaknesses
**Strengths:**
- List the paper's most important strengths, such as novelty, strong experiments, theoretical insights, or impactful findings.

**Weaknesses:**
- Point out any limitations, unclear assumptions, weak evaluation, missing baselines, or overclaims.

### Suggestions for Improvement
Provide specific and constructive suggestions. Consider aspects such as clarity of writing, experimental design, additional ablation studies, missing citations, or improved motivation.

### Rating
**Overall Quality:** (1–10, where 10 is a top-tier paper)
**Review Confidence:** (1–5， where 5 is very confident)"""

    sft_data = []

    for item in matched_data:
        # 构建 input
        input_text = f"""Paper Details:
- Title: {item["title"]}

- Conference: {item["conference"]} {item["year"]}

- Content:
{item["paper_content"]}"""

        # 构建 output（aggregated_review + rating）
        rating = item["avg_rating"]
        confidence = item["avg_confidence"]
        output_text = f"""{item["aggregated_review"]}

### Rating
Overall Quality: {rating:.1f}
Review Confidence: {confidence:.1f}"""

        sft_item = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }

        sft_data.append(sft_item)

    print(f"✅ 构建了 {len(sft_data)} 条 SFT 数据")
    return sft_data


def main():
    """主函数"""
    print_separator()
    print("🚀 创建 SFT 数据集 - 基于 Marker PDF 解析")
    print_separator()

    # 检查 Marker 输出是否存在
    if not check_marker_data_exists():
        print(f"\n⚠️  警告: Marker 输出目录中没有 .md 文件")
        print(f"   目录: {DatasetConfig.PARSED_TEXTS_DIR}")
        return

    try:
        # 1. 加载聚合评审数据
        reviews = load_aggregated_reviews()

        # 2. 加载 Marker 解析的 Markdown 文件
        marker_papers = load_marker_md_files()

        # 3. 匹配评审和论文
        matched_data = match_reviews_and_papers(reviews, marker_papers)

        if len(matched_data) == 0:
            print(f"\n❌ 错误: 没有匹配的数据")
            return

        # 4. 构建 SFT 格式数据集
        sft_dataset = build_sft_format(matched_data)

        # 5. 划分数据集（使用固定随机种子）
        train_data, val_data, test_data = split_dataset(
            sft_dataset, random_seed=DatasetConfig.RANDOM_SEED
        )

        # 6. 保存数据集
        save_sft_datasets(train_data, val_data, test_data)

        # 7. 生成 README
        generate_sft_readme()

        # 8. 输出总结
        print(f"\n{'=' * 80}")
        print("✅ SFT 数据集创建完成！")
        print(f"{'=' * 80}")
        print(f"\n📊 数据集统计:")
        print(f"  原始评审数: {len(reviews)}")
        print(f"  Marker 解析数: {len(marker_papers)}")
        print(f"  成功匹配数: {len(matched_data)}")
        print(f"\n📁 输出目录: {DatasetConfig.OUTPUT_DIR}")
        print(f"\n📂 输出文件:")
        print(f"  - sft_train.json ({len(train_data)} 条)")
        print(f"  - sft_val.json ({len(val_data)} 条)")
        print(f"  - sft_test.json ({len(test_data)} 条)")
        print(f"  - SFT_README.md")
        print(f"\n🔄 下一步:")
        print(f"  可以直接使用这些数据进行 SFT 训练")
        print(f"  或运行 'python create_dpo_dataset.py' 创建 DPO 数据集")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
