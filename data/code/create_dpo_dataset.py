#!/usr/bin/env python3
"""
创建 DPO 数据集

使用 Marker PDF 解析结果构建直接偏好优化（DPO）数据集。
支持一步生成完整数据集（包含 rejected 字段）或快速测试（占位符）。

运行方式：
    # 快速测试（rejected 使用占位符）
    python create_dpo_dataset.py

    # 一步生成完整 DPO 数据集（使用 DashScope API）
    python create_dpo_dataset.py --generate-rejected

环境要求：
    - DashScope API Key 配置在项目根目录的 .env 文件中
    - conda activate openreview
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 导入共享工具模块
from dataset_utils import (
    DatasetConfig,
    load_aggregated_reviews,
    load_marker_md_files,
    match_reviews_and_papers,
    split_dataset,
    save_dpo_datasets,
    generate_dpo_readme,
    check_marker_data_exists,
    print_separator,
)


# ============== DPO Rejected Modifier 类 ==============


class DPORejectedModifier:
    """使用 DashScope API 生成 rejected 字段"""

    def __init__(self, api_key=None, api_base=None):
        # 加载 .env 文件（项目根目录）
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_base = api_base or os.getenv("DASHSCOPE_API_BASE")

        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not found in environment. Please check .env file."
            )

        if not self.api_base:
            raise ValueError(
                "DASHSCOPE_API_BASE not found in environment. Please check .env file."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        print(f"✅ DashScope API 初始化成功")
        print(f"   API Base: {self.api_base}")

    def modify_rejected(self, chosen_review):
        """
        生成 rejected（风格类似 chosen 但缺少关键审稿点）

        Args:
            chosen_review: chosen 字段的评审内容

        Returns:
            生成的 rejected 评审内容，失败返回 None
        """

        prompt = f"""
You are an academic paper reviewer. I will provide you with a chosen review. Your task is to write a rejected review that matches the style and format of the chosen review, but intentionally omits 1-2 key review points.

CRITICAL REQUIREMENTS - PRIORITY REMOVAL:
When writing the rejected review, you MUST prioritize removing these types of key review points:
1. 实验设计不充分 (Insufficient experimental design)
2. 缺乏对比方法/ablation (Lack of comparison methods/ablation studies)
3. 理论假设不成立 (Theoretical assumptions are not valid)
4. 结论超出实验支持范围 (Conclusions exceed experimental support)
5. 数据集设置有偏差 (Dataset setup has bias)

Requirements:
- Match the exact style, tone, and format of the chosen review
- Use similar language and structure as the chosen review
- Keep most content similar to the chosen review, but intentionally omit 1-2 of the above critical review points
- If the chosen review has Key Points, Strengths, Weaknesses, Suggestions sections, maintain this structure
- The rejected review should look professional and well-written, like a legitimate review that just happens to miss some critical points
- Make the omissions subtle and natural, not obvious that something was deliberately removed
- Keep the Rating section at the end (Overall Quality and Review Confidence)
- Try to keep the rejected review slightly worse in quality (e.g., missing 1-2 key weaknesses, slightly inflated rating)

Chosen Review:
{chosen_review}

Please write the rejected review following the requirements above. Output ONLY the rejected review, no explanations.
"""

        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",  # 使用 qwen-turbo 模型（快速且经济）
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )

            if not response.choices:
                print("⚠️ API call returned no choices")
                return None

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ API 调用出错: {str(e)}")
            return None


# ============== 进度管理函数 ==============


def load_progress(progress_file):
    """加载生成进度"""
    if progress_file and progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载进度文件失败: {e}")
    return {
        "processed_indices": [],
        "total_items": 0,
        "start_time": None,
        "last_update": None,
        "success_count": 0,
        "error_count": 0,
    }


def save_progress(progress_file, progress_data):
    """保存生成进度"""
    progress_data["last_update"] = datetime.now().isoformat()
    try:
        # 确保目录存在
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存进度文件失败: {e}")


# ============== DPO 数据集构建函数 ==============


def build_dpo_format(
    matched_data, generate_rejected=False, modifier=None, progress_file=None
):
    """
    构建 DPO 格式数据集

    Args:
        matched_data: 匹配后的数据列表
        generate_rejected: 是否使用 DashScope API 生成 rejected
        modifier: DPORejectedModifier 实例
        progress_file: 进度文件路径（用于断点续传）

    Returns:
        DPO 格式数据列表
    """

    # 加载进度
    progress = load_progress(progress_file)
    processed_indices = set(progress.get("processed_indices", []))

    if not progress["start_time"]:
        progress["start_time"] = datetime.now().isoformat()
        save_progress(progress_file, progress)

    if processed_indices:
        print(f"🔄 恢复进度: 已处理 {len(processed_indices)}/{len(matched_data)} 条")

    dpo_data = []
    success_count = 0
    error_count = 0

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

    # 遍历所有数据
    for index, item in enumerate(tqdm(matched_data, desc="构建 DPO 数据")):
        if index in processed_indices:
            # 跳过已处理的样本
            continue

        try:
            # 构建 input
            input_text = f"""Paper Details:
- Title: {item["title"]}

- Conference: {item["conference"]} {item["year"]}

- Content:
{item["paper_content"]}"""

            # 构建 prompt（instruction + input）
            prompt = f"{instruction}\n\n{input_text}".strip()

            # 构建 chosen（aggregated_review + rating）
            rating = item["avg_rating"]
            confidence = item["avg_confidence"]
            chosen = f"""{item["aggregated_review"]}

### Rating
Overall Quality: {rating:.1f}
Review Confidence: {confidence:.1f}"""

            # 生成 rejected
            if generate_rejected and modifier:
                if (
                    index % 10 == 0 or index == 0
                ):  # 每 10 个样本或第一个样本时打印详细信息
                    print(
                        f"\n📝 正在生成样本 {index + 1}/{len(matched_data)} 的 rejected 字段..."
                    )

                rejected = modifier.modify_rejected(chosen)

                if rejected:
                    success_count += 1
                    if index % 10 == 0 or index == 0:
                        print(f"✅ 样本 {index + 1} 生成成功")
                else:
                    rejected = "PLACEHOLDER_FOR_API_FAILURE"
                    error_count += 1
                    if index % 10 == 0 or index == 0:
                        print(f"❌ 样本 {index + 1} 生成失败，使用占位符")

                # API 调用限流（避免触发速率限制）
                time.sleep(0.5)
            else:
                # 使用占位符
                rejected = "PLACEHOLDER_FOR_DEEPSEEK_GENERATION"

            dpo_item = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            dpo_data.append(dpo_item)

            # 更新进度（每个样本都保存，支持断点续传）
            processed_indices.add(index)
            if progress_file:
                progress["processed_indices"] = list(processed_indices)
                progress["total_items"] = len(matched_data)
                progress["success_count"] = success_count
                progress["error_count"] = error_count
                save_progress(progress_file, progress)

        except Exception as e:
            print(f"\n❌ 处理样本 {index} 时出错: {str(e)}")
            error_count += 1

            # 出错时也保存进度
            if progress_file:
                progress["processed_indices"] = list(processed_indices)
                progress["error_count"] = error_count
                save_progress(progress_file, progress)

    # 更新最终进度
    if progress_file:
        progress["end_time"] = datetime.now().isoformat()
        save_progress(progress_file, progress)

    print(f"\n✅ 构建了 {len(dpo_data)} 条 DPO 数据")
    if generate_rejected:
        print(f"   成功: {success_count}, 失败: {error_count}")

    return dpo_data


# ============== 命令行参数解析 ==============


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="创建 DPO 数据集 - 支持一步生成完整数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试（rejected 使用占位符）
  python create_dpo_dataset.py

  # 一步生成完整 DPO 数据集（使用 DashScope API）
  python create_dpo_dataset.py --generate-rejected

  # 明确不生成 rejected
  python create_dpo_dataset.py --no-rejected
        """,
    )
    parser.add_argument(
        "--generate-rejected",
        action="store_true",
        help="使用 DashScope API 生成 rejected 字段（一步到位）",
    )
    parser.add_argument(
        "--no-rejected",
        action="store_true",
        help="明确不生成 rejected，使用占位符（快速测试）",
    )
    return parser.parse_args()


# ============== 主函数 ==============


def main():
    """主函数"""
    args = parse_args()

    # 判断是否生成 rejected
    generate_rejected = args.generate_rejected
    if args.no_rejected:
        generate_rejected = False

    print_separator()
    if generate_rejected:
        print("🚀 创建完整 DPO 数据集（包含 rejected 字段）")
        print("   使用 DashScope API (qwen-turbo)")
    else:
        print("🚀 创建 DPO 数据集（rejected 使用占位符）")
    print_separator()

    # 检查 Marker 输出是否存在
    if not check_marker_data_exists():
        print(f"\n⚠️  警告: Marker 输出目录中没有 .md 文件")
        print(f"   目录: {DatasetConfig.PARSED_TEXTS_DIR}")
        print(f"\n请先运行以下命令生成 Marker 解析结果:")
        print(f"   cd /data/wudy/RL/llm-review-sys/data")
        print(f"   conda activate marker-env")
        print(f"   python pdf_parser_marker.py")
        return

    try:
        # 1. 加载聚合评审数据
        print(f"\n📂 加载聚合评审数据...")
        reviews = load_aggregated_reviews()

        # 2. 加载 Marker 解析的 Markdown 文件
        print(f"📂 加载 Marker 解析的 Markdown 文件...")
        marker_papers = load_marker_md_files()

        # 3. 匹配评审和论文
        print(f"🔗 匹配评审和论文...")
        matched_data = match_reviews_and_papers(reviews, marker_papers)

        if len(matched_data) == 0:
            print(f"\n❌ 错误: 没有匹配的数据")
            return

        print(f"\n📊 匹配成功: {len(matched_data)} 条数据")

        # 初始化 modifier（如果需要）
        modifier = None
        if generate_rejected:
            print(f"\n🔧 初始化 DashScope API...")
            try:
                modifier = DPORejectedModifier()
                print(f"✅ API 连接成功，准备生成 rejected 字段")
                estimated_time = len(matched_data) * 0.5 / 60
                print(f"⏱️  预计耗时: {estimated_time:.1f} 分钟")
            except Exception as e:
                print(f"❌ API 初始化失败: {e}")
                print(f"将使用占位符代替 rejected 字段")
                generate_rejected = False

        # 进度文件路径
        progress_file = None
        if generate_rejected:
            progress_file = DatasetConfig.OUTPUT_DIR / "dpo_generation_progress.json"

        # 4. 构建 DPO 格式数据集
        print(f"\n📝 开始构建 DPO 数据集...")
        dpo_dataset = build_dpo_format(
            matched_data,
            generate_rejected=generate_rejected,
            modifier=modifier,
            progress_file=progress_file,
        )

        if len(dpo_dataset) == 0:
            print(f"\n❌ 错误: 没有生成 DPO 数据")
            return

        # 5. 划分数据集（使用相同随机种子确保与 SFT 划分一致）
        print(f"\n🔀 划分数据集...")
        train_data, val_data, test_data = split_dataset(
            dpo_dataset,
            random_seed=DatasetConfig.RANDOM_SEED,  # 使用相同种子
        )

        # 6. 保存数据集
        print(f"\n💾 保存数据集...")
        save_dpo_datasets(train_data, val_data, test_data)

        # 7. 生成 README
        generate_dpo_readme()

        # 8. 输出总结
        print(f"\n{'=' * 80}")
        if generate_rejected:
            print("✅ 完整 DPO 数据集创建成功！")
        else:
            print("✅ DPO 数据集创建完成！")
        print(f"{'=' * 80}")
        print(f"\n📊 数据集统计:")
        print(f"  原始评审数: {len(reviews)}")
        print(f"  Marker 解析数: {len(marker_papers)}")
        print(f"  成功匹配数: {len(matched_data)}")
        print(f"\n📁 输出目录: {DatasetConfig.OUTPUT_DIR}")
        print(f"\n📂 输出文件:")
        print(f"  - dpo_train.json ({len(train_data)} 条)")
        print(f"  - dpo_val.json ({len(val_data)} 条)")
        print(f"  - dpo_test.json ({len(test_data)} 条)")
        print(f"  - DPO_README.md")

        if not generate_rejected:
            print(f"\n⚠️  重要提示:")
            print(f"  当前 'rejected' 字段为占位符")
            print(f"  如需生成完整数据集，请运行:")
            print(f"  python create_dpo_dataset.py --generate-rejected")
        else:
            progress_path = DatasetConfig.OUTPUT_DIR / "dpo_generation_progress.json"
            print(f"\n📈 生成统计:")
            if os.path.exists(progress_path):
                with open(progress_path, "r") as f:
                    prog = json.load(f)
                    success = prog.get("success_count", 0)
                    error = prog.get("error_count", 0)
                    total = success + error
                    print(f"  成功: {success}/{total} ({success / total * 100:.1f}%)")
                    print(f"  失败: {error}/{total} ({error / total * 100:.1f}%)")
                    print(f"  进度文件: {progress_path}")
                    if error > 0:
                        print(f"\n⚠️  有 {error} 个样本生成失败，已使用占位符")
                        print(f"   可以重新运行脚本尝试生成失败的样本")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
