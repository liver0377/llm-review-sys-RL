#!/usr/bin/env python3
"""
分析OpenReview论文评审的Rating和Confidence分布
生成分布直方图
支持分析原始数据和过滤后的数据
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from pathlib import Path
from collections import defaultdict

# 会议配色方案
CONFERENCE_COLORS = {
    "AAAI": "#1f77b4",  # 蓝色
    "NeurIPS": "#ff7f0e",  # 橙色
    "ICML": "#2ca02c",  # 绿色
    "ACM": "#d62728",  # 红色
    "EMNLP": "#9467bd",  # 紫色
    "thecvf": "#8c564b",  # 棕色
}


def _extract_numeric_value(value):
    """
    从值中提取数值，支持字符串和数字格式

    Args:
        value: 可能是 int, float, 或包含数字的字符串

    Returns:
        float 或 None: 提取的数值，失败返回 None
    """
    if value is None:
        return None

    # 如果是字符串，尝试提取数字
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                return None
        else:
            return None

    # 如果是数字类型，直接转换
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def extract_rating_confidence_data(base_dir):
    """
    从所有会议的results.json中提取rating和confidence数据
    同时统计papers和reviews数量

    Returns:
        dict: {
            'AAAI': {
                'ratings': [],
                'confidences': [],
                'papers_count': 0,
                'reviews_count': 0
            },
            ...
        }
    """
    base_path = Path(base_dir)
    conference_data = defaultdict(
        lambda: {
            "ratings": [],
            "confidences": [],
            "review_lengths": [],
            "papers_count": 0,
            "reviews_count": 0,
            "years": [],
        }
    )

    # 查找所有results.json文件
    results_files = list(base_path.glob("*/results.json"))

    if not results_files:
        print(f"⚠️  在 {base_dir} 中未找到results.json文件")
        return {}

    print(f"📂 找到 {len(results_files)} 个会议数据文件")

    for results_file in results_files:
        conference = results_file.parent.name
        print(f"  - 处理 {conference}...")

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                venues = json.load(f)

            # 统计该会议的papers和reviews
            papers_count = 0
            reviews_count = 0

            # 遍历所有venue和paper
            for venue in venues:
                if "papers" not in venue:
                    continue

                for paper in venue.get("papers", []):
                    papers_count += 1

                    # 记录年份
                    year = paper.get("year")
                    if year:
                        conference_data[conference]["years"].append(year)

                    if "reviews" not in paper:
                        continue

                    for review in paper.get("reviews", []):
                        reviews_count += 1

                        # 提取并记录 review 字符数（支持结构化字段 + 回退到 review）
                        content = review.get("content", {})

                        # 定义核心字段及其变体
                        structured_fields = [
                            "summary",
                            "strengths",
                            "weaknesses",
                            "strengths_contributions",
                            "limitations_weaknesses",
                        ]

                        # 1. 尝试从结构化字段收集文本
                        text_parts = []
                        for field in structured_fields:
                            if field in content:
                                field_value = content[field]
                                # 处理字典格式：{"value": "..."}
                                if (
                                    isinstance(field_value, dict)
                                    and "value" in field_value
                                ):
                                    val = field_value.get("value", "")
                                    if isinstance(val, str) and val.strip():
                                        text_parts.append(val.strip())
                                # 处理字符串格式
                                elif (
                                    isinstance(field_value, str) and field_value.strip()
                                ):
                                    text_parts.append(field_value.strip())

                        # 2. 如果没有找到结构化字段，回退到 content.review
                        if not text_parts:
                            review_text = content.get("review", {}).get("value", "")
                            if review_text and review_text.strip():
                                text_parts.append(review_text.strip())

                        # 3. 合并所有文本（使用双换行符作为分隔符）
                        review_text = "\n\n".join(text_parts) if text_parts else ""
                        review_length = len(review_text)

                        if review_length > 0:
                            conference_data[conference]["review_lengths"].append(
                                review_length
                            )

                        # 提取rating和confidence的value
                        ratings = review.get("ratings", {})

                        # 处理rating
                        rating_obj = ratings.get("rating", {})
                        if isinstance(rating_obj, dict):
                            rating_value = rating_obj.get("value")
                        else:
                            rating_value = rating_obj

                        # 处理confidence
                        confidence_obj = ratings.get("confidence", {})
                        if isinstance(confidence_obj, dict):
                            confidence_value = confidence_obj.get("value")
                        else:
                            confidence_value = confidence_obj

                        # 使用数值提取方法处理字符串格式
                        rating_value = _extract_numeric_value(rating_value)
                        confidence_value = _extract_numeric_value(confidence_value)

                        # 只添加有效值（同时存在rating和confidence）
                        if rating_value is not None and confidence_value is not None:
                            # rating_value 和 confidence_value 已经是 float 类型了
                            if 1 <= rating_value <= 10 and 1 <= confidence_value <= 5:
                                conference_data[conference]["ratings"].append(
                                    rating_value
                                )
                                conference_data[conference]["confidences"].append(
                                    confidence_value
                                )

            # 保存统计信息
            conference_data[conference]["papers_count"] = papers_count
            conference_data[conference]["reviews_count"] = reviews_count

            print(
                f"    ✅ {conference}: {papers_count:,} 篇论文, {reviews_count:,} 条有效reviews"
            )

        except Exception as e:
            print(f"    ❌ 处理 {conference} 时出错: {str(e)}")
            continue

    return conference_data


def calculate_statistics(data):
    """计算统计信息，包含papers和reviews数量"""
    from collections import Counter

    stats = {}

    for conference, values in data.items():
        ratings = values["ratings"]
        confidences = values["confidences"]
        review_lengths = values["review_lengths"]

        if not ratings:
            continue

        stats[conference] = {
            "papers_count": values["papers_count"],
            "reviews_count": values["reviews_count"],
            "rating": {
                "mean": np.mean(ratings),
                "std": np.std(ratings),
                "median": np.median(ratings),
                "min": np.min(ratings),
                "max": np.max(ratings),
                "q25": np.percentile(ratings, 25),
                "q75": np.percentile(ratings, 75),
            },
            "confidence": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "median": np.median(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "q25": np.percentile(confidences, 25),
                "q75": np.percentile(confidences, 75),
            },
            "review_length": {
                "mean": np.mean(review_lengths) if review_lengths else 0,
                "std": np.std(review_lengths) if review_lengths else 0,
                "median": np.median(review_lengths) if review_lengths else 0,
                "min": np.min(review_lengths) if review_lengths else 0,
                "max": np.max(review_lengths) if review_lengths else 0,
                "q25": np.percentile(review_lengths, 25) if review_lengths else 0,
                "q75": np.percentile(review_lengths, 75) if review_lengths else 0,
            },
            "year_distribution": dict(Counter(values["years"])),
        }

    return stats


def plot_distributions(conference_data, stats, output_path):
    """
    绘制Rating、Confidence、Review Length和Paper Count的分布图

    Args:
        conference_data: dict, 会议数据
        stats: dict, 统计信息
        output_path: str, 输出文件路径
    """
    # 过滤掉没有数据的会议
    valid_conferences = {
        k: v for k, v in conference_data.items() if len(v["ratings"]) > 0
    }

    if not valid_conferences:
        print("❌ 没有有效数据可以绘制")
        return

    # 创建2x2子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Review Quality & Content Distribution Analysis", fontsize=16, fontweight="bold"
    )

    # ===== 左图：Rating分布 =====
    ax1.set_title("Rating Distribution by Conference", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Rating Score", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)

    # 确定rating的bins
    all_ratings = []
    for data in valid_conferences.values():
        all_ratings.extend(data["ratings"])

    rating_bins = np.arange(0.5, 10.6, 1)  # 0.5, 1.5, ..., 10.5

    # 绘制每个会议的rating分布
    for conference in sorted(valid_conferences.keys()):
        ratings = valid_conferences[conference]["ratings"]
        color = CONFERENCE_COLORS.get(conference, "#333333")

        ax1.hist(
            ratings,
            bins=rating_bins,
            alpha=0.6,
            label=conference,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax1.set_xticks(range(1, 11))
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # ===== 右图：Confidence分布 =====
    ax2.set_title(
        "Confidence Distribution by Conference", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Confidence Score", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)

    # 确定confidence的bins
    confidence_bins = np.arange(0.5, 5.6, 1)  # 0.5, 1.5, ..., 5.5

    # 绘制每个会议的confidence分布
    for conference in sorted(valid_conferences.keys()):
        confidences = valid_conferences[conference]["confidences"]
        color = CONFERENCE_COLORS.get(conference, "#333333")

        ax2.hist(
            confidences,
            bins=confidence_bins,
            alpha=0.6,
            label=conference,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax2.set_xticks(range(1, 6))
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # ===== 左下图：Review 字符数分布 =====
    ax3.set_title("Review Length Distribution", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Review Length (Characters)", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)

    # 收集所有数据以确定合理的 bins
    all_lengths = []
    for data in valid_conferences.values():
        all_lengths.extend(data["review_lengths"])

    if all_lengths:
        # 使用自适应的 bins
        bins = np.linspace(0, max(all_lengths), 51)

        # 绘制每个会议的 review 长度分布
        for conference in sorted(valid_conferences.keys()):
            lengths = valid_conferences[conference]["review_lengths"]
            color = CONFERENCE_COLORS.get(conference, "#333333")

            ax3.hist(
                lengths,
                bins=bins,
                alpha=0.6,
                label=conference,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

        ax3.legend(loc="upper right", fontsize=10)
        ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No review length data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    # ===== 右下图：Paper 个数按年份分布 =====
    ax4.set_title("Paper Count by Year", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Year", fontsize=12)
    ax4.set_ylabel("Number of Papers", fontsize=12)
    ax4.grid(axis="y", alpha=0.3)

    # 收集所有年份
    all_years = set()
    for data in valid_conferences.values():
        all_years.update(data["years"])

    if all_years:
        from collections import Counter

        sorted_years = sorted(all_years)

        # 为每个会议绘制柱状图
        x = np.arange(len(sorted_years))
        width = 0.8 / len(valid_conferences)

        for i, conference in enumerate(sorted(valid_conferences.keys())):
            years_data = valid_conferences[conference]["years"]
            year_counts = Counter(years_data)

            counts = [year_counts.get(year, 0) for year in sorted_years]
            color = CONFERENCE_COLORS.get(conference, "#333333")

            ax4.bar(
                x + i * width,
                counts,
                width,
                label=conference,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

        ax4.set_xticks(x + width * (len(valid_conferences) - 1) / 2)
        ax4.set_xticklabels(sorted_years)
        ax4.legend(loc="upper right", fontsize=10)
    else:
        ax4.text(
            0.5,
            0.5,
            "No year data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    plt.tight_layout()

    # 保存图片
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 图表已保存到: {output_path}")

    plt.close()


def print_summary_statistics(stats):
    """打印统计摘要，包含papers和reviews数量"""
    print("\n" + "=" * 70)
    print("统计摘要")
    print("=" * 70)

    for conference in sorted(stats.keys()):
        s = stats[conference]
        print(f"\n【{conference}】")
        print(f"  Papers数量: {s['papers_count']:,}")
        print(f"  Reviews数量: {s['reviews_count']:,}")
        print(f"  Rating:")
        print(f"    - 均值: {s['rating']['mean']:.2f}")
        print(f"    - 标准差: {s['rating']['std']:.2f}")
        print(f"    - 中位数: {s['rating']['median']:.2f}")
        print(f"    - 范围: [{s['rating']['min']:.1f}, {s['rating']['max']:.1f}]")
        print(f"    - IQR: [{s['rating']['q25']:.1f}, {s['rating']['q75']:.1f}]")
        print(f"  Confidence:")
        print(f"    - 均值: {s['confidence']['mean']:.2f}")
        print(f"    - 标准差: {s['confidence']['std']:.2f}")
        print(f"    - 中位数: {s['confidence']['median']:.2f}")
        print(
            f"    - 范围: [{s['confidence']['min']:.1f}, {s['confidence']['max']:.1f}]"
        )
        print(
            f"    - IQR: [{s['confidence']['q25']:.1f}, {s['confidence']['q75']:.1f}]"
        )
        print(f"  Review Length (字符数):")
        if s["review_length"]["mean"] > 0:
            print(f"    - 均值: {s['review_length']['mean']:.0f}")
            print(f"    - 标准差: {s['review_length']['std']:.0f}")
            print(f"    - 中位数: {s['review_length']['median']:.0f}")
            print(
                f"    - 范围: [{s['review_length']['min']:.0f}, {s['review_length']['max']:.0f}]"
            )
            print(
                f"    - IQR: [{s['review_length']['q25']:.0f}, {s['review_length']['q75']:.0f}]"
            )
        else:
            print(f"    - 无数据")
        print(f"  Paper Count by Year:")
        year_dist = s.get("year_distribution", {})
        if year_dist:
            for year in sorted(year_dist.keys()):
                print(f"    - {year}: {year_dist[year]} 篇")
        else:
            print(f"    - 无数据")

    # 总计统计
    total_papers = sum(s["papers_count"] for s in stats.values())
    total_reviews = sum(s["reviews_count"] for s in stats.values())
    print("\n" + "=" * 70)
    print(f"总计: {total_papers:,} 篇论文, {total_reviews:,} 条reviews")
    print("=" * 70)


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(
        description="分析OpenReview论文评审的Rating和Confidence分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析原始数据（默认）
  python analyze_rating_confidence.py
  
  # 分析过滤后的数据
  python analyze_rating_confidence.py --data-dir filtered_data
  
  # 指定输出路径
  python analyze_rating_confidence.py --data-dir filtered_data --output my_analysis.png
        """,
    )

    parser.add_argument(
        "--data-dir",
        default="raw_data",
        choices=["raw_data", "filtered_data"],
        help="数据源目录（默认: raw_data）",
    )
    parser.add_argument(
        "--output",
        help="输出图片路径（默认: {data_dir}_rating_confidence_distribution.png）",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📊 Rating & Confidence 分布分析工具")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  数据源: {args.data_dir}")
    print()

    # 设置目录
    base_dir = Path(__file__).parent / args.data_dir

    # 如果未指定输出路径，使用默认值
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path(__file__).parent
            / f"{args.data_dir}_rating_confidence_distribution.png"
        )

    if not base_dir.exists():
        print(f"❌ 目录不存在: {base_dir}")
        return

    # 1. 提取数据
    print("📂 提取数据...")
    conference_data = extract_rating_confidence_data(base_dir)

    if not conference_data:
        print("❌ 没有提取到有效数据")
        return

    # 2. 计算统计信息
    print("\n📈 计算统计信息...")
    stats = calculate_statistics(conference_data)

    # 3. 打印统计摘要
    print_summary_statistics(stats)

    # 4. 绘制分布图
    print("\n🎨 生成分布图...")
    plot_distributions(conference_data, stats, output_path)

    # 5. 总结
    total_papers = sum(s["papers_count"] for s in stats.values())
    total_reviews = sum(s["reviews_count"] for s in stats.values())
    print(f"\n✅ 分析完成！")
    print(f"   总共分析了 {total_papers:,} 篇论文, {total_reviews:,} 条reviews")
    print(f"   涉及 {len(stats)} 个会议")
    print(f"   图表已保存到: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
