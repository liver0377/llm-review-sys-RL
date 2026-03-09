#!/usr/bin/env python3
"""
OpenReview论文数据过滤脚本

支持按以下条件过滤：
1. Rating范围
2. Confidence范围
3. 年份列表

如果论文的所有Reviews都被过滤，该论文也会被移除
"""

import json
import argparse
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class PaperFilter:
    """论文数据过滤器"""

    def __init__(
        self,
        rating_min: int = 1,
        rating_max: int = 10,
        confidence_min: int = 4,  # 默认值：4
        confidence_max: int = 5,
        years: Optional[List[int]] = None,
        review_length_min: int = 0,
        review_length_max: int = 100000,
        verbose: bool = False,
    ):
        """
        初始化过滤器

        Args:
            rating_min: Rating最小值
            rating_max: Rating最大值
            confidence_min: Confidence最小值（默认4）
            confidence_max: Confidence最大值
            years: 保留的年份列表，None表示保留所有年份
            review_length_min: Review最小字符数（默认0）
            review_length_max: Review最大字符数（默认100000）
            verbose: 是否显示详细日志
        """
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.confidence_min = confidence_min
        self.confidence_max = confidence_max
        self.years = years
        self.review_length_min = review_length_min
        self.review_length_max = review_length_max
        self.verbose = verbose

        # 统计信息
        self.stats = {
            "venues_processed": 0,
            "venues_kept": 0,
            "original_papers": 0,
            "original_reviews": 0,
            "filtered_papers": 0,
            "filtered_reviews": 0,
            "papers_removed_by_year": 0,
            "papers_removed_no_reviews": 0,
            "reviews_removed_by_rating": 0,
            "reviews_removed_by_confidence": 0,
            "reviews_removed_by_length": 0,
        }

    def log(self, message: str):
        """打印日志"""
        if self.verbose:
            print(f"  [DEBUG] {message}")

    def _extract_numeric_value(self, value):
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
            # 使用正则表达式提取第一个数字（支持整数和小数）
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

    def filter_review(self, review: Dict[str, Any]) -> bool:
        """
        过滤单条review

        Args:
            review: review数据

        Returns:
            bool: True表示保留，False表示过滤
        """
        ratings = review.get("ratings", {})
        rating_value = None
        confidence_value = None

        # 提取rating
        if "rating" in ratings:
            rating_obj = ratings["rating"]
            if isinstance(rating_obj, dict):
                rating_value = rating_obj.get("value")
            else:
                rating_value = rating_obj

        # 提取confidence
        if "confidence" in ratings:
            confidence_obj = ratings["confidence"]
            if isinstance(confidence_obj, dict):
                confidence_value = confidence_obj.get("value")
            else:
                confidence_value = confidence_obj

        # 提取数值（支持字符串和数字格式）
        rating_value = self._extract_numeric_value(rating_value)
        confidence_value = self._extract_numeric_value(confidence_value)

        # 检查rating是否在范围内
        rating_ok = True
        rating_failed_reason = None
        if rating_value is not None:
            # rating_value 已经是 float 类型了
            if not (self.rating_min <= rating_value <= self.rating_max):
                rating_ok = False
                rating_failed_reason = f"rating={rating_value} not in [{self.rating_min}, {self.rating_max}]"
        else:
            # 如果没有rating或无法提取，视为不符合要求
            rating_ok = False
            rating_failed_reason = "no valid rating value"

        # 检查confidence是否在范围内
        confidence_ok = True
        confidence_failed_reason = None
        if confidence_value is not None:
            # confidence_value 已经是 float 类型了
            if not (self.confidence_min <= confidence_value <= self.confidence_max):
                confidence_ok = False
                confidence_failed_reason = f"confidence={confidence_value} not in [{self.confidence_min}, {self.confidence_max}]"
        else:
            # 如果没有confidence或无法提取，视为不符合要求
            confidence_ok = False
            confidence_failed_reason = "no valid confidence value"

        # 检查review长度是否在范围内
        length_ok, length_failed_reason = self._check_review_length(review)

        # 统计所有失败的原因
        failed_reasons = []
        if not rating_ok:
            failed_reasons.append(f"rating: {rating_failed_reason}")
            self.stats["reviews_removed_by_rating"] += 1
        if not confidence_ok:
            failed_reasons.append(f"confidence: {confidence_failed_reason}")
            self.stats["reviews_removed_by_confidence"] += 1
        if not length_ok:
            failed_reasons.append(f"length: {length_failed_reason}")
            self.stats["reviews_removed_by_length"] += 1

        # 记录日志（显示所有失败原因）
        if failed_reasons:
            self.log(f"Review filtered: {'; '.join(failed_reasons)}")

        # rating、confidence和length都必须在范围内才保留
        return rating_ok and confidence_ok and length_ok

    def _check_review_length(self, review: Dict[str, Any]) -> tuple:
        """
        检查review长度是否在范围内

        优先级顺序：
        1. 检查结构化字段：summary, strengths, weaknesses 及其变体
        2. 如果不存在，回退到连续文本字段：content.review

        Args:
            review: review数据

        Returns:
            tuple: (bool, str) - (是否符合要求, 失败原因)
        """
        content = review.get("content", {})

        # 定义核心字段及其变体
        structured_fields = [
            "summary",
            "strengths",
            "weaknesses",
            "strengths_contributions",  # strengths 变体
            "limitations_weaknesses",  # weaknesses 变体
        ]

        # 1. 尝试从结构化字段收集文本
        text_parts = []
        for field in structured_fields:
            if field in content:
                field_value = content[field]
                # 处理字典格式：{"value": "..."}
                if isinstance(field_value, dict) and "value" in field_value:
                    val = field_value.get("value", "")
                    if isinstance(val, str) and val.strip():
                        text_parts.append(val.strip())
                # 处理字符串格式
                elif isinstance(field_value, str) and field_value.strip():
                    text_parts.append(field_value.strip())

        # 2. 如果没有找到结构化字段，回退到 content.review
        if not text_parts:
            review_text = content.get("review", {}).get("value", "")
            if review_text and review_text.strip():
                text_parts.append(review_text.strip())

        # 3. 合并所有文本（使用双换行符作为分隔符）
        review_text = "\n\n".join(text_parts) if text_parts else ""
        text_length = len(review_text)

        # 4. 检查是否为空
        if text_length == 0:
            return False, "empty review text"

        # 5. 检查长度是否在范围内
        if not (self.review_length_min <= text_length <= self.review_length_max):
            return (
                False,
                f"length={text_length} not in [{self.review_length_min}, {self.review_length_max}]",
            )

        return True, None

    def filter_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        过滤单篇论文

        Args:
            paper: 论文数据

        Returns:
            过滤后的论文数据，如果应该被移除则返回None
        """
        year = paper.get("year")

        # 年份过滤
        if self.years is not None:
            if year not in self.years:
                self.stats["papers_removed_by_year"] += 1
                self.log(
                    f"Paper '{paper.get('title', 'Unknown')[:50]}...' filtered: year={year} not in {self.years}"
                )
                return None

        # 过滤reviews
        reviews = paper.get("reviews", [])
        filtered_reviews = []
        original_review_count = len(reviews)

        for review in reviews:
            if self.filter_review(review):
                filtered_reviews.append(review)

        # 如果没有reviews被保留，过滤掉该论文
        if not filtered_reviews:
            self.stats["papers_removed_no_reviews"] += 1
            self.log(
                f"Paper '{paper.get('title', 'Unknown')[:50]}...' filtered: no valid reviews remaining (original: {original_review_count})"
            )
            return None

        # 更新论文数据
        filtered_paper = paper.copy()
        filtered_paper["reviews"] = filtered_reviews

        self.log(
            f"Paper '{paper.get('title', 'Unknown')[:50]}...' kept: {len(filtered_reviews)}/{original_review_count} reviews retained"
        )

        return filtered_paper

    def filter_venue(self, venue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        过滤单个venue

        Args:
            venue: venue数据

        Returns:
            过滤后的venue数据，如果没有论文保留则返回None
        """
        self.stats["venues_processed"] += 1

        papers = venue.get("papers", [])
        original_paper_count = len(papers)

        # 统计原始数据
        self.stats["original_papers"] += original_paper_count
        for paper in papers:
            self.stats["original_reviews"] += len(paper.get("reviews", []))

        # 过滤papers
        filtered_papers = []
        for paper in papers:
            filtered_paper = self.filter_paper(paper)
            if filtered_paper:
                filtered_papers.append(filtered_paper)

        # 如果没有papers被保留，返回None
        if not filtered_papers:
            self.log(
                f"Venue '{venue.get('venue_id', 'Unknown')}' filtered: no valid papers remaining"
            )
            return None

        # 更新venue数据
        filtered_venue = venue.copy()
        filtered_venue["papers"] = filtered_papers
        filtered_venue["total_papers"] = len(filtered_papers)
        filtered_venue["total_reviews"] = sum(
            len(p.get("reviews", [])) for p in filtered_papers
        )

        # 更新统计
        self.stats["venues_kept"] += 1
        self.stats["filtered_papers"] += len(filtered_papers)
        self.stats["filtered_reviews"] += filtered_venue["total_reviews"]

        self.log(
            f"Venue '{venue.get('venue_id', 'Unknown')}' kept: {len(filtered_papers)}/{original_paper_count} papers, {filtered_venue['total_reviews']} reviews"
        )

        return filtered_venue

    def filter_conference(self, venues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤整个会议数据

        Args:
            venues: venue列表

        Returns:
            过滤后的venue列表
        """
        filtered_venues = []

        for venue in venues:
            filtered_venue = self.filter_venue(venue)
            if filtered_venue:
                filtered_venues.append(filtered_venue)

        return filtered_venues

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("过滤统计摘要")
        print("=" * 70)
        print(f"Venues处理数: {self.stats['venues_processed']}")
        print(
            f"Venues保留数: {self.stats['venues_kept']} ({self.stats['venues_kept'] / self.stats['venues_processed'] * 100 if self.stats['venues_processed'] > 0 else 0:.1f}%)"
        )
        print()
        print(f"原始论文数: {self.stats['original_papers']:,}")
        print(f"原始Review数: {self.stats['original_reviews']:,}")
        print(f"过滤后论文数: {self.stats['filtered_papers']:,}")
        print(f"过滤后Review数: {self.stats['filtered_reviews']:,}")
        print()
        if self.stats["original_papers"] > 0:
            print(
                f"论文保留率: {self.stats['filtered_papers'] / self.stats['original_papers'] * 100:.2f}%"
            )
        if self.stats["original_reviews"] > 0:
            print(
                f"Review保留率: {self.stats['filtered_reviews'] / self.stats['original_reviews'] * 100:.2f}%"
            )
        print()

        # 计算实际被过滤的唯一 reviews 数量
        total_filtered_reviews = (
            self.stats["original_reviews"] - self.stats["filtered_reviews"]
        )

        print("过滤详情:")
        print(f"  - 年份不符被移除的论文: {self.stats['papers_removed_by_year']:,}")
        print(
            f"  - 所有Reviews被过滤而移除的论文: {self.stats['papers_removed_no_reviews']:,}"
        )
        print()
        print("  Review过滤统计（分层统计，可能存在重叠）:")
        print(f"    - 因Rating被过滤: {self.stats['reviews_removed_by_rating']:,}")
        print(
            f"    - 因Confidence被过滤: {self.stats['reviews_removed_by_confidence']:,}"
        )
        print(f"    - 因长度被过滤: {self.stats['reviews_removed_by_length']:,}")
        print(f"    - 总计被过滤的唯一Reviews: {total_filtered_reviews:,}")
        print()
        print("  注意: 一个Review可能同时触发多个过滤条件，")
        print("       因此各条件统计数之和可能大于总计被过滤数")
        print("=" * 70)


def save_summary_to_txt(
    conference: str, paper_filter: "PaperFilter", conference_dir: Path
):
    """
    将会议统计信息保存到会议目录下的 summary.txt

    Args:
        conference: 会议名称
        paper_filter: PaperFilter实例（包含统计信息）
        conference_dir: 会议目录路径
    """
    filename = conference_dir / "summary.txt"

    with open(filename, "w", encoding="utf-8") as f:
        # 基本信息
        f.write(f"Conference: {conference}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFilter Configuration:\n")
        f.write(
            f"  Rating Range: [{paper_filter.rating_min}, {paper_filter.rating_max}]\n"
        )
        f.write(
            f"  Confidence Range: [{paper_filter.confidence_min}, {paper_filter.confidence_max}]\n"
        )
        f.write(f"  Years: {paper_filter.years if paper_filter.years else 'All'}\n")
        f.write(
            f"  Review Length Range: [{paper_filter.review_length_min}, {paper_filter.review_length_max}] characters\n"
        )

        # 统计数据
        f.write(f"\nStatistics:\n")
        stats = paper_filter.stats
        f.write(f"  Original Papers: {stats['original_papers']:,}\n")
        f.write(f"  Filtered Papers: {stats['filtered_papers']:,}\n")
        if stats["original_papers"] > 0:
            retention_rate = stats["filtered_papers"] / stats["original_papers"] * 100
            f.write(f"  Retention Rate: {retention_rate:.2f}%\n")

        f.write(f"\n  Original Reviews: {stats['original_reviews']:,}\n")
        f.write(f"  Filtered Reviews: {stats['filtered_reviews']:,}\n")
        if stats["original_reviews"] > 0:
            retention_rate = stats["filtered_reviews"] / stats["original_reviews"] * 100
            f.write(f"  Retention Rate: {retention_rate:.2f}%\n")

        # 过滤详情
        f.write(f"\nFilter Details:\n")
        f.write(f"  Papers Removed by Year: {stats['papers_removed_by_year']:,}\n")
        f.write(
            f"  Papers Removed (No Valid Reviews): {stats['papers_removed_no_reviews']:,}\n"
        )
        f.write(f"\n  Review Filtering (Layered Statistics):\n")
        f.write(f"    Removed by Rating: {stats['reviews_removed_by_rating']:,}\n")
        f.write(
            f"    Removed by Confidence: {stats['reviews_removed_by_confidence']:,}\n"
        )
        f.write(f"    Removed by Length: {stats['reviews_removed_by_length']:,}\n")
        # 计算实际被过滤的唯一 reviews 数量
        total_filtered_reviews = stats["original_reviews"] - stats["filtered_reviews"]
        f.write(f"    Total Unique Reviews Filtered: {total_filtered_reviews:,}\n")
        f.write(f"\n  Note: A review may fail multiple criteria.\n")
        f.write("\n" + "-" * 40 + "\n")


def process_conference(
    input_path: Path,
    output_path: Path,
    paper_filter: PaperFilter,
    dry_run: bool = False,
) -> bool:
    """
    处理单个会议的results.json

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        paper_filter: 过滤器实例
        dry_run: 是否为试运行

    Returns:
        bool: 处理是否成功
    """
    print(f"\n{'=' * 70}")
    print(f"处理文件: {input_path}")
    print(f"{'=' * 70}")

    try:
        # 加载数据
        with open(input_path, "r", encoding="utf-8") as f:
            venues = json.load(f)

        print(f"✅ 成功加载 {len(venues)} 个venues")

        # 执行过滤
        filtered_venues = paper_filter.filter_conference(venues)

        print(f"✅ 过滤完成: {len(filtered_venues)} 个venues被保留")

        # 保存结果
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存 results.json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(filtered_venues, f, ensure_ascii=False, indent=2)
            print(f"✅ results.json 已保存到: {output_path}")

            # 生成 summary.txt
            conference_dir = output_path.parent
            conference_name = conference_dir.name
            save_summary_to_txt(conference_name, paper_filter, conference_dir)
            print(f"✅ summary.txt 已保存到: {conference_dir / 'summary.txt'}")
        else:
            print(f"⚠️  试运行模式：文件未保存")

        return True

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="过滤OpenReview论文数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认过滤（只保留confidence>=4的reviews）
  python filter_papers.py

  # 只保留rating>=6的reviews
  python filter_papers.py --rating-min 6

  # 只保留2023-2024年的论文
  python filter_papers.py --years 2023 2024

  # 组合过滤：2023-2024年，rating>=6，confidence>=4
  python filter_papers.py --years 2023 2024 --rating-min 6 --confidence-min 4

  # 只处理特定会议
  python filter_papers.py --conferences AAAI NeurIPS

  # 试运行（不保存文件）
  python filter_papers.py --dry-run
        """,
    )

    # 输入输出
    parser.add_argument(
        "--input", default="raw_data", help="输入目录（默认: raw_data）"
    )
    parser.add_argument(
        "--output", default="filtered_data", help="输出目录（默认: filtered_data）"
    )
    parser.add_argument(
        "--conferences", nargs="+", help="指定会议列表（默认: 所有会议）"
    )

    # 过滤条件
    parser.add_argument(
        "--years", type=int, nargs="+", help="保留的年份列表（默认: 所有年份）"
    )
    parser.add_argument(
        "--rating-min", type=int, default=1, help="Rating最小值（默认: 1）"
    )
    parser.add_argument(
        "--rating-max", type=int, default=10, help="Rating最大值（默认: 10）"
    )
    parser.add_argument(
        "--confidence-min", type=int, default=4, help="Confidence最小值（默认: 4）"
    )
    parser.add_argument(
        "--confidence-max", type=int, default=5, help="Confidence最大值（默认: 5）"
    )
    parser.add_argument(
        "--review-length-min",
        type=int,
        default=0,
        help="Review最小字符数（默认: 0，不限制）",
    )
    parser.add_argument(
        "--review-length-max",
        type=int,
        default=100000,
        help="Review最大字符数（默认: 100000，不限制）",
    )

    # 其他选项
    parser.add_argument(
        "--dry-run", action="store_true", help="试运行，只显示统计不保存文件"
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")

    args = parser.parse_args()

    # 打印配置
    print("=" * 70)
    print("OpenReview 论文数据过滤工具")
    print("=" * 70)
    print("\n过滤配置:")
    print(f"  输入目录: {args.input}")
    print(f"  输出目录: {args.output}")
    if args.conferences:
        print(f"  指定会议: {', '.join(args.conferences)}")
    else:
        print(f"  指定会议: 所有会议")
    print(f"\n  Rating范围: [{args.rating_min}, {args.rating_max}]")
    print(f"  Confidence范围: [{args.confidence_min}, {args.confidence_max}]")
    print(
        f"  Review长度范围: [{args.review_length_min}, {args.review_length_max}] 字符"
    )
    if args.years:
        print(f"  年份列表: {args.years}")
    else:
        print(f"  年份列表: 所有年份")
    print(f"  试运行模式: {'是' if args.dry_run else '否'}")
    print()

    # 创建过滤器
    paper_filter = PaperFilter(
        rating_min=args.rating_min,
        rating_max=args.rating_max,
        confidence_min=args.confidence_min,
        confidence_max=args.confidence_max,
        years=args.years,
        review_length_min=args.review_length_min,
        review_length_max=args.review_length_max,
        verbose=args.verbose,
    )

    # 确定要处理的会议
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return

    # 查找所有results.json
    if args.conferences:
        conference_dirs = [input_dir / conf for conf in args.conferences]
    else:
        conference_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    # 过滤掉没有results.json的目录
    conference_dirs = [d for d in conference_dirs if (d / "results.json").exists()]

    if not conference_dirs:
        print(f"❌ 在 {input_dir} 中未找到任何results.json文件")
        return

    print(f"找到 {len(conference_dirs)} 个会议数据文件")

    # 处理每个会议
    success_count = 0
    for conf_dir in conference_dirs:
        conference_name = conf_dir.name
        input_path = conf_dir / "results.json"
        output_path = Path(args.output) / conference_name / "results.json"

        print(f"\n处理会议: {conference_name}")

        success = process_conference(
            input_path=input_path,
            output_path=output_path,
            paper_filter=paper_filter,
            dry_run=args.dry_run,
        )

        if success:
            success_count += 1

    # 打印总体统计
    print("\n" + "=" * 70)
    print(f"处理完成: {success_count}/{len(conference_dirs)} 个会议成功")
    paper_filter.print_statistics()

    if not args.dry_run:
        print(f"\n✅ 过滤后的数据已保存到: {args.output}")
        print(f"💡 提示: 你可以使用以下命令查看结果:")
        print(f"   ls -lh {args.output}/*/")
    else:
        print(f"\n⚠️  试运行模式：文件未保存")
        print(f"💡 如需保存结果，请去掉 --dry-run 参数重新运行")

    print("=" * 70)


if __name__ == "__main__":
    main()
