import json
import os
from pathlib import Path


def analyze_single_conference(results_path):
    """分析单个会议的rating和confidence分布"""

    with open(results_path, "r", encoding="utf-8") as f:
        venues = json.load(f)

    total_papers = 0
    papers_with_rating = 0
    papers_without_rating = 0
    papers_with_confidence = 0
    papers_without_confidence = 0
    papers_with_both = 0
    papers_with_neither = 0

    for venue in venues:
        for paper in venue.get("papers", []):
            total_papers += 1
            reviews = paper.get("reviews", [])

            has_rating = False
            has_confidence = False

            for review in reviews:
                ratings = review.get("ratings", {})
                if "rating" in ratings:
                    has_rating = True
                if "confidence" in ratings:
                    has_confidence = True

            if has_rating:
                papers_with_rating += 1
            else:
                papers_without_rating += 1

            if has_confidence:
                papers_with_confidence += 1
            else:
                papers_without_confidence += 1

            if has_rating and has_confidence:
                papers_with_both += 1
            elif not has_rating and not has_confidence:
                papers_with_neither += 1

    return {
        "total_papers": total_papers,
        "papers_with_rating": papers_with_rating,
        "papers_without_rating": papers_without_rating,
        "papers_with_confidence": papers_with_confidence,
        "papers_without_confidence": papers_without_confidence,
        "papers_with_both": papers_with_both,
        "papers_with_neither": papers_with_neither,
    }



def main():
    base_dir = Path(__file__).resolve().parent / "raw_data"
    conferences = [
        "ICML",
        "AAAI",
        "NeurIPS",
        # "ICLR",
        # "CVPR",
    ]

    print("=" * 70)
    print("Rating & Confidence Distribution Analysis")
    print("=" * 70)
    print()

    all_stats = {}
    total_papers_all = 0
    total_without_rating_all = 0
    total_without_confidence_all = 0
    total_with_neither_all = 0

    for conf in conferences:
        results_path = base_dir / conf / "results.json"

        if not results_path.exists():
            print(f"⚠️  {conf}: results.json not found, skipping...")
            continue

        stats = analyze_single_conference(results_path)
        all_stats[conf] = stats

        total_papers = stats["total_papers"]
        papers_without_rating = stats["papers_without_rating"]
        papers_without_confidence = stats["papers_without_confidence"]
        papers_with_neither = stats["papers_with_neither"]

        total_papers_all += total_papers
        total_without_rating_all += papers_without_rating
        total_without_confidence_all += papers_without_confidence
        total_with_neither_all += papers_with_neither

        # 打印单个会议统计
        print(f"Conference: {conf}")
        print(f"  Total Papers: {total_papers:,}")
        print(
            f"  ✅ Has Rating: {stats['papers_with_rating']:,} ({stats['papers_with_rating'] / total_papers * 100:.1f}%)"
        )
        print(
            f"  ❌ No Rating: {papers_without_rating:,} ({papers_without_rating / total_papers * 100:.1f}%)"
        )
        print(
            f"  ✅ Has Confidence: {stats['papers_with_confidence']:,} ({stats['papers_with_confidence'] / total_papers * 100:.1f}%)"
        )
        print(
            f"  ❌ No Confidence: {papers_without_confidence:,} ({papers_without_confidence / total_papers * 100:.1f}%)"
        )
        print(
            f"  ✅ Has Both Rating & Confidence: {stats['papers_with_both']:,} ({stats['papers_with_both'] / total_papers * 100:.1f}%)"
        )
        print(
            f"  ❌ Has Neither: {papers_with_neither:,} ({papers_with_neither / total_papers * 100:.1f}%)"
        )

        print()


    # 打印总结
    print("=" * 70)
    print("Summary Across All Conferences")
    print("=" * 70)
    print(f"Total Papers: {total_papers_all:,}")
    print(
        f"Papers Without Rating: {total_without_rating_all:,} ({total_without_rating_all / total_papers_all * 100:.2f}%)"
    )
    print(
        f"Papers Without Confidence: {total_without_confidence_all:,} ({total_without_confidence_all / total_papers_all * 100:.2f}%)"
    )
    print(
        f"Papers With Neither: {total_with_neither_all:,} ({total_with_neither_all / total_papers_all * 100:.2f}%)"
    )
    print()


if __name__ == "__main__":
    main()
