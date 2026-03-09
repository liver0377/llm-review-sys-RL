#!/usr/bin/env python3
"""
DPO 训练 Token 长度统计脚本

统计 DPO 数据集中 chosen 和 rejected 的 token 分布，
对比分析三种数据集类型（base/hard/vllm）。

运行方式：
    conda activate openreview
    python data/analyze_dpo_tokens.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer


# ============== 配置 ==============

# 模型路径
MODEL_PATH = "/data/wudy/RL/llm-review-sys/pretrained/Qwen/Qwen3-8B"

# 数据集配置
DATA_DIR = Path("/data/wudy/RL/llm-review-sys/data/openreview_dataset")
DATASET_TYPES = ["base", "hard", "vllm"]

# 最大长度阈值（来自 DPO 配置）
MAX_LENGTH = 18000

# 输出路径
OUTPUT_DIR = Path("/data/wudy/RL/llm-review-sys/data")
OUTPUT_IMAGE = OUTPUT_DIR / "dpo_token_analysis.png"
OUTPUT_STATS = OUTPUT_DIR / "dpo_token_stats.json"


# ============== 数据加载 ==============


def load_dpo_datasets(dataset_type):
    """加载指定类型的数据集（train + val）"""
    train_file = DATA_DIR / f"dpo_{dataset_type}_as_rejected_train_cleaned.json"
    val_file = DATA_DIR / f"dpo_{dataset_type}_as_rejected_val_cleaned.json"

    if not train_file.exists():
        raise FileNotFoundError(f"训练集不存在: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"验证集不存在: {val_file}")

    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    return train_data, val_data


def load_all_datasets():
    """加载所有三种类型的数据集"""
    print("📁 加载 DPO 数据集...")

    datasets = {}
    for dtype in DATASET_TYPES:
        try:
            train_data, val_data = load_dpo_datasets(dtype)
            datasets[dtype] = {"train": train_data, "val": val_data}
            print(f"  ✅ {dtype}: train={len(train_data):,}, val={len(val_data):,}")
        except FileNotFoundError as e:
            print(f"  ⚠️  {dtype}: {e}")

    total_train = sum(len(d["train"]) for d in datasets.values())
    total_val = sum(len(d["val"]) for d in datasets.values())

    print(f"\n📝 总样本数: train={total_train:,}, val={total_val:,}")

    return datasets


# ============== Token 统计 ==============


def count_dpo_tokens(prompt, chosen, rejected, tokenizer):
    """统计 DPO 样本的 token 数量

    Returns:
        dict: 包含各个部分的 token 统计
    """
    prompt_toks = len(tokenizer.encode(prompt, add_special_tokens=False))
    chosen_toks = len(tokenizer.encode(chosen, add_special_tokens=False))
    rejected_toks = len(tokenizer.encode(rejected, add_special_tokens=False))

    # 完整序列（用于训练）
    prompt_chosen = len(tokenizer.encode(prompt + chosen, add_special_tokens=False))
    prompt_rejected = len(tokenizer.encode(prompt + rejected, add_special_tokens=False))

    return {
        "prompt_tokens": prompt_toks,
        "chosen_tokens": chosen_toks,
        "rejected_tokens": rejected_toks,
        "prompt_chosen_tokens": prompt_chosen,
        "prompt_rejected_tokens": prompt_rejected,
    }


def analyze_dpo_tokens(datasets, tokenizer):
    """分析所有数据集的 token 数量"""
    print(f"\n🔢 开始统计 token 数量...")
    print(f"   使用 tokenizer: {tokenizer.name_or_path}")

    results = {}

    for dtype, data in datasets.items():
        print(f"\n  分析 {dtype} 数据集...")

        results[dtype] = {"train": [], "val": []}

        # 分析训练集
        for item in tqdm(data["train"], desc=f"  {dtype} train", leave=False):
            tokens = count_dpo_tokens(
                item["prompt"], item["chosen"], item["rejected"], tokenizer
            )
            results[dtype]["train"].append(tokens)

        # 分析验证集
        for item in tqdm(data["val"], desc=f"  {dtype} val", leave=False):
            tokens = count_dpo_tokens(
                item["prompt"], item["chosen"], item["rejected"], tokenizer
            )
            results[dtype]["val"].append(tokens)

    return results


# ============== 统计计算 ==============


def calculate_percentiles(data):
    """计算分位数统计"""
    return {
        "min": int(np.min(data)),
        "max": int(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "percentiles": {
            "25": float(np.percentile(data, 25)),
            "50": float(np.percentile(data, 50)),
            "75": float(np.percentile(data, 75)),
            "90": float(np.percentile(data, 90)),
            "95": float(np.percentile(data, 95)),
            "99": float(np.percentile(data, 99)),
        },
    }


def calculate_dpo_statistics(all_results):
    """计算 DPO 数据集的统计指标"""
    print(f"\n📊 计算统计指标...")

    stats = {}

    for dtype, results in all_results.items():
        stats[dtype] = {}

        for split, data in results.items():
            # 提取各项指标
            prompt_chosen = [r["prompt_chosen_tokens"] for r in data]
            prompt_rejected = [r["prompt_rejected_tokens"] for r in data]
            chosen_only = [r["chosen_tokens"] for r in data]
            rejected_only = [r["rejected_tokens"] for r in data]
            prompt_only = [r["prompt_tokens"] for r in data]

            # 计算超过 MAX_LENGTH 的样本数
            exceeded_chosen = sum(1 for x in prompt_chosen if x > MAX_LENGTH)
            exceeded_rejected = sum(1 for x in prompt_rejected if x > MAX_LENGTH)

            # 比较 chosen 和 rejected
            chosen_longer = sum(
                1 for c, r in zip(prompt_chosen, prompt_rejected) if c > r
            )
            rejected_longer = sum(
                1 for c, r in zip(prompt_chosen, prompt_rejected) if r > c
            )
            equal_length = sum(
                1 for c, r in zip(prompt_chosen, prompt_rejected) if c == r
            )

            stats[dtype][split] = {
                "total_samples": len(data),
                "exceeded_max": {
                    "chosen": exceeded_chosen,
                    "rejected": exceeded_rejected,
                    "chosen_ratio": exceeded_chosen / len(data) * 100,
                    "rejected_ratio": exceeded_rejected / len(data) * 100,
                },
                "full_sequences": {
                    "chosen": calculate_percentiles(prompt_chosen),
                    "rejected": calculate_percentiles(prompt_rejected),
                },
                "responses_only": {
                    "chosen": calculate_percentiles(chosen_only),
                    "rejected": calculate_percentiles(rejected_only),
                },
                "prompts_only": calculate_percentiles(prompt_only),
                "comparison": {
                    "avg_diff": float(
                        np.mean(prompt_rejected) - np.mean(prompt_chosen)
                    ),
                    "chosen_longer": chosen_longer,
                    "rejected_longer": rejected_longer,
                    "equal_length": equal_length,
                    "chosen_longer_ratio": chosen_longer / len(data) * 100,
                    "rejected_longer_ratio": rejected_longer / len(data) * 100,
                },
            }

    return stats


# ============== 可视化 ==============


def create_visualizations(results, stats):
    """创建 DPO token 分析可视化"""
    print(f"\n📊 生成可视化图表...")

    # 创建 3x2 子图布局
    fig = plt.figure(figsize=(18, 12))

    # 图1: Chosen vs Rejected 分布对比（训练集）
    ax1 = plt.subplot(3, 2, 1)
    for dtype, color in [
        ("base", "steelblue"),
        ("hard", "coral"),
        ("vllm", "lightgreen"),
    ]:
        train_data = results[dtype]["train"]
        chosen_tokens = [r["prompt_chosen_tokens"] for r in train_data]
        ax1.hist(
            chosen_tokens,
            bins=50,
            alpha=0.3,
            color=color,
            label=f"{dtype} chosen",
            density=True,
        )
        rejected_tokens = [r["prompt_rejected_tokens"] for r in train_data]
        ax1.hist(
            rejected_tokens,
            bins=50,
            alpha=0.15,
            color=color,
            label=f"{dtype} rejected",
            density=True,
            linestyle="--",
        )

    ax1.axvline(
        MAX_LENGTH,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Max Length ({MAX_LENGTH})",
    )
    ax1.set_xlabel("Token Count", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        "Token Distribution: Chosen vs Rejected (Training Set)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # 图2: 数据集类型对比（平均 token 数）
    ax2 = plt.subplot(3, 2, 2)
    dataset_names = []
    chosen_means = []
    rejected_means = []

    for dtype in DATASET_TYPES:
        train_stats = stats[dtype]["train"]["full_sequences"]
        dataset_names.append(dtype.upper())
        chosen_means.append(train_stats["chosen"]["mean"])
        rejected_means.append(train_stats["rejected"]["mean"])

    x = np.arange(len(dataset_names))
    width = 0.35

    ax2.bar(
        x - width / 2, chosen_means, width, label="Chosen", color="steelblue", alpha=0.8
    )
    ax2.bar(
        x + width / 2, rejected_means, width, label="Rejected", color="coral", alpha=0.8
    )

    ax2.set_ylabel("Average Token Count", fontsize=11)
    ax2.set_title("Average Token Count by Dataset Type", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 添加数值标签
    for i, (c, r) in enumerate(zip(chosen_means, rejected_means)):
        ax2.text(
            i - width / 2, c + 100, f"{c:.0f}", ha="center", va="bottom", fontsize=9
        )
        ax2.text(
            i + width / 2, r + 100, f"{r:.0f}", ha="center", va="bottom", fontsize=9
        )

    # 图3: 超过 MAX_LENGTH 的样本比例
    ax3 = plt.subplot(3, 2, 3)
    exceeded_data = []
    for dtype in DATASET_TYPES:
        train_stats = stats[dtype]["train"]["exceeded_max"]
        exceeded_data.append(
            [train_stats["chosen_ratio"], train_stats["rejected_ratio"]]
        )

    exceeded_data = np.array(exceeded_data)
    x = np.arange(len(DATASET_TYPES))
    width = 0.35

    ax3.bar(
        x - width / 2,
        exceeded_data[:, 0],
        width,
        label="Chosen",
        color="steelblue",
        alpha=0.8,
    )
    ax3.bar(
        x + width / 2,
        exceeded_data[:, 1],
        width,
        label="Rejected",
        color="coral",
        alpha=0.8,
    )

    ax3.set_ylabel("Percentage (%)", fontsize=11)
    ax3.set_title(
        f"Samples Exceeding Max Length ({MAX_LENGTH} tokens)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.upper() for d in DATASET_TYPES])
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # 添加数值标签
    for i, (c, r) in enumerate(exceeded_data):
        ax3.text(
            i - width / 2, c + 0.05, f"{c:.1f}%", ha="center", va="bottom", fontsize=9
        )
        ax3.text(
            i + width / 2, r + 0.05, f"{r:.1f}%", ha="center", va="bottom", fontsize=9
        )

    # 图4: 累积分布函数（CDF）- Base 数据集
    ax4 = plt.subplot(3, 2, 4)
    train_data = results["base"]["train"]
    chosen_tokens = sorted([r["prompt_chosen_tokens"] for r in train_data])
    rejected_tokens = sorted([r["prompt_rejected_tokens"] for r in train_data])

    cdf_chosen = np.arange(1, len(chosen_tokens) + 1) / len(chosen_tokens)
    cdf_rejected = np.arange(1, len(rejected_tokens) + 1) / len(rejected_tokens)

    ax4.plot(chosen_tokens, cdf_chosen, linewidth=2, color="steelblue", label="Chosen")
    ax4.plot(
        rejected_tokens,
        cdf_rejected,
        linewidth=2,
        color="coral",
        label="Rejected",
        linestyle="--",
    )

    # 标注关键分位点
    for percentile in [90, 95, 99]:
        val = np.percentile(chosen_tokens, percentile)
        idx = np.searchsorted(chosen_tokens, val)
        ax4.plot(val, cdf_chosen[idx], "ro", markersize=5)
        ax4.annotate(
            f"P{percentile}: {int(val)}",
            xy=(val, cdf_chosen[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    ax4.set_xlabel("Token Count", fontsize=11)
    ax4.set_ylabel("Cumulative Probability", fontsize=11)
    ax4.set_title(
        "Cumulative Distribution (Base Dataset)", fontsize=12, fontweight="bold"
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 图5: Chosen vs Rejected 长度比较
    ax5 = plt.subplot(3, 2, 5)
    comparison_data = []
    labels = []

    for dtype in DATASET_TYPES:
        comp = stats[dtype]["train"]["comparison"]
        comparison_data.append(
            [comp["chosen_longer_ratio"], comp["rejected_longer_ratio"]]
        )
        labels.append(dtype.upper())

    comparison_data = np.array(comparison_data)
    x = np.arange(len(labels))
    width = 0.35

    ax5.bar(
        x - width / 2,
        comparison_data[:, 0],
        width,
        label="Chosen > Rejected",
        color="steelblue",
        alpha=0.8,
    )
    ax5.bar(
        x + width / 2,
        comparison_data[:, 1],
        width,
        label="Rejected > Chosen",
        color="coral",
        alpha=0.8,
    )

    ax5.set_ylabel("Percentage (%)", fontsize=11)
    ax5.set_title(
        "Which is Longer: Chosen or Rejected?", fontsize=12, fontweight="bold"
    )
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    # 添加数值标签
    for i, (c, r) in enumerate(comparison_data):
        ax5.text(
            i - width / 2, c + 0.5, f"{c:.1f}%", ha="center", va="bottom", fontsize=9
        )
        ax5.text(
            i + width / 2, r + 0.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=9
        )

    # 图6: 箱线图对比
    ax6 = plt.subplot(3, 2, 6)
    box_data = []
    box_labels = []

    for dtype in DATASET_TYPES:
        train_data = results[dtype]["train"]
        chosen_tokens = [r["prompt_chosen_tokens"] for r in train_data]
        rejected_tokens = [r["prompt_rejected_tokens"] for r in train_data]
        box_data.extend([chosen_tokens, rejected_tokens])
        box_labels.extend([f"{dtype}\nChosen", f"{dtype}\nRejected"])

    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)

    # 为每个箱子设置颜色
    colors = ["steelblue", "coral"] * 3
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax6.set_ylabel("Token Count", fontsize=11)
    ax6.set_title("Token Count Box Plot Comparison", fontsize=12, fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # 保存图表
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    print(f"  ✅ 图表已保存: {OUTPUT_IMAGE}")

    plt.close()


# ============== 打印统计报告 ==============


def print_dpo_statistics(stats):
    """打印 DPO 统计报告"""
    print(f"\n{'=' * 80}")
    print(f"📊 DPO 训练 Token 长度统计报告")
    print(f"{'=' * 80}")
    print(f"\n🤖 模型: Qwen3-8B")
    print(f"✂️  最大长度阈值: {MAX_LENGTH:,} tokens")
    print(f"📁 数据集类型: {', '.join(DATASET_TYPES)}")

    for dtype in DATASET_TYPES:
        print(f"\n{'=' * 80}")
        print(f"数据集: {dtype.upper()}")
        print(f"{'=' * 80}")

        train_stats = stats[dtype]["train"]
        val_stats = stats[dtype]["val"]

        print(f"\n训练集: {train_stats['total_samples']:,} 样本")
        print(f"验证集: {val_stats['total_samples']:,} 样本")

        # 超长样本分析
        print(f"\n⚠️  超长样本分析（训练集）:")
        print(
            f"  Chosen  超过 MAX_LENGTH: {train_stats['exceeded_max']['chosen']:,} ({train_stats['exceeded_max']['chosen_ratio']:.2f}%)"
        )
        print(
            f"  Rejected 超过 MAX_LENGTH: {train_stats['exceeded_max']['rejected']:,} ({train_stats['exceeded_max']['rejected_ratio']:.2f}%)"
        )

        # 完整序列统计
        print(f"\n📏 完整序列统计 (Prompt + Response):")
        fc_chosen = train_stats["full_sequences"]["chosen"]
        fc_rejected = train_stats["full_sequences"]["rejected"]

        print(f"  Chosen:")
        print(
            f"    平均: {fc_chosen['mean']:,.0f} | 中位数: {fc_chosen['median']:,.0f}"
        )
        print(f"    范围: [{fc_chosen['min']:,}, {fc_chosen['max']:,}]")
        print(f"    标准差: {fc_chosen['std']:,.0f}")

        print(f"  Rejected:")
        print(
            f"    平均: {fc_rejected['mean']:,.0f} | 中位数: {fc_rejected['median']:,.0f}"
        )
        print(f"    范围: [{fc_rejected['min']:,}, {fc_rejected['max']:,}]")
        print(f"    标准差: {fc_rejected['std']:,.0f}")

        # 分位数
        print(f"\n  分位数 (Chosen):")
        for p, val in fc_chosen["percentiles"].items():
            print(f"    {p}%: {int(val):,} tokens")

        # Chosen vs Rejected 比较
        comp = train_stats["comparison"]
        print(f"\n🔄 Chosen vs Rejected 比较:")
        print(f"  平均差异: {comp['avg_diff']:+.0f} tokens (Rejected - Chosen)")
        print(
            f"  Chosen 更长: {comp['chosen_longer']:,} ({comp['chosen_longer_ratio']:.1f}%)"
        )
        print(
            f"  Rejected 更长: {comp['rejected_longer']:,} ({comp['rejected_longer_ratio']:.1f}%)"
        )
        print(f"  长度相等: {comp['equal_length']:,}")

        # Response only 统计
        resp_chosen = train_stats["responses_only"]["chosen"]
        resp_rejected = train_stats["responses_only"]["rejected"]

        print(f"\n📝 仅 Response 统计:")
        print(
            f"  Chosen  - 平均: {resp_chosen['mean']:,.0f} | 中位数: {resp_chosen['median']:,.0f}"
        )
        print(
            f"  Rejected - 平均: {resp_rejected['mean']:,.0f} | 中位数: {resp_rejected['median']:,.0f}"
        )

    print(f"\n{'=' * 80}")
    print(f"📊 图表已保存: {OUTPUT_IMAGE}")
    print(f"📄 详细统计: {OUTPUT_STATS}")
    print(f"{'=' * 80}\n")


# ============== 保存详细统计 ==============


def save_statistics(stats):
    """保存详细统计到 JSON 文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 详细统计已保存: {OUTPUT_STATS}")


# ============== 主函数 ==============


def main():
    """主函数"""
    print(f"\n{'=' * 80}")
    print(f"🚀 开始 DPO Token 长度统计")
    print(f"{'=' * 80}\n")

    # 1. 加载 tokenizer
    print(f"📦 加载 Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  ✅ Tokenizer 加载成功: {tokenizer.name_or_path}")
    print(f"  词汇表大小: {len(tokenizer):,}\n")

    # 2. 加载所有数据集
    datasets = load_all_datasets()

    if len(datasets) == 0:
        print("❌ 错误: 没有找到任何数据")
        return

    # 3. 统计 token 数量
    results = analyze_dpo_tokens(datasets, tokenizer)

    # 4. 计算统计指标
    stats = calculate_dpo_statistics(results)

    # 5. 打印统计报告
    print_dpo_statistics(stats)

    # 6. 创建可视化
    create_visualizations(results, stats)

    # 7. 保存统计结果
    save_statistics(stats)

    print(f"\n✅ 统计完成！\n")


if __name__ == "__main__":
    main()
