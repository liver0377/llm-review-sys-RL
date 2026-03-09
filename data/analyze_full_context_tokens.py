#!/usr/bin/env python3
"""
完整上下文 Token 长度统计脚本

统计 qlora_train.json 和 qlora_validation.json 中
拼接后的完整上下文（prompt + response）的 token 分布。

运行方式：
    conda activate openreview
    python analyze_full_context_tokens.py
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

# 数据集路径
DATA_DIR = Path("/data/wudy/RL/llm-review-sys/data/openreview_dataset")
TRAIN_FILE = DATA_DIR / "sft_train.json"
VAL_FILE = DATA_DIR / "sft_val.json"

# 最大长度阈值（来自配置文件）
MAX_LENGTH = 4096

# 输出路径
OUTPUT_DIR = Path("/data/wudy/RL/llm-review-sys/data")
OUTPUT_IMAGE = OUTPUT_DIR / "full_context_token_analysis.png"
OUTPUT_STATS = OUTPUT_DIR / "full_context_token_stats.json"


# ============== 数据加载 ==============


def load_json_files():
    """加载训练和验证数据集"""
    print("📁 加载数据集...")

    datasets = []

    # 加载训练集
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        datasets.extend(train_data)
        print(f"  ✅ 训练集: {len(train_data)} 条")
    else:
        print(f"  ⚠️  训练集不存在: {TRAIN_FILE}")

    # 加载验证集
    if VAL_FILE.exists():
        with open(VAL_FILE, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        datasets.extend(val_data)
        print(f"  ✅ 验证集: {len(val_data)} 条")
    else:
        print(f"  ⚠️  验证集不存在: {VAL_FILE}")

    total = len(datasets)
    print(f"\n📝 总样本数: {total}")

    return datasets


# ============== Token 统计 ==============


def build_full_context(instruction, input_text, output):
    """
    构建完整上下文（与 train_full_context.py 逻辑一致）

    格式：
    prompt = f"{instruction}\n\n{input}"
    full_text = f"{prompt}\n\n### Response:\n{response}"
    """
    prompt = f"{instruction}\n\n{input_text}".strip()
    full_text = f"{prompt}\n\n### Response:\n{output}"
    return full_text, prompt, output


def count_tokens(text, tokenizer):
    """统计文本的 token 数量"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def analyze_tokens(datasets, tokenizer):
    """
    分析所有样本的 token 数量

    Returns:
        stats: 包含统计信息的字典
    """
    print(f"\n🔢 开始统计 token 数量...")
    print(f"   使用 tokenizer: {tokenizer.name_or_path}")

    results = []

    for item in tqdm(datasets, desc="处理样本"):
        instruction = item["instruction"]
        input_text = item["input"]
        output = item["output"]

        # 构建完整上下文
        full_context, prompt, response = build_full_context(
            instruction, input_text, output
        )

        # 统计 token 数量
        full_tokens = count_tokens(full_context, tokenizer)
        prompt_tokens = count_tokens(prompt, tokenizer)
        response_tokens = count_tokens(response, tokenizer)

        results.append(
            {
                "full_tokens": full_tokens,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "exceeds_max": full_tokens > MAX_LENGTH,
            }
        )

    return results


# ============== 统计计算 ==============


def calculate_statistics(results):
    """计算统计指标"""
    full_tokens = [r["full_tokens"] for r in results]
    prompt_tokens = [r["prompt_tokens"] for r in results]
    response_tokens = [r["response_tokens"] for r in results]
    exceeded = [r["exceeds_max"] for r in results]

    stats = {
        "total_samples": len(results),
        "exceeded_count": sum(exceeded),
        "exceeded_ratio": sum(exceeded) / len(results) * 100,
        # 完整上下文统计
        "full_context": {
            "min": int(np.min(full_tokens)),
            "max": int(np.max(full_tokens)),
            "mean": float(np.mean(full_tokens)),
            "median": float(np.median(full_tokens)),
            "std": float(np.std(full_tokens)),
            "percentiles": {
                "25": float(np.percentile(full_tokens, 25)),
                "50": float(np.percentile(full_tokens, 50)),
                "75": float(np.percentile(full_tokens, 75)),
                "90": float(np.percentile(full_tokens, 90)),
                "95": float(np.percentile(full_tokens, 95)),
                "99": float(np.percentile(full_tokens, 99)),
            },
        },
        # Prompt 统计
        "prompt": {
            "mean": float(np.mean(prompt_tokens)),
            "median": float(np.median(prompt_tokens)),
            "min": int(np.min(prompt_tokens)),
            "max": int(np.max(prompt_tokens)),
        },
        # Response 统计
        "response": {
            "mean": float(np.mean(response_tokens)),
            "median": float(np.median(response_tokens)),
            "min": int(np.min(response_tokens)),
            "max": int(np.max(response_tokens)),
        },
        # 长度分段统计
        "bins": calculate_bins(full_tokens),
    }

    return stats


def calculate_bins(tokens_list):
    """计算长度分段统计"""
    bins = {
        "< 1k": 0,
        "1k-2k": 0,
        "2k-3k": 0,
        "3k-4k": 0,
        "4k-5k": 0,
        "> 5k": 0,
    }

    for tokens in tokens_list:
        if tokens < 1000:
            bins["< 1k"] += 1
        elif tokens < 2000:
            bins["1k-2k"] += 1
        elif tokens < 3000:
            bins["2k-3k"] += 1
        elif tokens < 4000:
            bins["3k-4k"] += 1
        elif tokens < 5000:
            bins["4k-5k"] += 1
        else:
            bins["> 5k"] += 1

    # 转换为百分比
    total = len(tokens_list)
    bins_percent = {k: round(v / total * 100, 1) for k, v in bins.items()}

    return {"count": bins, "percent": bins_percent}


# ============== 可视化 ==============


def create_visualizations(results, stats):
    """创建可视化图表"""
    print(f"\n📊 生成可视化图表...")

    full_tokens = [r["full_tokens"] for r in results]
    prompt_tokens = [r["prompt_tokens"] for r in results]
    response_tokens = [r["response_tokens"] for r in results]

    # 创建 4x2 子图布局
    fig = plt.figure(figsize=(16, 12))

    # 图1: 完整上下文 Token 分布
    ax1 = plt.subplot(3, 2, 1)
    ax1.hist(full_tokens, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(
        MAX_LENGTH,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Max Length ({MAX_LENGTH})",
    )
    ax1.axvline(
        np.median(full_tokens),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median ({int(np.median(full_tokens))})",
    )
    ax1.set_xlabel("Token Count", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Full Context Token Distribution", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 图2: Prompt vs Response Token 分布
    ax2a = plt.subplot(3, 2, 2)
    ax2a.hist(prompt_tokens, bins=50, color="coral", alpha=0.7, edgecolor="black")
    ax2a.set_xlabel("Token Count", fontsize=11)
    ax2a.set_ylabel("Frequency", fontsize=11)
    ax2a.set_title(
        "Prompt Token Distribution (instruction + input)",
        fontsize=12,
        fontweight="bold",
    )
    ax2a.grid(axis="y", alpha=0.3)

    # 图3: 累积分布函数（CDF）
    ax3 = plt.subplot(3, 2, 3)
    sorted_tokens = np.sort(full_tokens)
    cdf = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens)
    ax3.plot(sorted_tokens, cdf, linewidth=2, color="darkblue")

    # 标注关键分位点
    for percentile in [50, 90, 95, 99]:
        val = np.percentile(full_tokens, percentile)
        idx = np.searchsorted(sorted_tokens, val)
        ax3.plot(val, cdf[idx], "ro", markersize=6)
        ax3.annotate(
            f"P{percentile}: {int(val)}",
            xy=(val, cdf[idx]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    ax3.set_xlabel("Token Count", fontsize=11)
    ax3.set_ylabel("Cumulative Probability", fontsize=11)
    ax3.set_title(
        "Cumulative Distribution Function (CDF)", fontsize=12, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3)

    # 图4: 长度分段柱状图
    ax4 = plt.subplot(3, 2, 4)
    bins_data = stats["bins"]["count"]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
    bars = ax4.bar(
        bins_data.keys(), bins_data.values(), color=colors, alpha=0.8, edgecolor="black"
    )

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        percentage = height / stats["total_samples"] * 100
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax4.set_xlabel("Token Range", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("Token Count Bins Distribution", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # 图5: 超长样本分析
    ax5 = plt.subplot(3, 2, 5)
    exceeded_count = stats["exceeded_count"]
    normal_count = stats["total_samples"] - exceeded_count

    sizes = [normal_count, exceeded_count]
    labels = [
        f"≤ {MAX_LENGTH} tokens\n({normal_count} samples, {100 - stats['exceeded_ratio']:.1f}%)",
        f"> {MAX_LENGTH} tokens\n({exceeded_count} samples, {stats['exceeded_ratio']:.1f}%)",
    ]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0, 0.1)

    ax5.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax5.set_title(
        f"Truncation Analysis (Max: {MAX_LENGTH} tokens)",
        fontsize=12,
        fontweight="bold",
    )

    # 图6: Token 数量箱线图
    ax6 = plt.subplot(3, 2, 6)
    box_data = [full_tokens, prompt_tokens, response_tokens]
    bp = ax6.boxplot(
        box_data, labels=["Full Context", "Prompt", "Response"], patch_artist=True
    )

    colors_box = ["steelblue", "coral", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax6.set_ylabel("Token Count", fontsize=11)
    ax6.set_title("Token Count Box Plot Comparison", fontsize=12, fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # 保存图表
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    print(f"  ✅ 图表已保存: {OUTPUT_IMAGE}")

    plt.close()


# ============== 打印统计报告 ==============


def print_statistics(stats):
    """打印统计报告"""
    print(f"\n{'=' * 80}")
    print(f"📊 完整上下文 Token 长度统计报告")
    print(f"{'=' * 80}")
    print(f"\n📁 数据集: qlora_train.json + qlora_validation.json")
    print(f"📝 总样本数: {stats['total_samples']:,}")
    print(f"🤖 模型: Qwen3-8B")
    print(f"✂️  最大长度阈值: {MAX_LENGTH} tokens")

    print(f"\n{'=' * 80}")
    print(f"整体统计（完整上下文）:")
    print(f"{'=' * 80}")
    fc = stats["full_context"]
    print(f"  最小值: {fc['min']:,} tokens")
    print(f"  最大值: {fc['max']:,} tokens")
    print(f"  平均值: {fc['mean']:,.1f} tokens")
    print(f"  中位数: {fc['median']:,.1f} tokens")
    print(f"  标准差: {fc['std']:,.1f} tokens")

    print(f"\n分位数:")
    for p, val in fc["percentiles"].items():
        print(f"  {p}%: {int(val):,} tokens")

    print(f"\n{'=' * 80}")
    print(f"⚠️  截断分析:")
    print(f"{'=' * 80}")
    print(
        f"  超过 {MAX_LENGTH} tokens 的样本: {stats['exceeded_count']:,} 条 ({stats['exceeded_ratio']:.1f}%)"
    )
    exceeded_max = fc["max"] - MAX_LENGTH
    if exceeded_max > 0:
        print(f"  最大超长样本: {fc['max']:,} tokens (超出 {exceeded_max:,} tokens)")

    print(f"\n{'=' * 80}")
    print(f"Prompt 部分 (instruction + input):")
    print(f"{'=' * 80}")
    p = stats["prompt"]
    print(f"  平均: {p['mean']:,.1f} tokens | 中位数: {p['median']:,.1f} tokens")
    print(f"  最小: {p['min']:,} tokens | 最大: {p['max']:,} tokens")

    print(f"\n{'=' * 80}")
    print(f"Response 部分 (output):")
    print(f"{'=' * 80}")
    r = stats["response"]
    print(f"  平均: {r['mean']:,.1f} tokens | 中位数: {r['median']:,.1f} tokens")
    print(f"  最小: {r['min']:,} tokens | 最大: {r['max']:,} tokens")

    print(f"\n{'=' * 80}")
    print(f"长度分段分布:")
    print(f"{'=' * 80}")
    for bin_name, count in stats["bins"]["count"].items():
        percent = stats["bins"]["percent"][bin_name]
        print(f"  {bin_name:8s}: {count:5,} 篇 ({percent:5.1f}%)")

    print(f"\n{'=' * 80}")
    print(f"📊 图表已保存: {OUTPUT_IMAGE}")
    print(f"📄 详细统计: {OUTPUT_STATS}")
    print(f"{'=' * 80}\n")


# ============== 保存详细统计 ==============


def save_statistics(stats, results):
    """保存详细统计到 JSON 文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存统计摘要
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 详细统计已保存: {OUTPUT_STATS}")

    # 可选：保存每个样本的 token 统计（如果需要详细分析）
    # detailed_results = OUTPUT_DIR / "full_context_token_detailed.json"
    # with open(detailed_results, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # print(f"  ✅ 详细结果已保存: {detailed_results}")


# ============== 主函数 ==============


def main():
    """主函数"""
    print(f"\n{'=' * 80}")
    print(f"🚀 开始完整上下文 Token 长度统计")
    print(f"{'=' * 80}\n")

    # 1. 加载 tokenizer
    print(f"📦 加载 Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  ✅ Tokenizer 加载成功: {tokenizer.name_or_path}")
    print(f"  词汇表大小: {len(tokenizer):,}\n")

    # 2. 加载数据集
    datasets = load_json_files()

    if len(datasets) == 0:
        print("❌ 错误: 没有找到任何数据")
        return

    # 3. 统计 token 数量
    results = analyze_tokens(datasets, tokenizer)

    # 4. 计算统计指标
    stats = calculate_statistics(results)

    # 5. 打印统计报告
    print_statistics(stats)

    # 6. 创建可视化
    create_visualizations(results, stats)

    # 7. 保存统计结果
    save_statistics(stats, results)

    print(f"\n✅ 统计完成！\n")


if __name__ == "__main__":
    main()
