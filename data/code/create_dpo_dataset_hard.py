#!/usr/bin/env python3
"""
创建 DPO 数据集（支持单进程并发生成 rejected）

- 占位符模式：默认 rejected=PLACEHOLDER_FOR_REJECTED
- 完整生成模式：--generate-rejected 使用 DashScope API 生成 rejected
- 断点续传：中间结果写入 JSONL，支持恢复
- 并发：使用线程池并发请求（方案 A），主线程顺序写入文件，避免并发写冲突

运行：
  python create_dpo_dataset.py
  python create_dpo_dataset.py --generate-rejected --concurrency 8
"""

from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataset_utils import (
    DatasetConfig,
    load_aggregated_reviews,
    load_marker_md_files,
    match_reviews_and_papers,
    split_dataset,
    check_marker_data_exists,
    print_separator,
)

# =========================
# 保存 DPO 数据集（带标签）
# =========================

def save_dpo_datasets_tagged(train_data, val_data, test_data, tag="hard"):
    output_dir = DatasetConfig.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"dpo_{tag}_as_rejected_train.json"
    val_path = output_dir / f"dpo_{tag}_as_rejected_val.json"
    test_path = output_dir / f"dpo_{tag}_as_rejected_test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    print(f"   - {test_path}")

# =========================
# DPO Rejected Modifier
# =========================

class HardNegativeRejectedModifier:
    """使用 DashScope API 生成 rejected 字段"""

    def __init__(self, api_key=None, api_base=None):
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_base = api_base or os.getenv("DASHSCOPE_API_BASE")

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment. Please check .env file.")
        if not self.api_base:
            raise ValueError("DASHSCOPE_API_BASE not found in environment. Please check .env file.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        print("✅ DashScope API 初始化成功")
        print(f"   API Base: {self.api_base}")

    def modify_rejected(self, chosen_review: str) -> Optional[str]:
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
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )
            if not resp.choices:
                return None
            return resp.choices[0].message.content
        except Exception as e:
            print(f"❌ API 调用出错: {e}")
            return None

# =========================
# 进度管理
# =========================

def load_progress(progress_file: Optional[Path]) -> dict:
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
        "end_time": None,
        "success_count": 0,
        "error_count": 0,
    }

def save_progress(progress_file: Optional[Path], progress_data: dict):
    if not progress_file:
        return
    progress_data["last_update"] = datetime.now().isoformat()
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

# =========================
# JSONL 中间结果（断点续传）
# =========================

def load_jsonl_existing(jsonl_path: Path) -> Dict[int, dict]:
    existing: Dict[int, dict] = {}
    if not jsonl_path.exists():
        return existing
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("index")
            item = obj.get("item")
            if isinstance(idx, int) and isinstance(item, dict):
                existing[idx] = item
    return existing

def append_jsonl(jsonl_path: Path, index: int, item: dict):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"index": index, "item": item}, ensure_ascii=False) + "\n")

# =========================
# DPO 样本构造（不含 rejected）
# =========================

INSTRUCTION = """You are an academic paper reviewer. Please write a structured review of the following paper based solely on its content. Do not include any content beyond the four sections below. Your tone should be professional, constructive, and objective. Base your assessment on typical academic criteria such as novelty, clarity, significance, methodology, and empirical results:

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

def build_prompt_and_chosen(item: dict) -> Tuple[str, str]:
    input_text = f"""Paper Details:
- Title: {item["title"]}

- Conference: {item["conference"]} {item["year"]}

- Content:
{item["paper_content"]}"""

    prompt = f"{INSTRUCTION}\n\n{input_text}".strip()

    rating = item["avg_rating"]
    confidence = item["avg_confidence"]
    chosen = f"""{item["aggregated_review"]}

### Rating
Overall Quality: {rating:.1f}
Review Confidence: {confidence:.1f}"""
    return prompt, chosen

# =========================
# 并发生成 rejected（线程池）
# =========================

def generate_rejected_with_retry(
    modifier: HardNegativeRejectedModifier,
    chosen: str,
    index: int,
    max_retries: int,
    sleep_s: float,
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    返回 (index, rejected_text or None, error_message or None)
    """
    last_err = None
    for attempt in range(max_retries + 1):
        rejected = modifier.modify_rejected(chosen)
        if rejected:
            return index, rejected, None
        last_err = f"empty_response_or_failure (attempt={attempt+1}/{max_retries+1})"
        if sleep_s > 0:
            time.sleep(sleep_s)
    return index, None, last_err

# =========================
# 构建 DPO 数据集（并发版）
# =========================

def build_dpo_format_concurrent(
    matched_data: List[dict],
    generate_rejected: bool,
    modifier: Optional[HardNegativeRejectedModifier],
    progress_file: Optional[Path],
    intermediate_jsonl: Optional[Path],
    concurrency: int = 8,
    max_retries: int = 2,
    per_call_sleep_s: float = 0.0,
):
    progress = load_progress(progress_file)
    processed_indices = set(progress.get("processed_indices", []))

    success_count = int(progress.get("success_count", 0))
    error_count = int(progress.get("error_count", 0))

    existing_items: Dict[int, dict] = {}
    if intermediate_jsonl:
        existing_items = load_jsonl_existing(intermediate_jsonl)
        processed_indices |= set(existing_items.keys())
        if existing_items:
            print(f"📌 从中间结果恢复: {len(existing_items)} 条已生成样本（JSONL）")

    if progress_file and not progress.get("start_time"):
        progress["start_time"] = datetime.now().isoformat()
        save_progress(progress_file, progress)

    if processed_indices:
        print(f"🔄 恢复进度: 已处理 {len(processed_indices)}/{len(matched_data)} 条")

    # 先装载已存在的完整样本，避免续跑得到残缺集合
    dpo_data: List[dict] = []
    if existing_items:
        for idx in sorted(existing_items.keys()):
            dpo_data.append(existing_items[idx])

    # 先把待处理 items 准备好（避免线程里去拼 prompt，保持逻辑一致）
    pending: List[Tuple[int, dict, str]] = []  # (index, dpo_item_without_rejected, chosen)
    for index, raw in enumerate(matched_data):
        if index in processed_indices:
            continue
        prompt, chosen = build_prompt_and_chosen(raw)
        dpo_item = {"prompt": prompt, "chosen": chosen, "rejected": None}
        pending.append((index, dpo_item, chosen))

    if not generate_rejected:
        # 占位符模式：直接填充并写回
        for index, dpo_item, _chosen in tqdm(pending, desc="构建 DPO 数据(占位符)"):
            dpo_item["rejected"] = "PLACEHOLDER_FOR_REJECTED"
            dpo_data.append(dpo_item)
            if intermediate_jsonl:
                append_jsonl(intermediate_jsonl, index=index, item=dpo_item)
            processed_indices.add(index)

        if progress_file:
            progress["processed_indices"] = sorted(list(processed_indices))
            progress["total_items"] = len(matched_data)
            progress["success_count"] = success_count
            progress["error_count"] = error_count
            progress["end_time"] = datetime.now().isoformat()
            save_progress(progress_file, progress)

        print(f"\n✅ 构建了 {len(dpo_data)} 条 DPO 数据")
        return dpo_data

    # 生成 rejected：并发
    if not modifier:
        raise ValueError("generate_rejected=True but modifier is None")

    print(f"⚙️ 并发生成 rejected: concurrency={concurrency}, max_retries={max_retries}")

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for index, dpo_item, chosen in pending:
            fut = ex.submit(
                generate_rejected_with_retry,
                modifier, chosen, index, max_retries, per_call_sleep_s
            )
            futures.append((fut, dpo_item))

        # 用 tqdm 看完成进度（按完成顺序）
        for fut, dpo_item in tqdm(futures, desc="生成 rejected(并发)", total=len(futures)):
            idx, rejected, err = fut.result()
            if rejected:
                dpo_item["rejected"] = rejected
                success_count += 1
            else:
                dpo_item["rejected"] = "PLACEHOLDER_FOR_API_FAILURE"
                error_count += 1

            # 主线程顺序写文件/更新进度（避免并发写）
            dpo_data.append(dpo_item)
            if intermediate_jsonl:
                append_jsonl(intermediate_jsonl, index=idx, item=dpo_item)

            processed_indices.add(idx)
            if progress_file:
                progress["processed_indices"] = sorted(list(processed_indices))
                progress["total_items"] = len(matched_data)
                progress["success_count"] = success_count
                progress["error_count"] = error_count
                save_progress(progress_file, progress)

    if progress_file:
        progress["end_time"] = datetime.now().isoformat()
        progress["success_count"] = success_count
        progress["error_count"] = error_count
        save_progress(progress_file, progress)

    print(f"\n✅ 构建了 {len(dpo_data)} 条 DPO 数据")
    total = success_count + error_count
    if total > 0:
        print(f"   成功: {success_count}/{total} ({success_count/total*100:.1f}%) | 失败: {error_count}/{total} ({error_count/total*100:.1f}%)")
    return dpo_data

# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="创建 DPO 数据集（并发生成 rejected）")
    parser.add_argument("--generate-rejected", action="store_true",
                        help="使用 DashScope API 生成 rejected 字段")
    parser.add_argument("--no-rejected", action="store_true",
                        help="不生成 rejected，使用占位符（快速测试）")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="并发请求数（线程数），建议 2~16")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="单条样本失败重试次数")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="每次失败重试前 sleep 秒数（也可用于轻微限速）")
    return parser.parse_args()

# =========================
# Main
# =========================

def main():
    args = parse_args()

    generate_rejected = bool(args.generate_rejected)
    if args.no_rejected:
        generate_rejected = False

    print_separator()
    if generate_rejected:
        print("🚀 创建完整 DPO 数据集（包含 rejected 字段）")
        print("   使用 DashScope API (qwen-turbo)")
    else:
        print("🚀 创建 DPO 数据集（rejected 使用占位符）")
    print_separator()

    if not check_marker_data_exists():
        print(f"\n⚠️  警告: Marker 输出目录中没有 .md 文件")
        print(f"   目录: {DatasetConfig.PARSED_TEXTS_DIR}")
        print(f"\n请先运行以下命令生成 Marker 解析结果:")
        print(f"   cd /data/wudy/RL/llm-review-sys/data")
        print(f"   conda activate marker-env")
        print(f"   python pdf_parser_marker.py")
        return

    try:
        print("\n📂 加载聚合评审数据...")
        reviews = load_aggregated_reviews()

        print("📂 加载 Marker 解析的 Markdown 文件...")
        marker_papers = load_marker_md_files()

        print("🔗 匹配评审和论文...")
        matched_data = match_reviews_and_papers(reviews, marker_papers)

        if len(matched_data) == 0:
            print("\n❌ 错误: 没有匹配的数据")
            return
        print(f"\n📊 匹配成功: {len(matched_data)} 条数据")

        modifier = None
        if generate_rejected:
            print("\n🔧 初始化 DashScope API...")
            try:
                modifier = HardNegativeRejectedModifier()
                print("✅ API 连接成功，准备生成 rejected 字段")
            except Exception as e:
                print(f"❌ API 初始化失败: {e}")
                print("将使用占位符代替 rejected 字段")
                generate_rejected = False

        progress_file = None
        intermediate_jsonl = None
        if generate_rejected:
            progress_file = DatasetConfig.OUTPUT_DIR / "dpo_hard_generation_progress.json"
            intermediate_jsonl = DatasetConfig.OUTPUT_DIR / "dpo_hard_generation_intermediate.jsonl"

        print("\n📝 开始构建 DPO 数据集...")
        dpo_dataset = build_dpo_format_concurrent(
            matched_data=matched_data,
            generate_rejected=generate_rejected,
            modifier=modifier,
            progress_file=progress_file,
            intermediate_jsonl=intermediate_jsonl,
            concurrency=max(1, int(args.concurrency)),
            max_retries=max(0, int(args.max_retries)),
            per_call_sleep_s=max(0.0, float(args.sleep)),
        )

        if len(dpo_dataset) == 0:
            print("\n❌ 错误: 没有生成 DPO 数据")
            return

        print("\n🔀 划分数据集...")
        train_data, val_data, test_data = split_dataset(
            dpo_dataset,
            random_seed=DatasetConfig.RANDOM_SEED,
        )

        print("\n💾 保存数据集...")
        save_dpo_datasets_tagged(train_data, val_data, test_data, tag="hard")

        print(f"\n{'=' * 80}")
        print("✅ DPO 数据集创建完成！")
        print(f"{'=' * 80}")
        print(f"\n📊 数据集统计:")
        print(f"  原始评审数: {len(reviews)}")
        print(f"  Marker 解析数: {len(marker_papers)}")
        print(f"  成功匹配数: {len(matched_data)}")

        print(f"\n📁 输出目录: {DatasetConfig.OUTPUT_DIR}")
        print(f"\n📂 输出文件:")
        print(f"  - dpo_hard_as_rejected_train.json ({len(train_data)} 条)")
        print(f"  - dpo_hard_as_rejected_val.json ({len(val_data)} 条)")
        print(f"  - dpo_hard_as_rejected_test.json ({len(test_data)} 条)")

        if not generate_rejected:
            print("\n⚠️  重要提示:")
            print("  当前 'rejected' 字段为占位符")
            print("  如需生成完整数据集，请运行:")
            print("  python create_dpo_dataset.py --generate-rejected")
        else:
            if progress_file and progress_file.exists():
                with open(progress_file, "r", encoding="utf-8") as f:
                    prog = json.load(f)
                success = prog.get("success_count", 0)
                error = prog.get("error_count", 0)
                total = success + error
                if total > 0:
                    print("\n📈 生成统计:")
                    print(f"  成功: {success}/{total} ({success/total*100:.1f}%)")
                    print(f"  失败: {error}/{total} ({error/total*100:.1f}%)")
                    print(f"  进度文件: {progress_file}")
                    if intermediate_jsonl:
                        print(f"  中间结果: {intermediate_jsonl}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
