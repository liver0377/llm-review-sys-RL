#!/usr/bin/env python3
"""
创建 DPO 数据集（base 版本，单进程并发生成 rejected，支持 16k 上下文）

关键特性（已包含你要的“必须改动 + 增强”）：
- rejected 基于论文内容生成：rejected = API(prompt)，prompt = INSTRUCTION + paper_content
- 并发：线程池 + as_completed（真正并发，不被 fut.result() 串行卡住）
- 16k 上下文：默认 max_context_tokens=16384（Qwen3-8B 支持），并做输入截断保护
- 断点续传不丢数据：JSONL 中间结果为真值源；progress 仅作统计
- 限速/退避：对 429/5xx/timeout 做指数退避 + 重试
- enable_thinking=False：修复 DashScope 400 报错
- 可配置：--concurrency / --max-retries / --sleep / --max-context / --max-new-tokens / --tokenizer-model / --api-model

运行：
  python create_dpo_dataset.py
  python create_dpo_dataset.py --generate-rejected --concurrency 8
"""

from __future__ import annotations

import os
import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 用于 token 级截断（可选；若加载失败则降级为字符截断）
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore

# 导入共享工具模块
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


def save_dpo_datasets_tagged(train_data, val_data, test_data, tag="base"):
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

    print("✅ 已保存:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    print(f"   - {test_path}")


# =========================
# 进度管理（仅统计用途；真值源是 JSONL）
# =========================


def load_progress(progress_file: Optional[Path]) -> dict:
    if progress_file and progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载进度文件失败: {e}")
    return {
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
# JSONL 中间结果（断点续传真值源）
# =========================


def load_jsonl_existing(jsonl_path: Path) -> Dict[int, dict]:
    """
    每行格式：{"index": int, "item": {...dpo_item...}}
    若重复 index，以“最后出现”为准（便于覆盖失败重试结果）。
    """
    existing: Dict[int, dict] = {}
    if not jsonl_path.exists():
        return existing
    try:
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
    except Exception as e:
        print(f"⚠️ 读取 JSONL 中间结果失败: {e}")
    return existing


def append_jsonl(jsonl_path: Path, index: int, item: dict):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"index": index, "item": item}, ensure_ascii=False) + "\n")


# =========================
# 模板与样本构造
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
**Review Confidence:** (1–5， where 5 is very confident)
""".strip()


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
# Token 截断（把 prompt 控制到 16k 上下文预算）
# =========================


class PromptTruncator:
    """
    尝试用 transformers tokenizer 做 token-level 截断。
    若 tokenizer 加载失败，则降级为字符截断。
    """

    def __init__(
        self,
        tokenizer_model: str,
        max_context_tokens: int = 16384,
        max_new_tokens: int = 2048,
        safety_margin_tokens: int = 256,
    ):
        self.max_context_tokens = int(max_context_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.safety_margin_tokens = int(safety_margin_tokens)
        self.max_input_tokens = max(
            256, self.max_context_tokens - self.max_new_tokens - self.safety_margin_tokens
        )

        self.tokenizer = None
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_model, trust_remote_code=True, use_fast=True
                )
            except Exception as e:
                print(f"⚠️ tokenizer 加载失败，将使用字符截断。原因: {e}")

    def truncate(self, prompt: str) -> str:
        if self.tokenizer is None:
            # 粗略字符截断：按经验给一个保守上限（中文更密，按 3 chars/token 粗估）
            approx_max_chars = self.max_input_tokens * 3
            if len(prompt) <= approx_max_chars:
                return prompt
            head = prompt[: approx_max_chars]
            return head + "\n\n[TRUNCATED]\n"

        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_input_tokens:
            return prompt

        # 保留开头 instruction + paper metadata；截掉 paper content 尾部
        trunc_ids = ids[: self.max_input_tokens]
        text = self.tokenizer.decode(trunc_ids, skip_special_tokens=True)
        return text + "\n\n[TRUNCATED]\n"


# =========================
# SimpleReviewer（API 生成 rejected）
# =========================


class SimpleReviewer:
    """使用 DashScope API 基于论文内容生成评审（作为 rejected）"""

    def __init__(
        self,
        api_model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        request_timeout_s: float = 120.0,
    ):
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_base = api_base or os.getenv("DASHSCOPE_API_BASE")
        self.api_model = api_model
        self.request_timeout_s = float(request_timeout_s)

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment. Please check .env file.")
        if not self.api_base:
            raise ValueError("DASHSCOPE_API_BASE not found in environment. Please check .env file.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        print("✅ DashScope API 初始化成功（SimpleReviewer）")
        print(f"   API Base: {self.api_base}")
        print(f"   Model: {self.api_model}")

    def generate_review_once(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        try:
            resp = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": False},
                timeout=self.request_timeout_s,
            )
            if not resp.choices:
                return None
            return resp.choices[0].message.content
        except Exception as e:
            # 上层重试逻辑会处理
            raise e


def is_retryable_error(e: Exception) -> bool:
    msg = str(e).lower()
    # 兼容不同网关返回
    retry_keywords = [
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "too many requests",
        "server error",
        "502",
        "503",
        "504",
        "connection reset",
        "connection aborted",
        "temporarily",
    ]
    return any(k in msg for k in retry_keywords)


# =========================
# 并发生成（线程池 + as_completed）
# =========================


def generate_review_with_retry(
    reviewer: SimpleReviewer,
    truncator: PromptTruncator,
    prompt: str,
    index: int,
    max_retries: int,
    base_sleep_s: float,
    max_output_tokens: int,
    temperature: float,
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Returns: (index, review_or_none, err_msg_or_none)
    """
    prompt = truncator.truncate(prompt)

    for attempt in range(max_retries + 1):
        try:
            out = reviewer.generate_review_once(prompt, max_tokens=max_output_tokens, temperature=temperature)
            if out and out.strip():
                return index, out, None
            err = f"empty_output (attempt={attempt+1}/{max_retries+1})"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if not is_retryable_error(e) and attempt < max_retries:
                # 非可重试错误，直接退出，不浪费配额
                return index, None, err

        # 退避：base_sleep * 2^attempt + jitter
        if attempt < max_retries:
            sleep_s = max(0.0, base_sleep_s) * (2 ** attempt)
            sleep_s += random.random() * 0.25
            time.sleep(sleep_s)

    return index, None, err


def build_dpo_format_concurrent(
    matched_data: List[dict],
    generate_rejected: bool,
    reviewer: Optional[SimpleReviewer],
    progress_file: Optional[Path],
    intermediate_jsonl: Optional[Path],
    truncator: PromptTruncator,
    concurrency: int = 8,
    max_retries: int = 2,
    base_sleep_s: float = 0.5,
    max_output_tokens: int = 2048,
    temperature: float = 0.7,
) -> List[dict]:
    # 以 JSONL 为“真已完成”源（避免 progress 和 JSONL 不一致导致跳过/重复）
    existing_items: Dict[int, dict] = {}
    processed_indices: set[int] = set()

    if intermediate_jsonl:
        existing_items = load_jsonl_existing(intermediate_jsonl)
        processed_indices = set(existing_items.keys())
        if existing_items:
            print(f"📌 从中间结果恢复: {len(existing_items)} 条已生成样本（JSONL）")

    progress = load_progress(progress_file)
    success_count = int(progress.get("success_count", 0))
    error_count = int(progress.get("error_count", 0))

    if progress_file and not progress.get("start_time"):
        progress["start_time"] = datetime.now().isoformat()
        save_progress(progress_file, progress)

    if processed_indices:
        print(f"🔄 恢复进度: 已处理 {len(processed_indices)}/{len(matched_data)} 条")

    # 先把已完成的装入 dpo_data（保持完整集合）
    dpo_data: List[dict] = []
    if existing_items:
        for idx in sorted(existing_items.keys()):
            dpo_data.append(existing_items[idx])

    # 构建 pending
    pending: List[Tuple[int, dict, str]] = []
    for index, raw in enumerate(matched_data):
        if index in processed_indices:
            continue
        prompt, chosen = build_prompt_and_chosen(raw)
        dpo_item = {"prompt": prompt, "chosen": chosen, "rejected": None}
        pending.append((index, dpo_item, prompt))

    # 占位符模式
    if not generate_rejected:
        for index, dpo_item, _prompt in tqdm(pending, desc="构建 DPO 数据(占位符)"):
            dpo_item["rejected"] = "PLACEHOLDER_FOR_REJECTED"
            dpo_data.append(dpo_item)
            if intermediate_jsonl:
                append_jsonl(intermediate_jsonl, index=index, item=dpo_item)
            processed_indices.add(index)

        if progress_file:
            progress["total_items"] = len(matched_data)
            progress["success_count"] = success_count
            progress["error_count"] = error_count
            progress["end_time"] = datetime.now().isoformat()
            save_progress(progress_file, progress)

        print(f"\n✅ 构建了 {len(dpo_data)} 条 DPO 数据")
        return dpo_data

    # rejected 生成模式
    if not reviewer:
        raise ValueError("generate_rejected=True but reviewer is None")

    print(f"⚙️ 并发生成 rejected: concurrency={concurrency}, max_retries={max_retries}")
    print(f"   max_context_tokens={truncator.max_context_tokens}, max_input_tokens≈{truncator.max_input_tokens}, max_output_tokens={max_output_tokens}")

    future_to_meta: Dict[Any, Tuple[int, dict]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for index, dpo_item, prompt in pending:
            fut = ex.submit(
                generate_review_with_retry,
                reviewer,
                truncator,
                prompt,
                index,
                max_retries,
                base_sleep_s,
                max_output_tokens,
                temperature,
            )
            future_to_meta[fut] = (index, dpo_item)

        for fut in tqdm(as_completed(future_to_meta), desc="生成 rejected(并发)", total=len(future_to_meta)):
            idx, dpo_item = future_to_meta[fut]
            try:
                real_idx, rejected, err = fut.result()
            except Exception as e:
                real_idx, rejected, err = idx, None, f"{type(e).__name__}: {e}"

            if rejected:
                dpo_item["rejected"] = rejected
                success_count += 1
            else:
                dpo_item["rejected"] = "PLACEHOLDER_FOR_API_FAILURE"
                error_count += 1
                if err:
                    # 可选：把错误信息塞进一个字段，便于排查（训练时可忽略）
                    dpo_item["_error"] = err

            dpo_data.append(dpo_item)

            if intermediate_jsonl:
                append_jsonl(intermediate_jsonl, index=real_idx, item=dpo_item)

            processed_indices.add(real_idx)

            # 进度：每完成一条就写（稳妥；你也可以改成每 N 条写）
            if progress_file:
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
    parser = argparse.ArgumentParser(
        description="创建 DPO 数据集（base，支持并发生成 rejected，16k 上下文）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generate-rejected", action="store_true", help="使用 DashScope API 生成 rejected 字段")
    parser.add_argument("--no-rejected", action="store_true", help="不生成 rejected，使用占位符（快速测试）")

    parser.add_argument("--concurrency", type=int, default=8, help="并发请求数（线程数），建议 2~16")
    parser.add_argument("--max-retries", type=int, default=2, help="单条样本失败重试次数")
    parser.add_argument("--sleep", type=float, default=0.5, help="退避基准 sleep 秒数（指数退避基数）")

    parser.add_argument("--max-context", type=int, default=16384, help="上下文 token 上限（默认 16384）")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="生成输出 token 上限（默认 2048）")
    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-8B", help="用于截断的 tokenizer 模型名/路径")
    parser.add_argument("--api-model", type=str, default="qwen3-8b", help="DashScope API 模型名")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度（默认 0.7）")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时（秒）")

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
        print("   rejected = API(prompt), prompt = instruction + paper content")
        print(f"   API Model: {args.api_model}")
        print(f"   Max context: {args.max_context}, Max new tokens: {args.max_new_tokens}")
    else:
        print("🚀 创建 DPO 数据集（rejected 使用占位符）")
    print_separator()

    if not check_marker_data_exists():
        print("\n⚠️  警告: Marker 输出目录中没有 .md 文件")
        print(f"   目录: {DatasetConfig.PARSED_TEXTS_DIR}")
        print("\n请先运行以下命令生成 Marker 解析结果:")
        print("   cd /data/wudy/RL/llm-review-sys/data")
        print("   conda activate marker-env")
        print("   python pdf_parser_marker.py")
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

        truncator = PromptTruncator(
            tokenizer_model=args.tokenizer_model,
            max_context_tokens=int(args.max_context),
            max_new_tokens=int(args.max_new_tokens),
            safety_margin_tokens=256,
        )

        reviewer = None
        progress_file = None
        intermediate_jsonl = None

        if generate_rejected:
            print("\n🔧 初始化 DashScope API（SimpleReviewer）...")
            try:
                reviewer = SimpleReviewer(
                    api_model=args.api_model,
                    request_timeout_s=float(args.timeout),
                )
                print("✅ API 连接成功，准备并发生成 rejected")
            except Exception as e:
                print(f"❌ API 初始化失败: {e}")
                print("将使用占位符代替 rejected 字段")
                generate_rejected = False

        if generate_rejected:
            progress_file = DatasetConfig.OUTPUT_DIR / "dpo_base_generation_progress.json"
            intermediate_jsonl = DatasetConfig.OUTPUT_DIR / "dpo_base_generation_intermediate.jsonl"

        print("\n📝 开始构建 DPO 数据集...")
        dpo_dataset = build_dpo_format_concurrent(
            matched_data=matched_data,
            generate_rejected=generate_rejected,
            reviewer=reviewer,
            progress_file=progress_file,
            intermediate_jsonl=intermediate_jsonl,
            truncator=truncator,
            concurrency=max(1, int(args.concurrency)),
            max_retries=max(0, int(args.max_retries)),
            base_sleep_s=max(0.0, float(args.sleep)),
            max_output_tokens=max(16, int(args.max_new_tokens)),
            temperature=float(args.temperature),
        )

        if len(dpo_dataset) == 0:
            print("\n❌ 错误: 没有生成 DPO 数据")
            return

        print("\n🔀 划分数据集...")
        train_data, val_data, test_data = split_dataset(
            dpo_dataset, random_seed=DatasetConfig.RANDOM_SEED
        )

        print("\n💾 保存数据集...")
        save_dpo_datasets_tagged(train_data, val_data, test_data, tag="base")

        print("\n" + "=" * 80)
        print("✅ DPO 数据集创建完成！")
        print("=" * 80)

        print("\n📊 数据集统计:")
        print(f"  原始评审数: {len(reviews)}")
        print(f"  Marker 解析数: {len(marker_papers)}")
        print(f"  成功匹配数: {len(matched_data)}")
        print(f"\n📁 输出目录: {DatasetConfig.OUTPUT_DIR}")
        print("\n📂 输出文件:")
        print(f"  - dpo_base_as_rejected_train.json ({len(train_data)} 条)")
        print(f"  - dpo_base_as_rejected_val.json ({len(val_data)} 条)")
        print(f"  - dpo_base_as_rejected_test.json ({len(test_data)} 条)")

        if not generate_rejected:
            print("\n⚠️  重要提示:")
            print("  当前 'rejected' 字段为占位符")
            print("  如需生成完整数据集，请运行:")
            print("  python create_dpo_dataset.py --generate-rejected --concurrency 8")
        else:
            if progress_file and progress_file.exists():
                with open(progress_file, "r", encoding="utf-8") as f:
                    prog = json.load(f)
                success = prog.get("success_count", 0)
                error = prog.get("error_count", 0)
                total = success + error
                print("\n📈 生成统计:")
                if total > 0:
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
