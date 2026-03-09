	#!/usr/bin/env python3
"""
Create DPO dataset - vLLM accelerated version (supports 16k+ long context)

Core optimizations:
✅ Use vLLM engine: replace HF native generation, typically much higher throughput
✅ Automatic batching: dynamic batching to utilize GPU more effectively
✅ Long context: manage memory for 15k+ token inputs
✅ Resume: keep JSONL as the source of truth; flush after each batch

Run:
  # Recommended: generate rejected using vLLM
  python create_dpo_dataset_vllm.py --generate-rejected

  # Specify batch size (adjust based on GPU memory)
  python create_dpo_dataset_vllm.py --generate-rejected --batch-size 4

  # Test mode
  python create_dpo_dataset_vllm.py --generate-rejected --test-mode --num-samples 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from transformers import AutoTokenizer

# vLLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Shared utilities
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
# Save DPO dataset
# =========================

def save_dpo_datasets_tagged(train_data, val_data, test_data, tag: str = "vllm"):
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

    print("✅ Saved:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    print(f"   - {test_path}")

# =========================
# Progress & JSONL resume
# =========================

def load_progress(progress_file: Optional[Path]) -> dict:
    if progress_file and progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
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

def load_jsonl_existing(jsonl_path: Path) -> Dict[int, dict]:
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
        print(f"⚠️ Failed to read JSONL intermediate: {e}")
    return existing

def append_jsonl(jsonl_path: Path, index: int, item: dict):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"index": index, "item": item}, ensure_ascii=False) + "\n")

# =========================
# Instruction template
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

RESPONSE_SEP = "\n\n### Response:\n"

def build_prompt_and_chosen(item: dict) -> Tuple[str, str]:
    input_text = f"""Paper Details:
- Title: {item["title"]}
- Conference: {item["conference"]} {item["year"]}
- Content:
{item["paper_content"]}"""

    # 强约束放在末尾（长上下文更有效）
    strict_tail = (
        "\n\nIMPORTANT:\n"
        "Output ONLY the four sections: Key Points, Strengths and Weaknesses, "
        "Suggestions for Improvement, Rating.\n"
        "Do NOT include a References section, citations list, or any numbered bibliography.\n"
        "Any output containing 'References' is invalid.\n"
    )

    # 关键：追加训练时用的 sep，触发 LoRA 学到的“回答模式”
    prompt = f"{INSTRUCTION}\n\n{input_text}{strict_tail}{RESPONSE_SEP}".strip()

    rating = item["avg_rating"]
    confidence = item["avg_confidence"]
    chosen = f"""{item["aggregated_review"]}

### Rating
Overall Quality: {rating:.1f}
Review Confidence: {confidence:.1f}"""
    return prompt, chosen


# =========================
# vLLM Reviewer
# =========================

class VLLMReviewer:
    """
    High-performance inference based on vLLM:
    - supports LoRA
    - supports long context (16k+)
    - automatic batching
    """

    def __init__(
        self,
        checkpoint: str = "checkpoint-700",
        model_path: str = "pretrained/Qwen/Qwen3-8B",
        adapter_root: str = "/models/qwen3_8b_qlora_full_context_32k",
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
        max_lora_rank: int = 64,
    ):
        self.checkpoint = checkpoint
        self.model_path = model_path
        self.adapter_path = str(Path(adapter_root) / checkpoint)

        if not Path(self.adapter_path).exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {self.adapter_path}")

        print("\n🚀 Initializing vLLM Engine...")
        print(f"   Model: {self.model_path}")
        print(f"   LoRA:  {self.adapter_path}")
        print(f"   Max Model Len: {max_model_len}")

        # vLLM engine
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            tensor_parallel_size=8,
            # enforce_eager=True,  # enable if you hit CUDA graph issues
        )

        # tokenizer (for truncation only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ vLLM Engine Ready.")

    def truncate_prompt_tokens(self, prompt: str, max_input_tokens: int) -> str:
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_input_tokens:
            return prompt
        ids = ids[:max_input_tokens]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text + "\n\n[TRUNCATED DUE TO LENGTH]\n"

    def generate_batch(
        self,
        prompts: List[str],
        max_context_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> List[Optional[str]]:
        # Keep a safety margin for formatting / EOS
        max_input_len = max(256, max_context_tokens - max_new_tokens - 256)
        processed_prompts = [self.truncate_prompt_tokens(p, max_input_len) for p in prompts]

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

        # LoRA request: (name, unique_id, path)
        lora_req = LoRARequest("reviewer_lora", 1, self.adapter_path)

        try:
            outputs = self.llm.generate(
                processed_prompts,
                sampling_params,
                lora_request=lora_req,
                use_tqdm=False,
            )

            results: List[Optional[str]] = []
            for out in outputs:
                if out.outputs:
                    text = out.outputs[0].text.strip()
                    results.append(text if text else None)
                else:
                    results.append(None)
            return results

        except Exception as e:
            print(f"\n❌ vLLM Generation Error: {e}")
            return [None] * len(prompts)

# =========================
# Build DPO (batch + resume)
# =========================

def build_dpo_format_vllm(
    matched_data: List[dict],
    generate_rejected: bool,
    reviewer: Optional[VLLMReviewer],
    progress_file: Optional[Path],
    intermediate_jsonl: Optional[Path],
    batch_size: int = 4,
    sample_ratio: float = 1.0,
    max_samples: Optional[int] = None,
    max_context_tokens: int = 16384,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> List[dict]:
    # 1) sampling
    if sample_ratio < 1.0:
        n = int(len(matched_data) * sample_ratio)
        matched_data = matched_data[:n]
    if max_samples is not None:
        matched_data = matched_data[:max_samples]

    # 2) resume from JSONL (source of truth)
    existing_items: Dict[int, dict] = {}
    processed_indices = set()
    if intermediate_jsonl:
        existing_items = load_jsonl_existing(intermediate_jsonl)
        processed_indices = set(existing_items.keys())
        if existing_items:
            print(f"📌 Resume: found {len(existing_items)} completed items in JSONL")

    # 3) prepare tasks
    pending_tasks: List[Tuple[int, dict, str]] = []
    for idx, raw in enumerate(matched_data):
        if idx in processed_indices:
            continue
        prompt, chosen = build_prompt_and_chosen(raw)
        dpo_item = {"prompt": prompt, "chosen": chosen, "rejected": None}
        pending_tasks.append((idx, dpo_item, prompt))

    # start building final list
    final_data: List[dict] = [existing_items[i] for i in sorted(existing_items.keys())]

    if not generate_rejected:
        print("\n📝 Mode: placeholders only")
        for idx, item, _ in tqdm(pending_tasks, desc="Fill placeholders"):
            item["rejected"] = "PLACEHOLDER"
            final_data.append(item)
            if intermediate_jsonl:
                append_jsonl(intermediate_jsonl, idx, item)
        return final_data

    if reviewer is None:
        raise ValueError("generate_rejected=True but reviewer is None")

    print("\n🚀 vLLM batch inference:")
    print(f"   Pending items: {len(pending_tasks)}")
    print(f"   Batch size:    {batch_size}")
    print(f"   Max context:   {max_context_tokens}")
    print(f"   Max new toks:  {max_new_tokens}")

    progress = load_progress(progress_file)
    if not progress.get("start_time"):
        progress["start_time"] = datetime.now().isoformat()
    progress["total_items"] = len(matched_data)

    success_count = int(progress.get("success_count", 0))
    error_count = int(progress.get("error_count", 0))

    # 4) batch loop
    for start in tqdm(range(0, len(pending_tasks), batch_size), desc="🚀 vLLM running"):
        batch = pending_tasks[start:start + batch_size]
        batch_indices = [x[0] for x in batch]
        batch_items = [x[1] for x in batch]
        batch_prompts = [x[2] for x in batch]

        batch_results = reviewer.generate_batch(
            prompts=batch_prompts,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        for idx, item, result in zip(batch_indices, batch_items, batch_results):
            if result:
                item["rejected"] = result
                success_count += 1
            else:
                item["rejected"] = "GENERATION_FAILED"
                item["_error"] = "empty_or_exception"
                error_count += 1

            final_data.append(item)
            if intermediate_jsonl:
                append_jsonl(intermediate_jsonl, idx, item)

        # update progress stats
        if progress_file:
            progress["success_count"] = success_count
            progress["error_count"] = error_count
            save_progress(progress_file, progress)

    # finalize progress
    if progress_file:
        progress["end_time"] = datetime.now().isoformat()
        progress["success_count"] = success_count
        progress["error_count"] = error_count
        save_progress(progress_file, progress)

    print(f"\n✅ Done. success={success_count}, error={error_count}")
    return final_data

# =========================
# CLI & main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM accelerated DPO dataset generator")
    parser.add_argument("--generate-rejected", action="store_true", help="Generate rejected with vLLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-846", help="LoRA checkpoint directory name")
    parser.add_argument("--model-path", type=str, default="pretrained/Qwen/Qwen3-8B")
    parser.add_argument("--adapter-root", type=str, default="/data/wudy/RL/llm-review-sys/models/qwen3_8b_qlora_full_context_32k")

    # vLLM performance params
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per generate() call")
    parser.add_argument("--max-model-len", type=int, default=32768, help="vLLM engine max model length")

    # generation params
    parser.add_argument("--max-context", type=int, default=16384, help="Input context token cap")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)

    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()

    print_separator()
    print("⚡ vLLM accelerated DPO generator")
    print(f"   Batch Size:  {args.batch_size}")
    print(f"   Max Context: {args.max_context}")
    print_separator()

    if not check_marker_data_exists():
        print("\n⚠️ Please run pdf_parser_marker.py first to generate marker outputs.")
        return

    # 1) load data
    print("\n📂 Loading data...")
    reviews = load_aggregated_reviews()
    marker_papers = load_marker_md_files()
    matched_data = match_reviews_and_papers(reviews, marker_papers)

    if args.test_mode:
        matched_data = matched_data[:args.num_samples]
        print(f"🧪 Test mode: using first {len(matched_data)} items")

    # 2) init reviewer
    reviewer = None
    if args.generate_rejected:
        reviewer = VLLMReviewer(
            checkpoint=args.checkpoint,
            model_path=args.model_path,
            adapter_root=args.adapter_root,
            max_model_len=args.max_model_len,
        )

    # 3) build dataset
    suffix = "_test" if args.test_mode else ""
    progress_file = DatasetConfig.OUTPUT_DIR / f"dpo_vllm_progress{suffix}.json"
    intermediate_jsonl = DatasetConfig.OUTPUT_DIR / f"dpo_vllm_intermediate{suffix}.jsonl"

    dpo_dataset = build_dpo_format_vllm(
        matched_data=matched_data,
        generate_rejected=args.generate_rejected,
        reviewer=reviewer,
        progress_file=progress_file,
        intermediate_jsonl=intermediate_jsonl,
        batch_size=args.batch_size,
        max_context_tokens=args.max_context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # 4) save
    print("\n💾 Saving dataset...")
    train, val, test = split_dataset(dpo_dataset)
    tag = "vllm_test" if args.test_mode else "vllm"
    save_dpo_datasets_tagged(train, val, test, tag=tag)

    print("\n" + "=" * 80)
    print("✅ All done!")
    print("=" * 80)

if __name__ == "__main__":
    main()