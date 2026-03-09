import json
import argparse
import random
import re
from pathlib import Path


def extract_instruction_only(prompt: str) -> str:
    """
    只提取 instruction，不包含任何论文信息
    """
    # Paper Details: 是论文内容的开始标记
    if "Paper Details:" in prompt:
        # 分割并只保留前面部分
        instruction = prompt.split("Paper Details:")[0].strip()
        return instruction
    # 如果没有 Paper Details，返回整个 prompt
    return prompt.strip()


def convert_dpo_to_rm(
    dpo_data_path: str,
    output_path: str,
    max_samples: int = None,
    random_sample: bool = True,
    seed: int = 42,
):
    with open(dpo_data_path, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)

    rm_data = []
    for item in dpo_data:
        # 只提取 instruction 作为 query
        prompt = item.get("prompt", "")
        query = extract_instruction_only(prompt)
        
        rm_item = {
            "query": query,
            "chosen": item.get("chosen", ""),
            "rejected": item.get("rejected", ""),
        }
        if rm_item["query"] and rm_item["chosen"] and rm_item["rejected"]:
            rm_data.append(rm_item)

    if max_samples and max_samples > 0 and max_samples < len(rm_data):
        total_count = len(rm_data)
        if random_sample:
            random.seed(seed)
            rm_data = random.sample(rm_data, max_samples)
            print(
                f"Randomly sampled {max_samples} from {total_count} total samples (seed={seed})"
            )
        else:
            rm_data = rm_data[:max_samples]
            print(f"Selected first {max_samples} samples from {total_count} total")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rm_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(rm_data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert DPO data to RM training data")
    parser.add_argument(
        "--dpo_train",
        type=str,
        default="data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json",
        help="Path to DPO training data",
    )
    parser.add_argument(
        "--dpo_val",
        type=str,
        default="data/openreview_dataset/dpo_vllm_as_rejected_val_cleaned.json",
        help="Path to DPO validation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/openreview_dataset",
        help="Output directory for RM data",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=5000,
        help="Maximum number of training samples (default: 5000, -1 for all)",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        default=True,
        help="Randomly sample data instead of taking first N samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    convert_dpo_to_rm(
        args.dpo_train,
        f"{args.output_dir}/rm_train.json",
        max_samples=args.max_train_samples,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    convert_dpo_to_rm(args.dpo_val, f"{args.output_dir}/rm_val.json")

    print("\nData conversion completed!")
    print(f"RM train data: {args.output_dir}/rm_train.json")
    print(f"RM val data: {args.output_dir}/rm_val.json")


if __name__ == "__main__":
    main()
