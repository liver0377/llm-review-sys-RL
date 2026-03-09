import json
import argparse
import random
from pathlib import Path


def prepare_grpo_data(
    sft_data_path: str,
    output_path: str,
    max_samples: int = None,
    random_sample: bool = True,
    seed: int = 42,
):
    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    grpo_data = []
    for item in sft_data:
        if isinstance(item, dict):
            if "input" in item and item["input"]:
                prompt = f"{item.get('instruction', '')}\n\n{item.get('input', '')}"
            else:
                prompt = item.get("instruction", "") or item.get("prompt", "")

            if prompt:
                # Swift GRPO 需要 messages 格式
                grpo_item = {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                if "output" in item:
                    grpo_item["reference"] = item["output"]
                grpo_data.append(grpo_item)

    if max_samples and max_samples > 0 and max_samples < len(grpo_data):
        total_count = len(grpo_data)
        if random_sample:
            random.seed(seed)
            grpo_data = random.sample(grpo_data, max_samples)
            print(
                f"Randomly sampled {max_samples} from {total_count} total samples (seed={seed})"
            )
        else:
            grpo_data = grpo_data[:max_samples]
            print(f"Selected first {max_samples} samples from {total_count} total")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grpo_data, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(grpo_data)} GRPO samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO training data")
    parser.add_argument(
        "--sft_train",
        type=str,
        default="data/openreview_dataset/sft_train.json",
        help="Path to SFT training data",
    )
    parser.add_argument(
        "--sft_val",
        type=str,
        default="data/openreview_dataset/sft_val.json",
        help="Path to SFT validation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/openreview_dataset",
        help="Output directory for GRPO data",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=3000,
        help="Maximum number of training samples (default: 3000, -1 for all)",
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

    prepare_grpo_data(
        args.sft_train,
        f"{args.output_dir}/grpo_train.json",
        args.max_train_samples,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    prepare_grpo_data(args.sft_val, f"{args.output_dir}/grpo_val.json")

    print("\nGRPO data preparation completed!")
    print(f"GRPO train data: {args.output_dir}/grpo_train.json")
    print(f"GRPO val data: {args.output_dir}/grpo_val.json")


if __name__ == "__main__":
    main()
