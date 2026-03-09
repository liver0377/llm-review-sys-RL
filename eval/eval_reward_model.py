import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from reward_model import RewardModel


class RewardEvalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=16384):
        print(f"Loading evaluation data from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_text = item["chosen_text"]
        rejected_text = item["rejected_text"]
        return chosen_text, rejected_text


def collate_fn(batch, tokenizer, max_length):
    chosen_texts = [item[0] for item in batch]
    rejected_texts = [item[1] for item in batch]

    chosen_inputs = tokenizer(
        chosen_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    rejected_inputs = tokenizer(
        rejected_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    return {
        "chosen_inputs": chosen_inputs,
        "rejected_inputs": rejected_inputs,
    }


def compute_metrics(chosen_rewards, rejected_rewards):
    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
    margin = (chosen_rewards - rejected_rewards).mean().item()
    margin_std = (chosen_rewards - rejected_rewards).std().item()

    correct_pairs = (chosen_rewards > rejected_rewards).sum().item()
    total_pairs = len(chosen_rewards)

    metrics = {
        "accuracy": accuracy,
        "margin": margin,
        "margin_std": margin_std,
        "correct_pairs": correct_pairs,
        "total_pairs": total_pairs,
    }
    return metrics


def evaluate_reward_model(model, dataloader, device):
    model.eval()
    all_chosen_rewards = []
    all_rejected_rewards = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            chosen_inputs = {k: v.to(device) for k, v in batch["chosen_inputs"].items()}
            rejected_inputs = {
                k: v.to(device) for k, v in batch["rejected_inputs"].items()
            }

            chosen_rewards = model(**chosen_inputs).cpu()
            rejected_rewards = model(**rejected_inputs).cpu()

            all_chosen_rewards.append(chosen_rewards)
            all_rejected_rewards.append(rejected_rewards)

    all_chosen_rewards = torch.cat(all_chosen_rewards)
    all_rejected_rewards = torch.cat(all_rejected_rewards)

    metrics = compute_metrics(all_chosen_rewards, all_rejected_rewards)
    return metrics, all_chosen_rewards, all_rejected_rewards


def print_metrics(metrics):
    print("\n" + "=" * 50)
    print("Reward Model Evaluation Results")
    print("=" * 50)
    print(
        f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_pairs']}/{metrics['total_pairs']})"
    )
    print(f"Mean Margin: {metrics['margin']:.4f} ± {metrics['margin_std']:.4f}")
    print("=" * 50 + "\n")


def save_results(metrics, chosen_rewards, rejected_rewards, output_path):
    results = {
        "metrics": metrics,
        "chosen_rewards": chosen_rewards.tolist(),
        "rejected_rewards": rejected_rewards.tolist(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reward Model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained reward model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data (RM format)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="pretrained/Qwen/Qwen3-8B",
        help="Base model path",
    )
    parser.add_argument(
        "--max_length", type=int, default=16384, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for results"
    )
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading reward model...")
    model = RewardModel.from_pretrained(args.model_path, dtype=torch.bfloat16)
    device = torch.device("cuda")
    model.to(device)

    print("Loading evaluation dataset...")
    dataset = RewardEvalDataset(args.data_path, tokenizer, args.max_length)

    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer, args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=4,
    )

    print("Starting evaluation...")
    metrics, chosen_rewards, rejected_rewards = evaluate_reward_model(
        model, dataloader, device
    )

    print_metrics(metrics)

    if args.output:
        save_results(metrics, chosen_rewards, rejected_rewards, args.output)


if __name__ == "__main__":
    main()
