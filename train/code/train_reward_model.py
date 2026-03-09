import os
import sys
import yaml
import argparse
from pathlib import Path


def train_reward_model(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["model"]["model_id_or_path"]
    train_dataset = config["dataset"]["train_dataset"]
    val_dataset = config["dataset"]["val_dataset"]
    output_dir = config["training"]["output_dir"]
    deepspeed_config = config["training"]["deepspeed"]

    cmd = f"""
NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

swift rlhf \\
    --rlhf_type rm \\
    --model {model_path} \\
    --dataset {train_dataset} \\
    --val_dataset {val_dataset} \\
    --output_dir {output_dir} \\
    --tuner_type {config["training"]["tuner_type"]} \\
    --torch_dtype {config["training"]["torch_dtype"]} \\
    --max_length {config["dataset"]["max_length"]} \\
    --per_device_train_batch_size {config["training"]["per_device_train_batch_size"]} \\
    --per_device_eval_batch_size {config["training"]["per_device_eval_batch_size"]} \\
    --gradient_accumulation_steps {config["training"]["gradient_accumulation_steps"]} \\
    --learning_rate {config["training"]["learning_rate"]} \\
    --weight_decay {config["training"]["weight_decay"]} \\
    --warmup_ratio {config["training"]["warmup_ratio"]} \\
    --num_train_epochs {config["training"]["num_train_epochs"]} \\
    --gradient_checkpointing {str(config["training"]["gradient_checkpointing"]).lower()} \\
    --bf16 {str(config["training"]["bf16"]).lower()} \\
    --eval_strategy {config["training"]["eval_strategy"]} \\
    --eval_steps {config["training"]["eval_steps"]} \\
    --save_steps {config["training"]["save_steps"]} \\
    --save_total_limit {config["training"]["save_total_limit"]} \\
    --logging_steps {config["training"]["logging_steps"]} \\
    --deepspeed {deepspeed_config} \\
    --beta {config["rlhf"]["beta"]} \\
    --report_to wandb \\
    --wandb_project {config["wandb"]["project"]} \\
    --wandb_run_name {config["wandb"]["run_name"]}
""".strip()

    print("=" * 80)
    print("Training Reward Model")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    os.system(cmd.replace("\n", " "))


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reward_model_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    train_reward_model(args.config)


if __name__ == "__main__":
    main()
