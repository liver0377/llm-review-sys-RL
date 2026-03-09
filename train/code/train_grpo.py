import os
import sys
import yaml
import argparse
from pathlib import Path


def train_grpo(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["model"]["model_id_or_path"]
    train_dataset = config["dataset"]["train_dataset"]
    val_dataset = config["dataset"]["val_dataset"]
    output_dir = config["training"]["output_dir"]
    deepspeed_config = config["training"]["deepspeed"]

    plugin_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(plugin_dir))

    cmd_parts = [
        "NPROC_PER_NODE=8",
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "",
        "swift rlhf",
        f"    --rlhf_type {config['rlhf']['rlhf_type']}",
        f"    --model {model_path}",
        f"    --dataset {train_dataset}",
        f"    --val_dataset {val_dataset}",
        f"    --output_dir {output_dir}",
        f"    --tuner_type {config['training']['tuner_type']}",
        f"    --torch_dtype {config['training']['torch_dtype']}",
        f"    --max_length {config['dataset']['max_length']}",
        f"    --max_new_tokens {config['dataset']['max_new_tokens']}",
        f"    --per_device_train_batch_size {config['training']['per_device_train_batch_size']}",
        f"    --per_device_eval_batch_size {config['training']['per_device_eval_batch_size']}",
        f"    --gradient_accumulation_steps {config['training']['gradient_accumulation_steps']}",
        f"    --learning_rate {config['training']['learning_rate']}",
        f"    --weight_decay {config['training']['weight_decay']}",
        f"    --warmup_ratio {config['training']['warmup_ratio']}",
        f"    --num_train_epochs {config['training']['num_train_epochs']}",
        f"    --gradient_checkpointing {str(config['training']['gradient_checkpointing']).lower()}",
        f"    --bf16 {str(config['training']['bf16']).lower()}",
        f"    --eval_strategy {config['training']['eval_strategy']}",
        f"    --eval_steps {config['training']['eval_steps']}",
        f"    --save_steps {config['training']['save_steps']}",
        f"    --save_total_limit {config['training']['save_total_limit']}",
        f"    --logging_steps {config['training']['logging_steps']}",
        f"    --deepspeed {deepspeed_config}",
        f"    --num_generations {config['rlhf']['num_generations']}",
        f"    --temperature {config['rlhf']['temperature']}",
        f"    --top_p {config['rlhf']['top_p']}",
        f"    --beta {config['rlhf']['beta']}",
        f"    --alpha {config['reward']['alpha']}",
        f"    --use_vllm {str(config['vllm']['use_vllm']).lower()}",
        f"    --vllm_mode {config['vllm']['vllm_mode']}",
        f"    --vllm_gpu_memory_utilization {config['vllm']['vllm_gpu_memory_utilization']}",
        f"    --vllm_max_model_len {config['vllm']['vllm_max_model_len']}",
        f"    --vllm_enforce_eager {str(config['vllm']['vllm_enforce_eager']).lower()}",
        f"    --offload_optimizer {str(config['optimization']['offload_optimizer']).lower()}",
        f"    --offload_model {str(config['optimization']['offload_model']).lower()}",
        f"    --sleep_level {config['optimization']['sleep_level']}",
        f"    --report_to wandb",
        f"    --wandb_project {config['wandb']['project']}",
        f"    --wandb_run_name {config['wandb']['run_name']}",
        f"    --external_plugins train.code.rm_plugin:get_rm_plugin",
        f"    --reward_model_plugin {config['reward']['reward_model_plugin']}",
    ]

    cmd = " \\\n".join(cmd_parts)

    print("=" * 80)
    print("Training GRPO")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Num generations: {config['rlhf']['num_generations']}")
    print(f"Alpha (format/RM balance): {config['reward']['alpha']}")
    print("=" * 80)

    os.system(cmd.replace("\n\n", "\n"))


def main():
    parser = argparse.ArgumentParser(description="Train GRPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    train_grpo(args.config)


if __name__ == "__main__":
    main()
