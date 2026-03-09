#!/bin/bash

# 测试 RM 训练（使用自定义数据集配置）

NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type rm \
    --model models/qwen3-8b-sft \
    --dataset data/openreview_dataset/rm_train.json \
    --val_dataset data/openreview_dataset/rm_val.json \
    --custom_dataset_info configs/custom_dataset_info.json \
    --output_dir /tmp/test_rm_output \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --max_steps 5 \
    --gradient_checkpointing true \
    --bf16 true \
    --logging_steps 1

echo "测试完成！"
