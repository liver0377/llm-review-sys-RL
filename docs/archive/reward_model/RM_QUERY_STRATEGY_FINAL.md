# Reward Model Query Strategy - Final Decision

## Date: 2026-03-06

## Decision: **指令 + 标题 + 完整论文**

### Rationale

After comprehensive token distribution analysis, we decided to include **FULL paper content** in the query for the following reasons:

1. **Maximum Context**: Provides complete paper content for RM to detect:
   - Fabricated claims (捏造内容)
   - Missed critical flaws (未指出真实缺陷)
   - Inaccurate assessments (不准确评价)

2. **Hardware Capacity**: 8x A100 80GB + DeepSpeed ZeRO-3 can handle large sequences

3. **Token Distribution**:
   - Q+C Mean: 9208 tokens
   - Q+C P95: 12582 tokens
   - Q+C P99: 15168 tokens
   - Q+C Max: 25901 tokens

4. **Coverage with max_length=16384**:
   - **99.4% samples covered** (only 0.6% exceed)
   - P99 (15168) well within limit
   - Excellent trade-off between context and efficiency

---

## Comparison with Other Strategies

| Strategy | Q+C Mean | Q+C P99 | max_length=2048 | max_length=4096 | max_length=16384 |
|----------|----------|---------|-----------------|-----------------|------------------|
| 仅指令 | 816 | 1263 | 0% 超出 | 0% 超出 | 0% 超出 |
| +标题+Abstract | 1084 | 1561 | 0% 超出 | 0% 超出 | 0% 超出 |
| +标题+Content(1k) | 1838 | 2283 | 10.6% 超出 | 0% 超出 | 0% 超出 |
| +标题+Content(3k) | 3832 | 4283 | 100% 超出 | 6.8% 超出 | 0% 超出 |
| **+标题+完整论文** | **9208** | **15168** | **100% 超出** | **98.8% 超出** | **0.6% 超出** |

**Conclusion**: Full paper content with max_length=16384 provides the best balance of context vs. efficiency.

---

## Training Configuration

### Data
- **Training samples**: 5,000
- **Validation samples**: 1,052
- **Source**: `dpo_vllm_as_rejected_train_cleaned.json`
- **Format**: `{query, chosen, rejected}`
- **Query format**: 
  ```
  {instruction}
  
  Paper Title: {title}
  
  Paper Content:
  {full_paper_content}
  ```

### Model
- **Base model**: `models/qwen3-8b-base`
- **Type**: Reward Model (full fine-tuning)
- **Dtype**: bfloat16

### Training Parameters
```bash
max_length=16384
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=1e-5
weight_decay=0.01
warmup_ratio=0.1
num_train_epochs=2
gradient_checkpointing=true
bf16=true
deepspeed=configs/deepspeed_zero2_config.json
```

### Hardware
- **GPUs**: 8x A100 80GB
- **Distributed**: DeepSpeed ZeRO-60
- **Estimated memory per GPU**: 60GB (should fit comfortably)

---

## Training Script

```bash
bash scripts/train_rm_full_content.sh
```

Or run directly:
```bash
WANDB_PROJECT=reward_model_grpo \
swift rlhf \
    --rlhf_type rm \
    --model models/qwen3-8b-base \
    --dataset data/openreview_dataset/rm_train.json \
    --val_dataset data/openreview_dataset/rm_val.json \
    --output_dir models/reward_model_qwen3_8b \
    --max_length 16384 \
    ... (see scripts/train_rm_full_content.sh for full config)
```

---

## Expected Training Time

Based on token counts and hardware:
- **Samples**: 5,000
- **Effective batch size**: 1 × 8 GPUs × 8 grad_accum = 64
- **Steps per epoch**: 5000 / 64 ≈ 78 steps
- **Total steps**: 78 × 2 epochs = 156 steps
- **Estimated time**: 2-4 hours (depending on sequence lengths)

---

## Files

### Data Files
- `data/openreview_dataset/rm_train.json` - Training data (5000 samples)
- `data/openreview_dataset/rm_val.json` - Validation data (1052 samples)

### Scripts
- `scripts/convert_dpo_to_rm_full_content.py` - Data conversion script
- `scripts/train_rm_full_content.sh` - Training launch script
- `scripts/analyze_rm_token_distribution.py` - Token analysis script
- `scripts/analyze_query_token_distribution_comprehensive.py` - Comprehensive analysis

### Documentation
- `docs/rm_token_stats.txt` - Token statistics
- `docs/rm_token_distribution.png` - Token distribution plots
- `docs/rm_query_token_distribution_comprehensive.txt` - Full analysis results
- `docs/rm_query_token_distribution_comprehensive.png` - Full analysis plots

---

## Next Steps

1. ✅ Data prepared
2. ✅ Training script ready
3. ⏳ Start training: `bash scripts/train_rm_full_content.sh`
4. ⏳ Monitor training (WandB)
5. ⏳ Evaluate reward model
6. ⏳ Proceed to GRPO training

---

## Notes

### Why not max_length=32768?
- Only 0.6% samples exceed 16384
- Doubling max_length would increase memory ~2x
- Not worth the cost for 0.6% more samples

### Why full paper instead of abstract?
- Abstract: ~500 tokens average, covers only high-level claims
- Full paper: ~8600 tokens average, covers all details
- RM can detect specific inaccuracies that abstract wouldn't reveal
- Example: "The reviewer claims method X doesn't work, but section 3.2 shows it achieves 95% accuracy"

### Risk mitigation
- 0.6% samples will be truncated at max_length=16384
- These are exceptionally long papers (>25k tokens)
- Truncation is acceptable as most papers are well within limit
