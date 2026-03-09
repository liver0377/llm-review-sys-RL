#!/usr/bin/env python3
"""
Convert DPO data to RM format with FULL paper content.
Strategy: Include complete paper content in query for maximum context.
"""

import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
import random

def convert_dpo_to_rm_full_content(
    dpo_train_path,
    dpo_val_path,
    output_dir,
    max_train_samples=None,
    max_val_samples=None,
    random_sample=False,
    seed=42
):
    """Convert DPO data to RM format with FULL paper content"""
    
    print("=" * 70)
    print("Converting DPO to RM format with FULL paper content")
    print("=" * 70)
    
    # Load training data
    print(f"\n[1/3] Loading training data from: {dpo_train_path}")
    with open(dpo_train_path, 'r') as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples")
    
    # Load validation data
    print(f"\n[2/3] Loading validation data from: {dpo_val_path}")
    with open(dpo_val_path, 'r') as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples")
    
    # Sample if requested
    if random_sample:
        random.seed(seed)
        if max_train_samples and len(train_data) > max_train_samples:
            train_data = random.sample(train_data, max_train_samples)
            print(f"Randomly sampled {len(train_data)} training samples")
        if max_val_samples and len(val_data) > max_val_samples:
            val_data = random.sample(val_data, max_val_samples)
            print(f"Randomly sampled {len(val_data)} validation samples")
    else:
        if max_train_samples:
            train_data = train_data[:max_train_samples]
            print(f"Using first {len(train_data)} training samples")
        if max_val_samples:
            val_data = val_data[:max_val_samples]
            print(f"Using first {len(val_data)} validation samples")
    
    # Convert training data
    print(f"\n[3/3] Converting data...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rm_train_data = []
    
    for item in tqdm(train_data, desc="Converting train"):
        prompt = item.get('prompt', '')
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        
        if not all([prompt, chosen, rejected]):
            continue
        
        # Check if prompt has paper details
        if 'Paper Details:' not in prompt:
            continue
        
        # Split instruction and paper details
        instruction = prompt.split('Paper Details:')[0].strip()
        paper_section = prompt.split('Paper Details:')[-1]
        
        # Extract title
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', paper_section)
        paper_title = title_match.group(1).strip() if title_match else ''
        
        # Extract content
        content_match = re.search(r'Content:\s*(.+)', paper_section, re.DOTALL)
        paper_content = content_match.group(1).strip() if content_match else ''
        
        if not paper_content:
            continue
        
        # Create query with FULL paper content
        query = f"{instruction}\n\nPaper Title: {paper_title}\n\nPaper Content:\n{paper_content}"
        
        rm_item = {
            "query": query,
            "chosen": chosen,
            "rejected": rejected
        }
        rm_train_data.append(rm_item)
    
    # Convert validation data
    rm_val_data = []
    
    for item in tqdm(val_data, desc="Converting val"):
        prompt = item.get('prompt', '')
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        
        if not all([prompt, chosen, rejected]):
            continue
        
        if 'Paper Details:' not in prompt:
            continue
        
        instruction = prompt.split('Paper Details:')[0].strip()
        paper_section = prompt.split('Paper Details:')[-1]
        
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', paper_section)
        paper_title = title_match.group(1).strip() if title_match else ''
        
        content_match = re.search(r'Content:\s*(.+)', paper_section, re.DOTALL)
        paper_content = content_match.group(1).strip() if content_match else ''
        
        if not paper_content:
            continue
        
        query = f"{instruction}\n\nPaper Title: {paper_title}\n\nPaper Content:\n{paper_content}"
        
        rm_item = {
            "query": query,
            "chosen": chosen,
            "rejected": rejected
        }
        rm_val_data.append(rm_item)
    
    # Save
    train_output = output_dir / "rm_train.json"
    val_output = output_dir / "rm_val.json"
    
    with open(train_output, 'w') as f:
        json.dump(rm_train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_output, 'w') as f:
        json.dump(rm_val_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"\nTraining data: {len(rm_train_data)} samples")
    print(f"Validation data: {len(rm_val_data)} samples")
    print(f"\nOutput files:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print("\nRecommended training config:")
    print("  max_length=16384  (covers 99.1% samples, P99=15397)")
    print("  or max_length=20000 (covers ~99.5% samples)")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DPO to RM format with full paper content")
    parser.add_argument('--dpo_train', type=str, required=True,
                       help='Path to DPO training data')
    parser.add_argument('--dpo_val', type=str, required=True,
                       help='Path to DPO validation data')
    parser.add_argument('--output_dir', type=str, default='data/openreview_dataset',
                       help='Output directory')
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Maximum training samples')
    parser.add_argument('--max_val_samples', type=int, default=None,
                       help='Maximum validation samples')
    parser.add_argument('--random_sample', action='store_true',
                       help='Random sample instead of taking first N')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    convert_dpo_to_rm_full_content(
        dpo_train_path=args.dpo_train,
        dpo_val_path=args.dpo_val,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        random_sample=args.random_sample,
        seed=args.seed
    )