#!/usr/bin/env python3
"""
Analyze token distribution for reward model training data.
This script tokenizes query+chosen and query+rejected pairs and plots the distribution.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    print("=" * 60)
    print("RM Token Distribution Analysis")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    model_path = "models/qwen3-8b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    # Load data
    print("\n[2/4] Loading training data...")
    data_path = "data/openreview_dataset/rm_train.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Analyze token lengths
    print("\n[3/4] Tokenizing samples...")
    query_lengths = []
    chosen_lengths = []
    rejected_lengths = []
    chosen_total_lengths = []
    rejected_total_lengths = []
    
    for item in tqdm(data, desc="Processing"):
        # Tokenize query
        query_tokens = tokenizer.encode(item['query'], add_special_tokens=False)
        query_len = len(query_tokens)
        query_lengths.append(query_len)
        
        # Tokenize chosen response
        chosen_tokens = tokenizer.encode(item['chosen'], add_special_tokens=False)
        chosen_len = len(chosen_tokens)
        chosen_lengths.append(chosen_len)
        
        # Tokenize rejected response
        rejected_tokens = tokenizer.encode(item['rejected'], add_special_tokens=False)
        rejected_len = len(rejected_tokens)
        rejected_lengths.append(rejected_len)
        
        # Total lengths (query + response)
        # For RM training, the format is typically: query + response
        # Swift will add special tokens automatically
        chosen_total_lengths.append(query_len + chosen_len)
        rejected_total_lengths.append(query_len + rejected_len)
    
    # Convert to numpy arrays
    query_lengths = np.array(query_lengths)
    chosen_lengths = np.array(chosen_lengths)
    rejected_lengths = np.array(rejected_lengths)
    chosen_total_lengths = np.array(chosen_total_lengths)
    rejected_total_lengths = np.array(rejected_total_lengths)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    
    print("\n[Query Length]")
    print(f"  Mean: {query_lengths.mean():.1f}")
    print(f"  Std:  {query_lengths.std():.1f}")
    print(f"  Min:  {query_lengths.min()}")
    print(f"  Max:  {query_lengths.max()}")
    print(f"  Median: {np.median(query_lengths):.1f}")
    print(f"  P95:  {np.percentile(query_lengths, 95):.1f}")
    print(f"  P99:  {np.percentile(query_lengths, 99):.1f}")
    
    print("\n[Chosen Response Length]")
    print(f"  Mean: {chosen_lengths.mean():.1f}")
    print(f"  Std:  {chosen_lengths.std():.1f}")
    print(f"  Min:  {chosen_lengths.min()}")
    print(f"  Max:  {chosen_lengths.max()}")
    print(f"  Median: {np.median(chosen_lengths):.1f}")
    print(f"  P95:  {np.percentile(chosen_lengths, 95):.1f}")
    print(f"  P99:  {np.percentile(chosen_lengths, 99):.1f}")
    
    print("\n[Rejected Response Length]")
    print(f"  Mean: {rejected_lengths.mean():.1f}")
    print(f"  Std:  {rejected_lengths.std():.1f}")
    print(f"  Min:  {rejected_lengths.min()}")
    print(f"  Max:  {rejected_lengths.max()}")
    print(f"  Median: {np.median(rejected_lengths):.1f}")
    print(f"  P95:  {np.percentile(rejected_lengths, 95):.1f}")
    print(f"  P99:  {np.percentile(rejected_lengths, 99):.1f}")
    
    print("\n[Total Length: Query + Chosen]")
    print(f"  Mean: {chosen_total_lengths.mean():.1f}")
    print(f"  Std:  {chosen_total_lengths.std():.1f}")
    print(f"  Min:  {chosen_total_lengths.min()}")
    print(f"  Max:  {chosen_total_lengths.max()}")
    print(f"  Median: {np.median(chosen_total_lengths):.1f}")
    print(f"  P95:  {np.percentile(chosen_total_lengths, 95):.1f}")
    print(f"  P99:  {np.percentile(chosen_total_lengths, 99):.1f}")
    
    print("\n[Total Length: Query + Rejected]")
    print(f"  Mean: {rejected_total_lengths.mean():.1f}")
    print(f"  Std:  {rejected_total_lengths.std():.1f}")
    print(f"  Min:  {rejected_total_lengths.min()}")
    print(f"  Max:  {rejected_total_lengths.max()}")
    print(f"  Median: {np.median(rejected_total_lengths):.1f}")
    print(f"  P95:  {np.percentile(rejected_total_lengths, 95):.1f}")
    print(f"  P99:  {np.percentile(rejected_total_lengths, 99):.1f}")
    
    # Count samples that exceed common max_length values
    print("\n[Samples Exceeding Max Length]")
    for max_len in [4096, 8192, 16384]:
        chosen_exceed = (chosen_total_lengths > max_len).sum()
        rejected_exceed = (rejected_total_lengths > max_len).sum()
        print(f"  max_length={max_len}: {chosen_exceed} chosen ({100*chosen_exceed/len(data):.1f}%), {rejected_exceed} rejected ({100*rejected_exceed/len(data):.1f}%)")
    
    # Create plots
    print("\n[4/4] Creating distribution plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Reward Model Training Data Token Distribution', fontsize=16, fontweight='bold')
    
    # Plot 1: Query length distribution
    ax = axes[0, 0]
    ax.hist(query_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(query_lengths), color='red', linestyle='--', label=f'Median: {np.median(query_lengths):.0f}')
    ax.axvline(np.percentile(query_lengths, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(query_lengths, 95):.0f}')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Query Length Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Chosen response length distribution
    ax = axes[0, 1]
    ax.hist(chosen_lengths, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(chosen_lengths), color='red', linestyle='--', label=f'Median: {np.median(chosen_lengths):.0f}')
    ax.axvline(np.percentile(chosen_lengths, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(chosen_lengths, 95):.0f}')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Chosen Response Length Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Rejected response length distribution
    ax = axes[0, 2]
    ax.hist(rejected_lengths, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(rejected_lengths), color='red', linestyle='--', label=f'Median: {np.median(rejected_lengths):.0f}')
    ax.axvline(np.percentile(rejected_lengths, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(rejected_lengths, 95):.0f}')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Rejected Response Length Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Query + Chosen total length
    ax = axes[1, 0]
    ax.hist(chosen_total_lengths, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(chosen_total_lengths), color='red', linestyle='--', label=f'Median: {np.median(chosen_total_lengths):.0f}')
    ax.axvline(np.percentile(chosen_total_lengths, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(chosen_total_lengths, 95):.0f}')
    ax.axvline(4096, color='green', linestyle=':', linewidth=2, label='max_length=4096')
    ax.axvline(8192, color='blue', linestyle=':', linewidth=2, label='max_length=8192')
    ax.axvline(16384, color='purple', linestyle=':', linewidth=2, label='max_length=16384')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Length (Query + Chosen) Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 5: Query + Rejected total length
    ax = axes[1, 1]
    ax.hist(rejected_total_lengths, bins=50, color='gold', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(rejected_total_lengths), color='red', linestyle='--', label=f'Median: {np.median(rejected_total_lengths):.0f}')
    ax.axvline(np.percentile(rejected_total_lengths, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(rejected_total_lengths, 95):.0f}')
    ax.axvline(4096, color='green', linestyle=':', linewidth=2, label='max_length=4096')
    ax.axvline(8192, color='blue', linestyle=':', linewidth=2, label='max_length=8192')
    ax.axvline(16384, color='purple', linestyle=':', linewidth=2, label='max_length=16384')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Length (Query + Rejected) Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 6: Combined box plot comparison
    ax = axes[1, 2]
    box_data = [query_lengths, chosen_lengths, rejected_lengths, chosen_total_lengths, rejected_total_lengths]
    box_labels = ['Query', 'Chosen', 'Rejected', 'Query+Chosen', 'Query+Rejected']
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'mediumpurple', 'gold']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Token Count')
    ax.set_title('Length Comparison (Box Plot)')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'docs/rm_token_distribution.png'
    os.makedirs('docs', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Save statistics to file
    stats_path = 'docs/rm_token_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Reward Model Training Data Token Statistics\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total samples: {len(data)}\n\n")
        
        f.write("[Query Length]\n")
        f.write(f"  Mean: {query_lengths.mean():.1f}\n")
        f.write(f"  Std:  {query_lengths.std():.1f}\n")
        f.write(f"  Min:  {query_lengths.min()}\n")
        f.write(f"  Max:  {query_lengths.max()}\n")
        f.write(f"  Median: {np.median(query_lengths):.1f}\n")
        f.write(f"  P95:  {np.percentile(query_lengths, 95):.1f}\n")
        f.write(f"  P99:  {np.percentile(query_lengths, 99):.1f}\n\n")
        
        f.write("[Chosen Response Length]\n")
        f.write(f"  Mean: {chosen_lengths.mean():.1f}\n")
        f.write(f"  Std:  {chosen_lengths.std():.1f}\n")
        f.write(f"  Min:  {chosen_lengths.min()}\n")
        f.write(f"  Max:  {chosen_lengths.max()}\n")
        f.write(f"  Median: {np.median(chosen_lengths):.1f}\n")
        f.write(f"  P95:  {np.percentile(chosen_lengths, 95):.1f}\n")
        f.write(f"  P99:  {np.percentile(chosen_lengths, 99):.1f}\n\n")
        
        f.write("[Rejected Response Length]\n")
        f.write(f"  Mean: {rejected_lengths.mean():.1f}\n")
        f.write(f"  Std:  {rejected_lengths.std():.1f}\n")
        f.write(f"  Min:  {rejected_lengths.min()}\n")
        f.write(f"  Max:  {rejected_lengths.max()}\n")
        f.write(f"  Median: {np.median(rejected_lengths):.1f}\n")
        f.write(f"  P95:  {np.percentile(rejected_lengths, 95):.1f}\n")
        f.write(f"  P99:  {np.percentile(rejected_lengths, 99):.1f}\n\n")
        
        f.write("[Total Length: Query + Chosen]\n")
        f.write(f"  Mean: {chosen_total_lengths.mean():.1f}\n")
        f.write(f"  Std:  {chosen_total_lengths.std():.1f}\n")
        f.write(f"  Min:  {chosen_total_lengths.min()}\n")
        f.write(f"  Max:  {chosen_total_lengths.max()}\n")
        f.write(f"  Median: {np.median(chosen_total_lengths):.1f}\n")
        f.write(f"  P95:  {np.percentile(chosen_total_lengths, 95):.1f}\n")
        f.write(f"  P99:  {np.percentile(chosen_total_lengths, 99):.1f}\n\n")
        
        f.write("[Total Length: Query + Rejected]\n")
        f.write(f"  Mean: {rejected_total_lengths.mean():.1f}\n")
        f.write(f"  Std:  {rejected_total_lengths.std():.1f}\n")
        f.write(f"  Min:  {rejected_total_lengths.min()}\n")
        f.write(f"  Max:  {rejected_total_lengths.max()}\n")
        f.write(f"  Median: {np.median(rejected_total_lengths):.1f}\n")
        f.write(f"  P95:  {np.percentile(rejected_total_lengths, 95):.1f}\n")
        f.write(f"  P99:  {np.percentile(rejected_total_lengths, 99):.1f}\n\n")
        
        f.write("[Samples Exceeding Max Length]\n")
        for max_len in [4096, 8192, 16384]:
            chosen_exceed = (chosen_total_lengths > max_len).sum()
            rejected_exceed = (rejected_total_lengths > max_len).sum()
            f.write(f"  max_length={max_len}: {chosen_exceed} chosen ({100*chosen_exceed/len(data):.1f}%), {rejected_exceed} rejected ({100*rejected_exceed/len(data):.1f}%)\n")
    
    print(f"Statistics saved to: {stats_path}")
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()