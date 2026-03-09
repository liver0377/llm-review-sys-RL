#!/usr/bin/env python3
"""
Comprehensive token distribution analysis for RM query design.
Test different paper context inclusion strategies in token dimension.
"""

import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path

def extract_abstract(content):
    """Extract abstract from paper content"""
    patterns = [
        r'Abstract\s*\n(.*?)(?=\n\s*\n|\n#{1,3}\s|\n[A-Z][a-z]+\s*\n)',
        r'ABSTRACT\s*\n(.*?)(?=\n\s*\n|\n#{1,3}\s|\n[A-Z][a-z]+\s*\n)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:
                return abstract
    
    return None

def truncate_by_tokens(text, tokenizer, max_tokens):
    """Truncate text to max_tokens"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens
    truncated_tokens = tokens[:max_tokens]
    # Decode back to text
    return tokenizer.decode(truncated_tokens)

def create_query_variants(instruction, paper_title, paper_content, tokenizer):
    """Create different query variants with different paper context strategies"""
    variants = {}
    
    # Base instruction
    base_query = instruction.strip()
    
    # Variant 1: Only instruction (current baseline)
    variants['1_instruction_only'] = base_query
    
    # Variant 2: Instruction + Title
    variants['2_instruction_title'] = f"{base_query}\n\nPaper Title: {paper_title}"
    
    # Variant 3: Instruction + Title + Abstract
    abstract = extract_abstract(paper_content)
    if abstract:
        variants['3_instruction_title_abstract'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nAbstract:\n{abstract}"
    
    # Variant 4: Instruction + Title + Abstract (truncated to 300 tokens)
    if abstract:
        abstract_300 = truncate_by_tokens(abstract, tokenizer, 300)
        variants['4_instruction_title_abstract_300t'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nAbstract:\n{abstract_300}"
    
    # Variant 5: Instruction + Title + Abstract (truncated to 500 tokens)
    if abstract:
        abstract_500 = truncate_by_tokens(abstract, tokenizer, 500)
        variants['5_instruction_title_abstract_500t'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nAbstract:\n{abstract_500}"
    
    # Variant 6: Instruction + Title + Content (truncated to 500 tokens)
    content_500 = truncate_by_tokens(paper_content, tokenizer, 500)
    variants['6_instruction_title_content_500t'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nPaper Content:\n{content_500}"
    
    # Variant 7: Instruction + Title + Content (truncated to 1000 tokens)
    content_1000 = truncate_by_tokens(paper_content, tokenizer, 1000)
    variants['7_instruction_title_content_1000t'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nPaper Content:\n{content_1000}"
    
    # Variant 8: Instruction + Title + Content (truncated to 2000 tokens)
    content_2000 = truncate_by_tokens(paper_content, tokenizer, 2000)
    variants['8_instruction_title_content_2000t'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nPaper Content:\n{content_2000}"
    
    # Variant 9: Instruction + Title + Content (truncated to 3000 tokens)
    content_3000 = truncate_by_tokens(paper_content, tokenizer, 3000)
    variants['9_instruction_title_content_3000t'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nPaper Content:\n{content_3000}"
    
    # Variant 10: Instruction + Title + Full Content
    variants['10_instruction_title_full_content'] = f"{base_query}\n\nPaper Title: {paper_title}\n\nPaper Content:\n{paper_content}"
    
    # Tokenize all variants
    tokenized_variants = {}
    for name, query in variants.items():
        tokens = tokenizer.encode(query, add_special_tokens=False)
        tokenized_variants[name] = {
            'text': query,
            'token_count': len(tokens)
        }
    
    return tokenized_variants

def main():
    print("=" * 80)
    print("Comprehensive Token Distribution Analysis for RM Query Design")
    print("=" * 80)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    model_path = "models/qwen3-8b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Load data
    print("\n[2/4] Loading dataset...")
    data_path = "data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Sample for analysis
    sample_size = min(2000, len(data))
    sampled_indices = np.random.choice(len(data), sample_size, replace=False)
    print(f"Analyzing {sample_size} samples...")
    
    # Analyze
    print("\n[3/4] Analyzing token distributions...")
    
    # Collect token counts for each variant
    variant_token_counts = {}
    chosen_lengths = []
    rejected_lengths = []
    
    for idx in tqdm(sampled_indices, desc="Processing"):
        item = data[idx]
        prompt = item.get('prompt', '')
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        
        if not all([prompt, chosen, rejected]):
            continue
        
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
        
        # Create query variants
        variants = create_query_variants(instruction, paper_title, paper_content, tokenizer)
        
        # Collect token counts
        for variant_name, variant_data in variants.items():
            if variant_name not in variant_token_counts:
                variant_token_counts[variant_name] = []
            variant_token_counts[variant_name].append(variant_data['token_count'])
        
        # Collect chosen/rejected lengths
        chosen_tokens = tokenizer.encode(chosen, add_special_tokens=False)
        rejected_tokens = tokenizer.encode(rejected, add_special_tokens=False)
        chosen_lengths.append(len(chosen_tokens))
        rejected_lengths.append(len(rejected_tokens))
    
    # Convert to numpy arrays
    chosen_lengths = np.array(chosen_lengths)
    rejected_lengths = np.array(rejected_lengths)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Response Token Statistics")
    print("=" * 80)
    
    print("\n[Chosen Response]")
    print(f"  Mean: {chosen_lengths.mean():.1f} tokens")
    print(f"  Std:  {chosen_lengths.std():.1f} tokens")
    print(f"  Min:  {chosen_lengths.min()} tokens")
    print(f"  Max:  {chosen_lengths.max()} tokens")
    print(f"  Median: {np.median(chosen_lengths):.1f} tokens")
    print(f"  P95:  {np.percentile(chosen_lengths, 95):.1f} tokens")
    print(f"  P99:  {np.percentile(chosen_lengths, 99):.1f} tokens")
    
    print("\n[Rejected Response]")
    print(f"  Mean: {rejected_lengths.mean():.1f} tokens")
    print(f"  Std:  {rejected_lengths.std():.1f} tokens")
    print(f"  Min:  {rejected_lengths.min()} tokens")
    print(f"  Max:  {rejected_lengths.max()} tokens")
    print(f"  Median: {np.median(rejected_lengths):.1f} tokens")
    print(f"  P95:  {np.percentile(rejected_lengths, 95):.1f} tokens")
    print(f"  P99:  {np.percentile(rejected_lengths, 99):.1f} tokens")
    
    print("\n" + "=" * 80)
    print("Query Variant Token Statistics")
    print("=" * 80)
    
    variant_stats = []
    for variant_name in sorted(variant_token_counts.keys()):
        token_counts = np.array(variant_token_counts[variant_name])
        
        # Calculate total lengths (query + chosen, query + rejected)
        total_chosen = token_counts + chosen_lengths[:len(token_counts)]
        total_rejected = token_counts + rejected_lengths[:len(token_counts)]
        
        stat = {
            'name': variant_name,
            'query_mean': token_counts.mean(),
            'query_median': np.median(token_counts),
            'query_p95': np.percentile(token_counts, 95),
            'query_p99': np.percentile(token_counts, 99),
            'query_max': token_counts.max(),
            'total_chosen_mean': total_chosen.mean(),
            'total_chosen_p95': np.percentile(total_chosen, 95),
            'total_chosen_p99': np.percentile(total_chosen, 99),
            'total_chosen_max': total_chosen.max(),
            'total_rejected_mean': total_rejected.mean(),
            'total_rejected_p95': np.percentile(total_rejected, 95),
            'total_rejected_p99': np.percentile(total_rejected, 99),
            'total_rejected_max': total_rejected.max(),
        }
        variant_stats.append(stat)
        
        print(f"\n[{variant_name}]")
        print(f"  Query Mean: {stat['query_mean']:.1f} tokens")
        print(f"  Query P95: {stat['query_p95']:.1f} tokens")
        print(f"  Query P99: {stat['query_p99']:.1f} tokens")
        print(f"  Query Max: {stat['query_max']} tokens")
        print(f"  Total (Query+Chosen) Mean: {stat['total_chosen_mean']:.1f} tokens")
        print(f"  Total (Query+Chosen) P95: {stat['total_chosen_p95']:.1f} tokens")
        print(f"  Total (Query+Chosen) P99: {stat['total_chosen_p99']:.1f} tokens")
        print(f"  Total (Query+Chosen) Max: {stat['total_chosen_max']} tokens")
        print(f"  Total (Query+Rejected) Mean: {stat['total_rejected_mean']:.1f} tokens")
        print(f"  Total (Query+Rejected) P95: {stat['total_rejected_p95']:.1f} tokens")
    
    # Count samples exceeding different max_length values
    print("\n" + "=" * 80)
    print("Samples Exceeding Max Length")
    print("=" * 80)
    
    for max_len in [2048, 4096, 8192, 16384]:
        print(f"\nmax_length = {max_len}:")
        for stat in variant_stats:
            total_chosen = np.array(variant_token_counts[stat['name']]) + chosen_lengths[:len(variant_token_counts[stat['name']])]
            exceed_chosen = (total_chosen > max_len).sum()
            pct_chosen = 100 * exceed_chosen / len(total_chosen)
            
            total_rejected = np.array(variant_token_counts[stat['name']]) + rejected_lengths[:len(variant_token_counts[stat['name']])]
            exceed_rejected = (total_rejected > max_len).sum()
            pct_rejected = 100 * exceed_rejected / len(total_rejected)
            
            print(f"  {stat['name']}: {exceed_chosen} chosen ({pct_chosen:.1f}%), {exceed_rejected} rejected ({pct_rejected:.1f}%)")
    
    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('RM Query Token Distribution - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Query token distribution for all variants
    ax = axes[0, 0]
    variant_names = [s['name'] for s in variant_stats]
    query_means = [s['query_mean'] for s in variant_stats]
    query_p95s = [s['query_p95'] for s in variant_stats]
    
    x = np.arange(len(variant_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, query_means, width, label='Mean', color='skyblue')
    bars2 = ax.bar(x + width/2, query_p95s, width, label='P95', color='lightcoral')
    
    ax.set_ylabel('Token Count')
    ax.set_title('Query Token Count by Variant')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in variant_names], fontsize=7, rotation=0)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Total (Query+Chosen) token distribution
    ax = axes[0, 1]
    total_chosen_means = [s['total_chosen_mean'] for s in variant_stats]
    total_chosen_p95s = [s['total_chosen_p95'] for s in variant_stats]
    
    bars1 = ax.bar(x - width/2, total_chosen_means, width, label='Mean', color='lightgreen')
    bars2 = ax.bar(x + width/2, total_chosen_p95s, width, label='P95', color='orange')
    
    ax.set_ylabel('Token Count')
    ax.set_title('Total (Query+Chosen) Token Count by Variant')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in variant_names], fontsize=7, rotation=0)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add horizontal lines for common max_length values
    for max_len in [2048, 4096]:
        ax.axhline(y=max_len, color='red', linestyle='--', alpha=0.5, label=f'max_length={max_len}')
    ax.legend(fontsize=7)
    
    # Plot 3: Total (Query+Rejected) token distribution
    ax = axes[0, 2]
    total_rejected_means = [s['total_rejected_mean'] for s in variant_stats]
    total_rejected_p95s = [s['total_rejected_p95'] for s in variant_stats]
    
    bars1 = ax.bar(x - width/2, total_rejected_means, width, label='Mean', color='mediumpurple')
    bars2 = ax.bar(x + width/2, total_rejected_p95s, width, label='P95', color='gold')
    
    ax.set_ylabel('Token Count')
    ax.set_title('Total (Query+Rejected) Token Count by Variant')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in variant_names], fontsize=7, rotation=0)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    for max_len in [2048, 4096]:
        ax.axhline(y=max_len, color='red', linestyle='--', alpha=0.5)
    
    # Plot 4: Histogram of query tokens for selected variants
    ax = axes[1, 0]
    selected_variants = ['1_instruction_only', '3_instruction_title_abstract', '9_instruction_title_content_3000t', '10_instruction_title_full_content']
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
    
    for variant_name, color in zip(selected_variants, colors):
        if variant_name in variant_token_counts:
            ax.hist(variant_token_counts[variant_name], bins=30, alpha=0.5, label=variant_name.replace('_', ' '), color=color, edgecolor='black')
    
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Query Token Distribution (Selected Variants)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 5: Histogram of total (query+chosen) tokens for selected variants
    ax = axes[1, 1]
    
    for variant_name, color in zip(selected_variants, colors):
        if variant_name in variant_token_counts:
            total = np.array(variant_token_counts[variant_name]) + chosen_lengths[:len(variant_token_counts[variant_name])]
            ax.hist(total, bins=30, alpha=0.5, label=variant_name.replace('_', ' '), color=color, edgecolor='black')
    
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Total (Query+Chosen) Token Distribution (Selected Variants)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Add max_length lines
    for max_len in [2048, 4096]:
        ax.axvline(x=max_len, color='red', linestyle='--', alpha=0.5, label=f'max_length={max_len}')
    ax.legend(fontsize=8)
    
    # Plot 6: Percentage of samples exceeding max_length
    ax = axes[1, 2]
    max_lengths = [2048, 4096, 8192, 16384]
    
    for i, max_len in enumerate(max_lengths):
        exceed_pcts = []
        for stat in variant_stats:
            total_chosen = np.array(variant_token_counts[stat['name']]) + chosen_lengths[:len(variant_token_counts[stat['name']])]
            exceed_pct = 100 * (total_chosen > max_len).sum() / len(total_chosen)
            exceed_pcts.append(exceed_pct)
        
        x_offset = (i - 1.5) * 0.2
        ax.bar(np.arange(len(variant_names)) + x_offset, exceed_pcts, width=0.2, label=f'max_length={max_len}')
    
    ax.set_ylabel('% Samples Exceeding')
    ax.set_title('Percentage of Samples Exceeding Max Length')
    ax.set_xticks(np.arange(len(variant_names)))
    ax.set_xticklabels([name.replace('_', '\n') for name in variant_names], fontsize=7, rotation=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'docs/rm_query_token_distribution_comprehensive.png'
    os.makedirs('docs', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Save detailed results
    results_path = 'docs/rm_query_token_distribution_comprehensive.txt'
    with open(results_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RM Query Token Distribution - Comprehensive Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total samples analyzed: {len(chosen_lengths)}\n\n")
        
        f.write("[Response Statistics]\n")
        f.write(f"Chosen Mean: {chosen_lengths.mean():.1f} tokens\n")
        f.write(f"Chosen P95: {np.percentile(chosen_lengths, 95):.1f} tokens\n")
        f.write(f"Chosen P99: {np.percentile(chosen_lengths, 99):.1f} tokens\n")
        f.write(f"Rejected Mean: {rejected_lengths.mean():.1f} tokens\n")
        f.write(f"Rejected P95: {np.percentile(rejected_lengths, 95):.1f} tokens\n")
        f.write(f"Rejected P99: {np.percentile(rejected_lengths, 99):.1f} tokens\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Query Variant Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        for stat in variant_stats:
            f.write(f"[{stat['name']}]\n")
            f.write(f"  Query Mean: {stat['query_mean']:.1f} tokens\n")
            f.write(f"  Query P95: {stat['query_p95']:.1f} tokens\n")
            f.write(f"  Query P99: {stat['query_p99']:.1f} tokens\n")
            f.write(f"  Query Max: {stat['query_max']} tokens\n")
            f.write(f"  Total (Q+C) Mean: {stat['total_chosen_mean']:.1f} tokens\n")
            f.write(f"  Total (Q+C) P95: {stat['total_chosen_p95']:.1f} tokens\n")
            f.write(f"  Total (Q+C) P99: {stat['total_chosen_p99']:.1f} tokens\n")
            f.write(f"  Total (Q+C) Max: {stat['total_chosen_max']} tokens\n")
            f.write(f"  Total (Q+R) Mean: {stat['total_rejected_mean']:.1f} tokens\n")
            f.write(f"  Total (Q+R) P95: {stat['total_rejected_p95']:.1f} tokens\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Samples Exceeding Max Length\n")
        f.write("=" * 80 + "\n\n")
        
        for max_len in [2048, 4096, 8192, 16384]:
            f.write(f"\nmax_length = {max_len}:\n")
            for stat in variant_stats:
                total_chosen = np.array(variant_token_counts[stat['name']]) + chosen_lengths[:len(variant_token_counts[stat['name']])]
                exceed_chosen = (total_chosen > max_len).sum()
                pct_chosen = 100 * exceed_chosen / len(total_chosen)
                
                total_rejected = np.array(variant_token_counts[stat['name']]) + rejected_lengths[:len(variant_token_counts[stat['name']])]
                exceed_rejected = (total_rejected > max_len).sum()
                pct_rejected = 100 * exceed_rejected / len(total_rejected)
                
                f.write(f"  {stat['name']}: {exceed_chosen} chosen ({pct_chosen:.1f}%), {exceed_rejected} rejected ({pct_rejected:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Recommendations\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Based on the analysis:\n\n")
        f.write("1. For max_length=2048:\n")
        for stat in variant_stats[:5]:
            total_chosen = np.array(variant_token_counts[stat['name']]) + chosen_lengths[:len(variant_token_counts[stat['name']])]
            exceed_pct = 100 * (total_chosen > 2048).sum() / len(total_chosen)
            f.write(f"   - {stat['name']}: {exceed_pct:.1f}% exceed\n")
        
        f.write("\n2. For max_length=4096:\n")
        for stat in variant_stats[:5]:
            total_chosen = np.array(variant_token_counts[stat['name']]) + chosen_lengths[:len(variant_token_counts[stat['name']])]
            exceed_pct = 100 * (total_chosen > 4096).sum() / len(total_chosen)
            f.write(f"   - {stat['name']}: {exceed_pct:.1f}% exceed\n")
        
        f.write("\n3. Recommended variants:\n")
        f.write("   - If using max_length=2048: instruction_title_abstract_300t or instruction_title_abstract\n")
        f.write("   - If using max_length=4096: instruction_title_content_1000t or instruction_title_content_2000t\n")
        f.write("   - For maximum context: instruction_title_content_3000t (requires max_length=8192)\n")
    
    print(f"Detailed results saved to: {results_path}")
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()