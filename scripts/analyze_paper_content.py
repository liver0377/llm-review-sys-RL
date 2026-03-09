#!/usr/bin/env python3
"""
Analyze paper content in the dataset to decide what to include in query for RM training.
This script helps find the optimal balance between:
1. Providing enough context for RM to detect hallucinations
2. Keeping token count reasonable
"""

import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter

def extract_paper_sections(content):
    """Extract different sections from paper content"""
    sections = {
        'abstract': '',
        'introduction': '',
        'method': '',
        'experiments': '',
        'conclusion': '',
        'full_content': content
    }
    
    # Try to extract abstract
    abstract_patterns = [
        r'Abstract\s*\n(.*?)(?=\n\s*\n|\n#{1,3}\s|[A-Z][a-z]+\s*\n)',
        r'ABSTRACT\s*\n(.*?)(?=\n\s*\n|\n#{1,3}\s|[A-Z][a-z]+\s*\n)',
    ]
    for pattern in abstract_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            sections['abstract'] = match.group(1).strip()
            break
    
    # Try to extract introduction
    intro_patterns = [
        r'(?:^|\n)\s*#{0,3}\s*Introduction\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*2\.|\n\s*Background|\n\s*Related Work)',
        r'(?:^|\n)\s*#{0,3}\s*1\.\s*Introduction\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*2\.)',
    ]
    for pattern in intro_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            sections['introduction'] = match.group(1).strip()
            break
    
    # Try to extract method section
    method_patterns = [
        r'(?:^|\n)\s*#{0,3}\s*(?:Method|Approach|Methodology)\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*Experiments|\n\s*Evaluation)',
        r'(?:^|\n)\s*#{0,3}\s*\d+\.\s*(?:Method|Approach)\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*\d+\.)',
    ]
    for pattern in method_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            sections['method'] = match.group(1).strip()
            break
    
    # Try to extract experiments section
    exp_patterns = [
        r'(?:^|\n)\s*#{0,3}\s*(?:Experiments|Evaluation|Results)\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*Conclusion|\n\s*Discussion)',
        r'(?:^|\n)\s*#{0,3}\s*\d+\.\s*(?:Experiments|Evaluation)\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*\d+\.)',
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            sections['experiments'] = match.group(1).strip()
            break
    
    # Try to extract conclusion
    conc_patterns = [
        r'(?:^|\n)\s*#{0,3}\s*Conclusion[s]?\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*References|\n\s*Acknowledgment|\Z)',
        r'(?:^|\n)\s*#{0,3}\s*\d+\.\s*Conclusion[s]?\s*\n(.*?)(?=\n\s*#{1,3}\s|\n\s*References|\Z)',
    ]
    for pattern in conc_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            sections['conclusion'] = match.group(1).strip()
            break
    
    return sections

def truncate_text(text, max_chars=2000):
    """Truncate text to max_chars, trying to end at sentence boundary"""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    # Try to end at last sentence
    last_period = truncated.rfind('.')
    last_exclaim = truncated.rfind('!')
    last_question = truncated.rfind('?')
    last_sentence_end = max(last_period, last_exclaim, last_question)
    
    if last_sentence_end > max_chars * 0.7:
        return truncated[:last_sentence_end + 1]
    return truncated + '...'

def create_query_variants(instruction, paper_title, paper_content, sections, tokenizer):
    """Create different query variants with different paper content strategies"""
    variants = {}
    
    # Base instruction
    base_query = instruction.strip()
    
    # Variant 1: Only instruction (current approach)
    variants['instruction_only'] = base_query
    
    # Variant 2: Instruction + Title
    query_with_title = f"{base_query}\n\nPaper Title: {paper_title}"
    variants['instruction_title'] = query_with_title
    
    # Variant 3: Instruction + Title + Abstract (full)
    if sections['abstract']:
        query_with_abstract = f"{query_with_title}\n\nAbstract:\n{sections['abstract']}"
        variants['instruction_title_abstract'] = query_with_abstract
    
    # Variant 4: Instruction + Title + Abstract (truncated to 1000 chars)
    if sections['abstract']:
        abstract_truncated = truncate_text(sections['abstract'], max_chars=1000)
        query_with_abstract_trunc = f"{query_with_title}\n\nAbstract:\n{abstract_truncated}"
        variants['instruction_title_abstract_1k'] = query_with_abstract_trunc
    
    # Variant 5: Instruction + Title + Introduction (truncated to 2000 chars)
    if sections['introduction']:
        intro_truncated = truncate_text(sections['introduction'], max_chars=2000)
        query_with_intro = f"{query_with_title}\n\nIntroduction:\n{intro_truncated}"
        variants['instruction_title_intro_2k'] = query_with_intro
    
    # Variant 6: Instruction + Title + Abstract + Intro (both truncated)
    if sections['abstract'] and sections['introduction']:
        abstract_trunc = truncate_text(sections['abstract'], max_chars=800)
        intro_trunc = truncate_text(sections['introduction'], max_chars=1500)
        query_with_both = f"{query_with_title}\n\nAbstract:\n{abstract_trunc}\n\nIntroduction:\n{intro_trunc}"
        variants['instruction_title_abstract_intro'] = query_with_both
    
    # Variant 7: Instruction + Title + First 3000 chars of paper
    content_truncated = truncate_text(paper_content, max_chars=3000)
    query_with_content_3k = f"{query_with_title}\n\nPaper Content:\n{content_truncated}"
    variants['instruction_title_content_3k'] = query_with_content_3k
    
    # Variant 8: Instruction + Title + First 5000 chars of paper
    content_truncated = truncate_text(paper_content, max_chars=5000)
    query_with_content_5k = f"{query_with_title}\n\nPaper Content:\n{content_truncated}"
    variants['instruction_title_content_5k'] = query_with_content_5k
    
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
    print("=" * 70)
    print("Paper Content Analysis for RM Query Design")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    model_path = "models/qwen3-8b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Load data
    print("\n[2/5] Loading dataset...")
    data_path = "data/openreview_dataset/dpo_vllm_as_rejected_train_cleaned.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Analyze paper content
    print("\n[3/5] Analyzing paper content...")
    
    paper_lengths = []
    abstract_lengths = []
    intro_lengths = []
    method_lengths = []
    exp_lengths = []
    conclusion_lengths = []
    
    variant_token_counts = {}
    
    # Sample for detailed analysis
    sample_size = min(1000, len(data))
    sampled_indices = np.random.choice(len(data), sample_size, replace=False)
    
    for idx in tqdm(sampled_indices, desc="Processing papers"):
        item = data[idx]
        prompt = item.get('prompt', '')
        
        # Extract paper details
        if 'Paper Details:' not in prompt:
            continue
        
        paper_section = prompt.split('Paper Details:')[-1]
        
        # Extract title
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', paper_section)
        paper_title = title_match.group(1).strip() if title_match else ''
        
        # Extract content
        content_match = re.search(r'Content:\s*(.+)', paper_section, re.DOTALL)
        paper_content = content_match.group(1).strip() if content_match else ''
        
        if not paper_content:
            continue
        
        # Extract sections
        sections = extract_paper_sections(paper_content)
        
        # Record section lengths
        paper_lengths.append(len(paper_content))
        
        if sections['abstract']:
            abstract_lengths.append(len(sections['abstract']))
        if sections['introduction']:
            intro_lengths.append(len(sections['introduction']))
        if sections['method']:
            method_lengths.append(len(sections['method']))
        if sections['experiments']:
            exp_lengths.append(len(sections['experiments']))
        if sections['conclusion']:
            conclusion_lengths.append(len(sections['conclusion']))
        
        # Create query variants for this sample
        instruction = prompt.split('Paper Details:')[0].strip()
        variants = create_query_variants(instruction, paper_title, paper_content, sections, tokenizer)
        
        for variant_name, variant_data in variants.items():
            if variant_name not in variant_token_counts:
                variant_token_counts[variant_name] = []
            variant_token_counts[variant_name].append(variant_data['token_count'])
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Paper Content Statistics")
    print("=" * 70)
    
    def print_stats(name, lengths):
        if not lengths:
            print(f"\n[{name}] No data found")
            return
        arr = np.array(lengths)
        print(f"\n[{name}]")
        print(f"  Samples found: {len(lengths)}")
        print(f"  Mean chars: {arr.mean():.1f}")
        print(f"  Median chars: {np.median(arr):.1f}")
        print(f"  Min chars: {arr.min()}")
        print(f"  Max chars: {arr.max()}")
        print(f"  P95 chars: {np.percentile(arr, 95):.1f}")
    
    print_stats("Full Paper Content", paper_lengths)
    print_stats("Abstract", abstract_lengths)
    print_stats("Introduction", intro_lengths)
    print_stats("Method", method_lengths)
    print_stats("Experiments", exp_lengths)
    print_stats("Conclusion", conclusion_lengths)
    
    print("\n" + "=" * 70)
    print("Query Variant Token Statistics")
    print("=" * 70)
    
    variant_stats = []
    for variant_name, token_counts in sorted(variant_token_counts.items()):
        arr = np.array(token_counts)
        stat = {
            'name': variant_name,
            'mean': arr.mean(),
            'median': np.median(arr),
            'min': arr.min(),
            'max': arr.max(),
            'p95': np.percentile(arr, 95),
            'p99': np.percentile(arr, 99)
        }
        variant_stats.append(stat)
        
        print(f"\n[{variant_name}]")
        print(f"  Mean tokens: {stat['mean']:.1f}")
        print(f"  Median tokens: {stat['median']:.1f}")
        print(f"  Min tokens: {stat['min']}")
        print(f"  Max tokens: {stat['max']}")
        print(f"  P95 tokens: {stat['p95']:.1f}")
        print(f"  P99 tokens: {stat['p99']:.1f}")
    
    # Create visualizations
    print("\n[4/5] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Paper Content Analysis for RM Query Design', fontsize=16, fontweight='bold')
    
    # Plot 1: Paper section character lengths
    ax = axes[0, 0]
    section_data = []
    section_labels = []
    for name, lengths in [('Full Paper', paper_lengths), ('Abstract', abstract_lengths), 
                          ('Introduction', intro_lengths), ('Method', method_lengths),
                          ('Experiments', exp_lengths), ('Conclusion', conclusion_lengths)]:
        if lengths:
            section_data.append(lengths)
            section_labels.append(name)
    
    bp = ax.boxplot(section_data, labels=section_labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(section_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Character Count')
    ax.set_title('Paper Section Lengths (Characters)')
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)
    
    # Plot 2: Query variant token counts
    ax = axes[0, 1]
    variant_names = [s['name'] for s in variant_stats]
    means = [s['mean'] for s in variant_stats]
    p95s = [s['p95'] for s in variant_stats]
    
    x = np.arange(len(variant_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, means, width, label='Mean', color='skyblue')
    bars2 = ax.bar(x + width/2, p95s, width, label='P95', color='lightcoral')
    
    ax.set_ylabel('Token Count')
    ax.set_title('Query Variant Token Counts')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in variant_names], fontsize=7)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=6)
    
    # Plot 3: Histogram of paper lengths
    ax = axes[1, 0]
    ax.hist(paper_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(paper_lengths), color='red', linestyle='--', 
               label=f'Median: {np.median(paper_lengths):.0f}')
    ax.axvline(5000, color='green', linestyle=':', linewidth=2, label='5000 chars')
    ax.axvline(10000, color='blue', linestyle=':', linewidth=2, label='10000 chars')
    ax.set_xlabel('Character Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Full Paper Content Length Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Comparison of abstract vs intro lengths
    ax = axes[1, 1]
    if abstract_lengths and intro_lengths:
        ax.scatter(abstract_lengths[:len(intro_lengths)], intro_lengths, alpha=0.5, color='purple')
        ax.set_xlabel('Abstract Length (chars)')
        ax.set_ylabel('Introduction Length (chars)')
        ax.set_title('Abstract vs Introduction Length')
        ax.grid(alpha=0.3)
        
        # Add diagonal lines for reference
        max_val = max(max(abstract_lengths), max(intro_lengths))
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3, label='y=x')
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'docs/paper_content_analysis.png'
    os.makedirs('docs', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Generate recommendation
    print("\n[5/5] Generating recommendations...")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("\nBased on the analysis, here are the recommended strategies:\n")
    
    # Find best variant
    best_variants = []
    for stat in variant_stats:
        # Good variant: mean < 1500, p95 < 2000, provides context
        score = 0
        if stat['mean'] < 1500:
            score += 1
        if stat['p95'] < 2000:
            score += 1
        if 'abstract' in stat['name'] or 'intro' in stat['name'] or 'content' in stat['name']:
            score += 1
        if stat['mean'] > 500:  # Not too short
            score += 1
        
        best_variants.append((stat['name'], score, stat['mean'], stat['p95']))
    
    best_variants.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 3 Recommended Query Strategies:")
    for i, (name, score, mean, p95) in enumerate(best_variants[:3], 1):
        print(f"\n{i}. {name}")
        print(f"   - Mean tokens: {mean:.1f}")
        print(f"   - P95 tokens: {p95:.1f}")
        print(f"   - Score: {score}/4")
    
    # Save detailed results
    results_path = 'docs/paper_content_analysis_results.txt'
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Paper Content Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total samples analyzed: {sample_size}\n\n")
        
        f.write("Section Statistics:\n")
        for name, lengths in [("Full Paper", paper_lengths), ("Abstract", abstract_lengths),
                             ("Introduction", intro_lengths), ("Method", method_lengths),
                             ("Experiments", exp_lengths), ("Conclusion", conclusion_lengths)]:
            if lengths:
                arr = np.array(lengths)
                f.write(f"\n{name}:\n")
                f.write(f"  Samples: {len(lengths)}\n")
                f.write(f"  Mean: {arr.mean():.1f} chars\n")
                f.write(f"  Median: {np.median(arr):.1f} chars\n")
                f.write(f"  Min: {arr.min()} chars\n")
                f.write(f"  Max: {arr.max()} chars\n")
                f.write(f"  P95: {np.percentile(arr, 95):.1f} chars\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Query Variant Statistics:\n")
        f.write("=" * 70 + "\n")
        
        for stat in variant_stats:
            f.write(f"\n{stat['name']}:\n")
            f.write(f"  Mean tokens: {stat['mean']:.1f}\n")
            f.write(f"  Median tokens: {stat['median']:.1f}\n")
            f.write(f"  Min tokens: {stat['min']}\n")
            f.write(f"  Max tokens: {stat['max']}\n")
            f.write(f"  P95 tokens: {stat['p95']:.1f}\n")
            f.write(f"  P99 tokens: {stat['p99']:.1f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Top 3 Recommended Strategies:\n")
        f.write("=" * 70 + "\n")
        for i, (name, score, mean, p95) in enumerate(best_variants[:3], 1):
            f.write(f"\n{i}. {name}\n")
            f.write(f"   Mean tokens: {mean:.1f}\n")
            f.write(f"   P95 tokens: {p95:.1f}\n")
            f.write(f"   Score: {score}/4\n")
    
    print(f"\nDetailed results saved to: {results_path}")
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()