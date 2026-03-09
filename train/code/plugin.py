"""
Custom reward function plugin for GRPO training.

This plugin provides:
1. Format score calculation for review outputs
2. Integration with Reward Model scores
3. Combined reward computation
"""

import re
import sys
import os
from typing import List, Dict, Optional
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_function import compute_format_score


class ReviewFormatReward:
    """
    Reward function for evaluating review format quality.

    Usage in swift command line:
        --reward_func train.code.plugin.review_format_reward
        --reward_weights 1.0
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            score = compute_format_score(completion)
            rewards.append(score)
        return rewards


review_format_reward = ReviewFormatReward()


def create_combined_reward_function(alpha: float = 1.0):
    """
    Create a reward function that combines format score and RM score.

    Args:
        alpha: Weight for RM score (default: 1.0)

    Returns:
        Reward function
    """

    class CombinedReward:
        def __init__(self, alpha: float):
            self.alpha = alpha

        def __call__(
            self,
            completions: List[str],
            rm_scores: Optional[List[float]] = None,
            **kwargs,
        ) -> List[float]:
            rewards = []
            for i, completion in enumerate(completions):
                format_score = compute_format_score(completion)
                rm_score = rm_scores[i] if rm_scores and i < len(rm_scores) else 0.0
                total = format_score + self.alpha * rm_score
                rewards.append(total)
            return rewards

    return CombinedReward(alpha)


combined_reward = create_combined_reward_function(alpha=1.0)


class FormatScoreBreakdown:
    """
    Detailed format score breakdown for analysis.

    Usage:
        --reward_func train.code.plugin.format_score_breakdown
        --reward_weights 1.0
    """

    FORMAT_PATTERNS = {
        "overall_quality": r"\*\*Overall Quality:\*\*\s*([1-9](?:\.[0-9])?|10(?:\.0)?)",
        "review_confidence": r"\*\*Review Confidence:\*\*\s*([1-5](?:\.[0-9])?)",
        "key_points": r"### Key Points",
        "strengths_weaknesses": r"### Strengths and Weaknesses",
        "suggestions": r"### Suggestions",
        "rating_section": r"### Rating",
    }

    FORMAT_SCORES = {
        "overall_quality": 1.0,
        "review_confidence": 1.0,
        "key_points": 0.5,
        "strengths_weaknesses": 0.5,
        "suggestions": 0.5,
        "rating_section": 0.5,
    }

    def __call__(self, completions: List[str], **kwargs) -> List[Dict]:
        breakdowns = []
        for completion in completions:
            breakdown = {}
            total = 0.0

            for key, pattern in self.FORMAT_PATTERNS.items():
                matched = bool(re.search(pattern, completion, re.IGNORECASE))
                score = self.FORMAT_SCORES[key] if matched else 0.0
                breakdown[f"{key}_matched"] = matched
                breakdown[f"{key}_score"] = score
                total += score

            breakdown["total_score"] = total
            breakdowns.append(breakdown)

        return breakdowns


format_score_breakdown = FormatScoreBreakdown()
