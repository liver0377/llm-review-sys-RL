"""
Custom reward function plugin for GRPO training.

This plugin provides:
1. Format score calculation for review outputs
2. Integration with Reward Model scores
3. Combined reward computation

Usage in swift command line:
    --reward_func train.code.plugin:review_format_reward
    --reward_func train.code.plugin:combined_reward
    --alpha 1.0
    --format_weight 1.0
"""

import re
import sys
import os
from typing import List, Dict, Optional
from swift.rewards import ORM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_function import compute_format_score


class ReviewFormatReward(ORM):
    """
    Reward function for evaluating review format quality.

    Usage in swift command line:
        --reward_func train.code.plugin:review_format_reward
        --reward_weights 1.0
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            score = compute_format_score(completion)
            rewards.append(score)
        return rewards


review_format_reward = ReviewFormatReward()


class CombinedReward(ORM):
    """
    Combined reward function that uses both format score and RM score.

    This function combines:
    1. Format score: based on the structure and completeness of the review
    2. RM score: from the generative reward model (Qwen3.5-35B-A3B)

    Usage in swift command line:
        --reward_func train.code.reward_function:combined_reward
        --alpha 1.0              # Weight for RM score
        --format_weight 1.0      # Weight for format score

    The total reward is: format_weight * format_score + alpha * rm_score
    """

    def __init__(self, alpha: float = 1.0, format_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.format_weight = format_weight

    def __call__(
        self,
        completions: List[str],
        rm_scores: Optional[List[float]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute combined rewards.

        Args:
            completions: List of generated review texts
            rm_scores: List of scores from the generative reward model
            **kwargs: Additional arguments

        Returns:
            List of combined reward scores
        """
        rewards = []
        for i, completion in enumerate(completions):
            # Compute format score (0-4 range)
            format_score = compute_format_score(completion)

            # Get RM score if available
            rm_score = rm_scores[i] if rm_scores and i < len(rm_scores) else 0.0

            # Combine scores with weights
            total = self.format_weight * format_score + self.alpha * rm_score
            rewards.append(total)

        return rewards


def create_combined_reward_function(alpha: float = 1.0, format_weight: float = 1.0):
    """
    Factory function to create a combined reward function.

    Args:
        alpha: Weight for RM score (default: 1.0)
        format_weight: Weight for format score (default: 1.0)

    Returns:
        CombinedReward instance
    """
    return CombinedReward(alpha=alpha, format_weight=format_weight)


# Default instance with alpha=1.0, format_weight=1.0
combined_reward = CombinedReward(alpha=1.0, format_weight=1.0)


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
