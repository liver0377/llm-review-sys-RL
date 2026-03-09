import re
from typing import List, Dict, Any
from swift.rewards import ORM

FORMAT_PATTERNS = {
    "overall_quality": r"\*{0,2}Overall Quality:\*{0,2}\s*([1-9](?:\.[0-9])?|10(?:\.0)?)",
    "review_confidence": r"\*{0,2}Review Confidence:\*{0,2}\s*([1-5](?:\.[0-9])?)",
    "key_points": r"### Key Points",
    "strengths_weaknesses": r"### Strengths and Weaknesses",
    "strengths": r"\*\*Strengths:\*\*",
    "weaknesses": r"\*\*Weaknesses:\*\*",
    "suggestions": r"### Suggestions for Improvement",
    "rating_section": r"### Rating",
}

FORMAT_SCORES = {
    "overall_quality": 1.0,
    "review_confidence": 1.0,
    "key_points": 0.5,
    "strengths_weaknesses": 0.25,
    "strengths": 0.25,
    "weaknesses": 0.25,
    "suggestions": 0.5,
    "rating_section": 0.25,
}


def compute_format_score(response: str) -> float:
    total_score = 0.0
    for key, pattern in FORMAT_PATTERNS.items():
        if re.search(pattern, response, re.IGNORECASE):
            total_score += FORMAT_SCORES[key]
    return total_score


def extract_overall_quality(response: str) -> float:
    match = re.search(
        r"\*\*Overall Quality:\*\*\s*([1-9](?:\.[0-9])?|10(?:\.0)?)",
        response,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return None


def extract_review_confidence(response: str) -> float:
    match = re.search(
        r"\*\*Review Confidence:\*\*\s*([1-5](?:\.[0-9])?)", response, re.IGNORECASE
    )
    if match:
        return float(match.group(1))
    return None


class FormatRewardFunction(ORM):
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            format_score = compute_format_score(completion)
            rewards.append(format_score)
        return rewards


class CombinedRewardFunction(ORM):
    def __init__(self, alpha: float = 1.0, format_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.format_weight = format_weight

    def __call__(
        self, completions: List[str], rm_scores: List[float] = None, **kwargs
    ) -> List[float]:
        rewards = []
        for i, completion in enumerate(completions):
            format_score = compute_format_score(completion)
            rm_score = rm_scores[i] if rm_scores and i < len(rm_scores) else 0.0
            total_reward = self.format_weight * format_score + self.alpha * rm_score
            rewards.append(total_reward)
        return rewards


def create_reward_function(alpha: float = 1.0):
    return FormatRewardFunction(alpha=alpha)


def analyze_review_quality(response: str) -> Dict[str, Any]:
    format_score = compute_format_score(response)
    overall_quality = extract_overall_quality(response)
    review_confidence = extract_review_confidence(response)

    has_key_points = bool(
        re.search(FORMAT_PATTERNS["key_points"], response, re.IGNORECASE)
    )
    has_strengths_weaknesses = bool(
        re.search(FORMAT_PATTERNS["strengths_weaknesses"], response, re.IGNORECASE)
    )
    has_strengths = bool(
        re.search(FORMAT_PATTERNS["strengths"], response, re.IGNORECASE)
    )
    has_weaknesses = bool(
        re.search(FORMAT_PATTERNS["weaknesses"], response, re.IGNORECASE)
    )
    has_suggestions = bool(
        re.search(FORMAT_PATTERNS["suggestions"], response, re.IGNORECASE)
    )
    has_rating = bool(
        re.search(FORMAT_PATTERNS["rating_section"], response, re.IGNORECASE)
    )

    return {
        "format_score": format_score,
        "max_format_score": 4.0,
        "format_percentage": format_score / 4.0 * 100,
        "overall_quality": overall_quality,
        "review_confidence": review_confidence,
        "has_key_points": has_key_points,
        "has_strengths_weaknesses": has_strengths_weaknesses,
        "has_strengths": has_strengths,
        "has_weaknesses": has_weaknesses,
        "has_suggestions": has_suggestions,
        "has_rating": has_rating,
        "is_complete": format_score >= 4.0,
    }
