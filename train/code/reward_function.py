"""
External Reward Function for Paper Review GRPO Training
Calls external vLLM RM service to compute rewards
"""

import asyncio
import logging
import os
import re
from typing import Dict, Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class ExternalReviewRM:
    """External Reward Model for paper review evaluation"""

    def __init__(
        self,
        rm_api_base: str = "http://127.0.0.1:8002/v1",
        max_concurrent: int = 32,
        timeout: int = 120,
    ):
        self.rm_api_base = rm_api_base
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        self.client = AsyncOpenAI(
            api_key="EMPTY", base_url=rm_api_base, timeout=timeout, max_retries=2
        )

        self.model_name = None
        self._init_model_name()

        logger.info(f"Initialized ExternalReviewRM")
        logger.info(f"  API Base: {rm_api_base}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Max Concurrent: {max_concurrent}")

    def _init_model_name(self):
        """Initialize model name from RM service"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models = loop.run_until_complete(self.client.models.list())
            self.model_name = (
                models.data[0].id if models.data else "models/Qwen3-8B-Base"
            )
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to get model name: {e}")
            self.model_name = "models/Qwen3-8B-Base"

    def compute_format_score(self, text: str) -> float:
        """Compute format score (0-4) based on review structure"""
        score = 0.0

        if re.search(r"\*\*Overall Quality:\*\*", text, re.IGNORECASE):
            score += 1.0
        if re.search(r"### Key Points", text, re.IGNORECASE):
            score += 0.5
        if re.search(r"\*\*Strengths:\*\*", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"\*\*Weaknesses:\*\*", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"### Suggestions for Improvement", text, re.IGNORECASE):
            score += 0.5
        if re.search(r"### Rating", text, re.IGNORECASE):
            score += 0.25

        return score

    async def compute_rm_score(self, prompt: str, response: str) -> float:
        """Compute quality score (0-10) using external RM service"""
        try:
            user_prompt = f"""<Paper Review Task>
{prompt[:1500]}
</Paper Review Task>

<Generated Review>
{response[:2500]}
</Generated Review>

Please assess the quality of this review (1-10). 
Provide a score in format: **Overall Quality:** X.X"""

            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a paper review quality evaluator.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=128,
                temperature=0.1,
            )

            content = resp.choices[0].message.content
            match = re.search(
                r"Overall Quality:\*{0,2}\s*(\d+\.?\d*)", content, re.IGNORECASE
            )
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)
            return 5.0
        except Exception as e:
            logger.error(f"RM API error: {e}")
            return 5.0

    async def compute_reward(
        self, prompt: str, response: str, **kwargs
    ) -> Dict[str, Any]:
        """Compute total reward for a review"""
        format_score = self.compute_format_score(response)
        rm_score = await self.compute_rm_score(prompt, response)

        total_score = format_score + rm_score

        return {
            "score": total_score,
            "format_score": format_score,
            "rm_score": rm_score,
        }


def get_reward_function(rm_api_base: str = "http://127.0.0.1:8002/v1", **kwargs):
    """Factory function to create reward function instance"""
    return ExternalReviewRM(rm_api_base=rm_api_base, **kwargs)


async def compute_score_async(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
):
    """
    Async function compatible with veRL's reward function interface

    This function is called by veRL during training to compute rewards.

    Args:
        data_source: Data source identifier
        solution_str: Generated review text
        ground_truth: Ground truth (not used for review generation)
        extra_info: Extra information from dataset

    Returns:
        Dict with 'score' key and other metadata
    """
    rm_api_base = kwargs.get("rm_api_base", "http://127.0.0.1:8002/v1")

    rm = get_reward_function(rm_api_base=rm_api_base)

    prompt = ""
    for key, value in extra_info.items():
        if isinstance(value, str) and len(value) > 100:
            prompt = value
            break

    if not prompt and "prompt" in extra_info:
        prompt = extra_info["prompt"]

    result = await rm.compute_reward(prompt, solution_str)

    return result
