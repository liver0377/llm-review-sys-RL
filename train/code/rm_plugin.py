from typing import List, Dict, Optional
import torch
from swift.plugin import DefaultRMPlugin
from swift import TransformersEngine


class ReviewRewardModelPlugin(DefaultRMPlugin):
    def __init__(
        self,
        model,
        template,
        alpha: float = 1.0,
        format_weight: float = 1.0,
        max_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(model, template, **kwargs)
        self.alpha = alpha
        self.format_weight = format_weight
        self.max_batch_size = max_batch_size

        self.engine = TransformersEngine(
            self.model, template=self.template, max_batch_size=self.max_batch_size
        )

        self._import_format_score()

    def _import_format_score(self):
        import sys
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        from reward_function import compute_format_score

        self.compute_format_score = compute_format_score

    def __call__(self, inputs: List[Dict]) -> List[float]:
        rewards = []

        for item in inputs:
            messages = item.get("messages", [])
            completion = ""
            for msg in messages:
                if msg.get("role") in ["assistant", "user"]:
                    completion = msg.get("content", "")

            format_score = self.compute_format_score(completion)

            rm_score = self._get_rm_score(messages)

            total_reward = self.format_weight * format_score + self.alpha * rm_score
            rewards.append(total_reward)

        return rewards

    @torch.no_grad()
    def _get_rm_score(self, messages: List[Dict]) -> float:
        from swift import RequestConfig
        from swift.utils import get_logger

        logger = get_logger()

        request_config = RequestConfig(
            max_tokens=10,
            temperature=0.0,
        )

        try:
            result = self.engine.infer([messages], request_config, use_tqdm=False)
            if result and len(result) > 0:
                response_text = result[0].choices[0].message.content.strip()

                try:
                    score = float(response_text)
                    return score
                except ValueError:
                    return 0.0
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting RM score: {e}")
            return 0.0


class FormatOnlyPlugin(DefaultRMPlugin):
    def __init__(self, model=None, template=None, **kwargs):
        super().__init__(model, template, **kwargs)
        self._import_format_score()

    def _import_format_score(self):
        import sys
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        from reward_function import compute_format_score

        self.compute_format_score = compute_format_score

    def __call__(self, inputs: List[Dict]) -> List[float]:
        rewards = []

        for item in inputs:
            messages = item.get("messages", [])
            completion = ""
            for msg in messages:
                if msg.get("role") in ["assistant", "user"]:
                    completion = msg.get("content", "")

            format_score = self.compute_format_score(completion)
            rewards.append(format_score)

        return rewards


def get_rm_plugin(
    plugin_type: str = "review",
    model=None,
    template=None,
    alpha: float = 1.0,
    format_weight: float = 1.0,
    **kwargs,
):
    if plugin_type == "review":
        return ReviewRewardModelPlugin(
            model=model,
            template=template,
            alpha=alpha,
            format_weight=format_weight,
            **kwargs,
        )
    elif plugin_type == "format_only":
        return FormatOnlyPlugin(model=model, template=template, **kwargs)
    else:
        raise ValueError(f"Unknown plugin type: {plugin_type}")
