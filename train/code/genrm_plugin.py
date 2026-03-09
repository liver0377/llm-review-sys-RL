"""
生成式奖励模型插件
用于使用 Qwen3.5-35B-A3B 对评审质量进行打分
"""

import re
import textwrap
import torch
from copy import deepcopy
from typing import List, Dict, Optional

from swift.rewards.rm_plugin import DefaultRMPlugin
from swift.infer_engine import TransformersEngine, RequestConfig
from swift.utils import get_logger

logger = get_logger()


class ReviewGenRMPlugin(DefaultRMPlugin):
    """
    生成式奖励模型插件 - 评审质量打分

    使用大语言模型（Qwen3.5-35B-A3B）对生成的评审进行质量评估，
    并组合格式分数和 RM 分数作为最终奖励。

    Args:
        model: 奖励模型
        template: 模板
        alpha (float): RM 分数的权重 (默认: 1.0)
        format_weight (float): 格式分数的权重 (默认: 1.0)
        max_tokens (int): 生成的最大 token 数 (默认: 512)
        temperature (float): 生成温度 (默认: 0.1，更确定性)
    """

    def __init__(
        self,
        model,
        template,
        alpha: float = 1.0,
        format_weight: float = 1.0,
        max_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(model, template, **kwargs)
        self.alpha = alpha
        self.format_weight = format_weight
        self.max_tokens = max_tokens
        self.temperature = temperature

        # 初始化推理引擎
        self.engine = TransformersEngine(
            self.model,
            template=self.template,
            max_batch_size=0  # 0 表示无限制
        )

        # 配置生成参数
        self.request_config = RequestConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )

        # 导入格式评分函数
        self._import_modules()

        # 设置评审打分的 system prompt
        self.system_prompt = textwrap.dedent("""
            You are an academic paper reviewer evaluating a generated review. Your task is to assess the quality of the given review based on its clarity, completeness, accuracy, and constructiveness.

            Provide a brief assessment (2-3 sentences) explaining your evaluation, then give a score.

            Your response should follow this format:
            <Your brief assessment here>

            **Overall Quality:** X.X

            Scoring guidelines:
            - 9.0-10.0: Excellent - comprehensive, insightful, well-structured
            - 7.0-8.9: Good - solid analysis with minor gaps
            - 5.0-6.9: Average - adequate but lacks depth or clarity
            - 3.0-4.9: Poor - significant issues or incomplete
            - 0.0-2.9: Very Poor - fails to meet basic requirements

            **Overall Quality:** (1–10, where 10 is top-tier)
        """).strip()

    def _import_modules(self):
        """动态导入依赖模块"""
        import sys
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        from reward_function import compute_format_score
        self.compute_format_score = compute_format_score

    def __call__(self, inputs: List[Dict], **kwargs) -> List[float]:
        """
        计算奖励分数

        Args:
            inputs: 输入列表，每个元素包含：
                - messages: 对话消息列表
                - 其他数据集字段

        Returns:
            List[float]: 奖励分数列表
        """
        rewards = []

        for item in inputs:
            # 提取生成的评审内容
            messages = item.get("messages", [])
            completion = self._extract_completion(messages)

            # 计算格式分数 (0-4)
            format_score = self.compute_format_score(completion)

            # 使用生成式 RM 获取质量分数 (0-10)
            rm_score = self._get_genrm_score(messages, completion)

            # 组合奖励
            total_reward = (
                self.format_weight * format_score +
                self.alpha * rm_score
            )

            rewards.append(total_reward)

            logger.debug(
                f"Format: {format_score:.2f}, RM: {rm_score:.2f}, "
                f"Total: {total_reward:.2f}"
            )

        return rewards

    def _extract_completion(self, messages: List[Dict]) -> str:
        """从消息中提取模型生成的评审内容"""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _get_genrm_score(self, messages: List[Dict], completion: str) -> float:
        """
        使用生成式奖励模型获取质量分数

        Args:
            messages: 原始对话消息
            completion: 生成的评审内容

        Returns:
            float: 质量分数 (0-10)
        """
        # 构造评审打分的 prompt
        rm_messages = self._prepare_rm_prompt(messages, completion)

        try:
            # 调用生成式 RM
            results = self.engine.infer(
                [rm_messages],
                self.request_config,
                use_tqdm=False
            )

            if results and len(results) > 0:
                response = results[0].choices[0].message.content.strip()

                # 从响应中提取评分
                score = self._extract_rating(response)

                if score is not None:
                    logger.debug(f"GenRM response: {response[:100]}...")
                    logger.debug(f"Extracted score: {score}")
                    return score
                else:
                    logger.warning(
                        f"Failed to extract rating from response: {response[:200]}"
                    )
                    return 5.0  # 默认中等分数
            else:
                logger.warning("Empty response from GenRM")
                return 5.0

        except Exception as e:
            logger.error(f"Error calling GenRM: {e}")
            return 5.0

    def _prepare_rm_prompt(self, original_messages: List[Dict], completion: str) -> List[Dict]:
        """
        准备给生成式 RM 的 prompt

        Args:
            original_messages: 原始对话消息
            completion: 生成的评审内容

        Returns:
            List[Dict]: 给 RM 的消息列表
        """
        # 提取原始任务（论文信息）
        query = ""
        for msg in original_messages:
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        # 构造评审内容描述
        review_content = textwrap.dedent(f"""
            [生成的评审内容]
            {completion}
            [/生成的评审内容]
        """).strip()

        # 组合成完整的 prompt
        user_prompt = textwrap.dedent(f"""
            [原始任务]
            {query}
            [/原始任务]

            {review_content}

            请根据上述评审内容给出质量评分。
        """).strip()

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _extract_rating(self, response: str) -> Optional[float]:
        """
        从模型响应中提取评分

        支持多种格式：
        - Overall Quality: 8.5
        - **Overall Quality:** 8.5
        - Rating: 8.5
        - rating: 8.5
        - 评分：8.5
        """
        # 模式 1: "**Overall Quality:** X.X" 或 "Overall Quality: X.X" (新格式)
        # 注意：10 要放在前面，否则会先匹配到 1
        pattern1 = r'\*{0,2}Overall Quality:\*{0,2}\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern1, response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # 模式 2: "Rating: X.X" 或 "rating: X.X" (旧格式)
        pattern2 = r'[Rr]ating:\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern2, response)
        if match:
            return float(match.group(1))

        # 模式 3: "评分：X.X" 或 "分数：X.X"
        pattern3 = r'[评分分数][：:]\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern3, response)
        if match:
            return float(match.group(1))

        # 模式 4: 直接提取最后出现的 0-10 之间的数字
        # 匹配 10 或 0-9 加可选小数
        pattern4 = r'(?:^|[\s\n])(10(?:\.0)?|[0-9](?:\.[0-9])?)(?:[\s\n]|$)'
        matches = re.findall(pattern4, response)
        if matches:
            # 取最后一个匹配
            last_score = float(matches[-1])
            if 0 <= last_score <= 10:
                return last_score

        # 如果都没找到，尝试在最后一行查找
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # 尝试提取数字，优先匹配 10
            numbers = re.findall(r'(?:^|[^0-9.])(10(?:\.0)?)(?:[^0-9.]|$)|([0-9](?:\.[0-9])?)', last_line)
            if numbers:
                # 优先使用 10
                for num_tuple in numbers:
                    for num in num_tuple:
                        if num:
                            score = float(num)
                            if 0 <= score <= 10:
                                return score

        return None


# 创建插件实例的工厂函数
def create_review_genrm_plugin(
    alpha: float = 1.0,
    format_weight: float = 1.0,
    **kwargs
):
    """
    创建 ReviewGenRMPlugin 实例的工厂函数

    Args:
        alpha: RM 分数权重
        format_weight: 格式分数权重

    Returns:
        ReviewGenRMPlugin 实例
    """
    return ReviewGenRMPlugin(
        alpha=alpha,
        format_weight=format_weight,
        **kwargs
    )


# 导出插件实例（用于 --external_plugins）
# 注意：这个实例会在 Swift 训练时被重新初始化
# 这里只是为了类型检查和文档
def get_review_genrm_plugin(model=None, template=None, **kwargs):
    """获取插件实例的函数"""
    return ReviewGenRMPlugin(model=model, template=template, **kwargs)
