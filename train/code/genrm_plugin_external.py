"""
外部部署版本的生成式奖励模型插件
使用 OpenAI Client 调用外部部署的 Reward Model 服务
"""

import re
import textwrap
from typing import List, Dict, Optional

from swift.rewards.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()

# 尝试导入 openai
try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI library is required for external deployment. "
        "Please install it: pip install openai"
    )


class ReviewGenRMPluginExternal(DefaultRMPlugin):
    """
    外部部署版本的生成式奖励模型插件

    使用 OpenAI Client 调用外部部署的 Reward Model 服务（通过 vLLM serve）
    对生成的评审进行质量评估，并组合格式分数和 RM 分数作为最终奖励。

    Args:
        model: 模型路径（在外部部署中不使用，但保持接口兼容）
        template: 模板（不使用）
        base_url: API 服务地址，例如 "http://127.0.0.1:8000/v1"
        api_key: API 密钥，vLLM 通常使用 "EMPTY"
        alpha (float): RM 分数的权重 (默认: 1.0)
        format_weight (float): 格式分数的权重 (默认: 1.0)
        max_tokens (int): 生成的最大 token 数 (默认: 128，外部部署更快)
        temperature (float): 生成温度 (默认: 0.1)
        timeout (int): 请求超时时间（秒）(默认: 120)
    """

    def __init__(
        self,
        model=None,
        template=None,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        alpha: float = 1.0,
        format_weight: float = 1.0,
        max_tokens: int = 128,
        temperature: float = 0.1,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model, template, **kwargs)
        self.alpha = alpha
        self.format_weight = format_weight
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = base_url
        self.api_key = api_key

        # 导入格式评分函数
        self._import_modules()

        # 设置评审打分的 system prompt（简化版）
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

        # 初始化 OpenAI Client
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            logger.info(f"Initialized OpenAI client for external RM: {base_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        # 验证模型名称
        self._verify_model()

    def _import_modules(self):
        """动态导入依赖模块"""
        import sys
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        from reward_function import compute_format_score
        self.compute_format_score = compute_format_score

    def _verify_model(self):
        """验证模型名称"""
        try:
            models = self.client.models.list()
            if models.data:
                self.model_name = models.data[0].id
                logger.info(f"Verified model name: {self.model_name}")
            else:
                logger.warning("No models found in service, using default")
                self.model_name = "qwen3.5-35b-a3b"
        except Exception as e:
            logger.warning(f"Failed to verify model name: {e}")
            self.model_name = "qwen3.5-35b-a3b"

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

            # 使用外部 RM 获取质量分数 (0-10)
            rm_score = self._get_genrm_score_external(messages, completion)

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

    def _get_genrm_score_external(self, messages: List[Dict], completion: str) -> float:
        """
        使用外部部署的 Reward Model 获取质量分数

        Args:
            messages: 原始对话消息
            completion: 生成的评审内容

        Returns:
            float: 质量分数 (0-10)
        """
        # 构造评审打分的 prompt
        rm_messages = self._prepare_rm_prompt(messages, completion)

        try:
            # 调用外部 RM 服务
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=rm_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )

            # 提取评分
            score = self._extract_rating(response.choices[0].message.content.strip())
            return score if score is not None else 5.0

        except Exception as e:
            logger.error(f"Error calling external RM service: {e}")
            return 5.0  # 默认中等分数

    def _prepare_rm_prompt(self, original_messages: List[Dict], completion: str) -> List[Dict]:
        """
        准备给外部 RM 的 prompt

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

        # 构造简化的 prompt
        user_prompt = textwrap.dedent(f"""
            <Paper Review Task>
            {query}
            </Paper Review Task>

            <Generated Review>
            {completion}
            </Generated Review>

            Please assess the quality of this review based on its clarity, completeness, accuracy, and constructiveness.
            Provide a brief assessment (2-3 sentences), then give a score in the format:
            **Overall Quality:** X.X (1–10, where 10 is top-tier)
        """).strip()

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _extract_rating(self, response: str) -> Optional[float]:
        """
        从模型响应中提取评分

        Args:
            response: 模型的输出字符串

        Returns:
            float: 提取的评分分数

        支持多种格式：
        - Overall Quality: 8.5
        - **Overall Quality:** 8.5
        - 评分：8.5
        """
        # 模式 1: "**Overall Quality:** X.X" 或 "Overall Quality: X.X"
        # 注意：10 要放在前面，否则会先匹配到 1
        pattern1 = r'\*{0,2}Overall Quality:\*{0,2}\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern1, response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # 模式 2: "Rating: X.X" 或 "rating: X.X" (向后兼容)
        pattern2 = r'[Rr]ating:\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern2, response)
        if match:
            return float(match.group(1))

        # 模式 3: "评分：X.X" 或 "分数：X.X"
        pattern3 = r'[评分分数][：:]\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern3, response)
        if match:
            return float(match.group(1))

        # 如果都没找到，尝试在最后一行查找
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # 尝试提取数字
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


# 工厂函数
def get_review_genrm_plugin_external(
    model=None,
    template=None,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    **kwargs
):
    """
    获取外部部署版本插件的工厂函数

    Args:
        model: 模型路径（不使用，但保持接口兼容）
        template: 模板（不使用）
        base_url: API 服务地址
        api_key: API 密钥
        **kwargs: 其他参数

    Returns:
        ReviewGenRMPluginExternal 实例
    """
    return ReviewGenRMPluginExternal(
        model=model,
        template=template,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )


# 导出实例（用于 --external_plugins）
review_genrm_external = ReviewGenRMPluginExternal()
