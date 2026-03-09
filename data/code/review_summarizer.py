from openai import OpenAI
import json
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple

load_dotenv()


class ReviewAggregator:
    def __init__(self, api_key_name: str, max_concurrent: int = 10):
        api_key = os.getenv(api_key_name)
        if not api_key:
            raise ValueError(
                "API key not found. Please set DEEPSEEK_API_KEY environment variable."
            )
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.base_url = os.getenv("DASHSCOPE_API_BASE", "https://api.deepseek.com/v1")

        # 保留同步客户端备用
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    async def _api_call(self, messages: List[Dict]) -> Optional[str]:
        """异步调用 DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": "deepseek-v3.2", "messages": messages, "temperature": 0.2}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            return None

    async def aggregate_reviews_async(self, reviews: List[str]) -> str:
        """异步聚合评审（支持分批处理）"""
        if not reviews:
            return ""

        total_chars = sum(len(review) for review in reviews)
        estimated_tokens = total_chars // 4

        if estimated_tokens > 6000:
            print(f"评论内容过长 (估计 {estimated_tokens} tokens)，将分批处理")

            batches = []
            current_batch = []
            current_chars = 0

            for review in reviews:
                review_chars = len(review)

                if current_chars + review_chars > 24000:
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [review]
                    current_chars = review_chars
                else:
                    current_batch.append(review)
                    current_chars += review_chars

            if current_batch:
                batches.append(current_batch)

            print(f"将评论分成 {len(batches)} 批处理")

            summaries = []
            for i, batch in enumerate(batches):
                print(f"处理第 {i + 1}/{len(batches)} 批评论...")
                batch_summary = await self._process_single_batch_async(batch)
                if batch_summary:
                    summaries.append(batch_summary)

            if len(summaries) > 1:
                print("合并所有批次的摘要...")
                final_summary = await self._merge_summaries_async(summaries)
                return final_summary if final_summary else summaries[0]
            else:
                return summaries[0] if summaries else ""
        else:
            return await self._process_single_batch_async(reviews)

    async def _process_single_batch_async(self, reviews: List[str]) -> str:
        """异步处理单批次评论"""
        numbered_reviews = "\n\n".join(
            [f"Review {i + 1}: {review}" for i, review in enumerate(reviews)]
        )

        prompt = f"""
        Please generate a concise summary based on the following review(s) with the following three sections:
        1. Key points
        2. Strengths and weaknesses
        3. Suggestions for improvement
        
        Critical Requirements:
        - Strictly maintain the exact core viewpoints of the original review(s)
        - Absolute prohibition on adding any new information
        - Summarize ONLY using the provided review content
        - Preserve verbatim any specific technical suggestions or terms
        - Ensure the summary captures the full depth and nuance of the original reviews
        - Match the original review's academic tone and technical precision
        - Reduce length by 20-30% without losing critical information
        - When making suggestions, write as if reviewers are directly addressing the authors (e.g., "We recommend that the authors improve..." rather than "The reviewers suggest..." or "We should improve...")
        - When discussing the paper in the Key points section, use phrases like "This paper presents..." or "The authors propose..." rather than "We present..."

        Format Requirements:
        - Use "### Key Points", "### Strengths and Weaknesses", and "### Suggestions for Improvement" as section headers
        - In the Strengths and Weaknesses section, write "Strengths:" and "Weaknesses:" on separate lines
        
        Review(s):
        {numbered_reviews}
        """

        result = await self._api_call([{"role": "user", "content": prompt}])
        if result is None:
            return f"处理评论时出错: API call failed"
        return result

    async def _merge_summaries_async(self, summaries: List[str]) -> Optional[str]:
        """异步合并多个批次的摘要"""
        combined_summaries = "\n\n".join(
            [f"Summary {i + 1}:\n{summary}" for i, summary in enumerate(summaries)]
        )

        prompt = f"""
        Please merge the following summaries into a single comprehensive summary with the following three sections:
        1. Key points
        2. Strengths and weaknesses
        3. Suggestions for improvement
        
        Strict Merging Requirements:
        - Preserve the exact core viewpoints from ALL summaries
        - Completely eliminate redundant information
        - Maintain 100% fidelity to the original review contents
        - Ensure the final summary is logically coherent
        - Keep the original academic and technical precision
        - Do NOT introduce any new interpretations or insights
        - When making suggestions, write as if reviewers are directly addressing the authors (e.g., "We recommend that the authors improve..." rather than "The reviewers suggest..." or "We should improve...")
        - When discussing the paper in the Key points section, use phrases like "This paper presents..." or "The authors propose..." rather than "We present..."

        Format Requirements:
        - Use "### Key Points", "### Strengths and Weaknesses", and "### Suggestions for Improvement" as section headers
        - In the Strengths and Weaknesses section, write "Strengths:" and "Weaknesses:" on separate lines
        
        Summaries to merge:
        {combined_summaries}
        """

        result = await self._api_call([{"role": "user", "content": prompt}])
        if result is None:
            error_msg = "合并摘要时出错: API call failed"
            self._log_error(error_msg)
            return "\n\n===== 摘要分割线 =====\n\n".join(summaries)
        return result

    def _log_error(self, error_msg: str):
        """记录错误信息到文件"""
        error_log_path = os.path.join(os.path.dirname(__file__), "error_log.txt")
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {error_msg}\n")

    def _save_result(self, result: Dict, existing_results_path: str, output_dir: str):
        """保存单个结果到文件（线程安全，每个文件唯一）"""
        # 读取现有结果
        if os.path.exists(existing_results_path):
            with open(existing_results_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        else:
            existing_results = []

        # 添加新结果
        existing_results.append(result)

        # 保存 JSON
        with open(existing_results_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)

        # 保存 TXT 文件
        paper_id = result["id"]
        filename = f"{paper_id}_{result['year']}.txt"

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(f"ID: {paper_id}\n")
            f.write(f"Title: {result['title']}\n")
            f.write(f"Conference: {result['conference']}\n")
            f.write(f"Year: {result['year']}\n")
            f.write(f"Number of Reviews: {result['reviews_count']}\n")
            if result["original_ratings"]:
                f.write(
                    f"Original Ratings: {', '.join(map(str, result['original_ratings']))}\n"
                )
            if result["original_confidences"]:
                f.write(
                    f"Original Confidences: {', '.join(map(str, result['original_confidences']))}\n"
                )
            f.write("\nAggregated Review:\n")
            f.write(result["aggregated_review"])

    async def _process_paper_async(
        self, title: str, reviews: List[Dict], paper_id: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """异步处理单篇论文"""
        try:
            review_texts = [r["review_text"] for r in reviews]
            processed_review = await self.aggregate_reviews_async(review_texts)

            if not processed_review or processed_review.startswith("处理评论时出错"):
                return (False, None, title)

            result = {
                "id": paper_id,
                "title": title,
                "reviews_count": len(reviews),
                "conference": reviews[0]["conference"],
                "year": reviews[0]["year"],
                "aggregated_review": processed_review,
                "original_ratings": [
                    r["rating"] for r in reviews if r["rating"] is not None
                ],
                "original_confidences": [
                    r["confidence"] for r in reviews if r["confidence"] is not None
                ],
            }

            return (True, result, None)

        except Exception as e:
            error_msg = f"处理论文 '{title}' 时发生异常: {str(e)}"
            print(error_msg)
            self._log_error(error_msg)
            return (False, None, title)

    async def process_openreview_dataset_async(
        self, dataset_path: str, output_dir: str
    ) -> int:
        """异步处理数据集并聚合评审意见"""

        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return 0

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            print("Dataset is empty.")
            return 0

        print(f"Loaded {len(data)} items from the dataset.")

        # 按论文标题组织评审
        paper_reviews = {}
        paper_ids = {}

        for item in data:
            title = item.get("title")
            paper_id = item.get("id", f"paper_{len(paper_ids) + 1}")
            reviews_array = item.get("reviews", [])

            if title and reviews_array:
                if title not in paper_reviews:
                    paper_reviews[title] = []
                    paper_ids[title] = paper_id

                for review in reviews_array:
                    review_text = review.get("review_text")
                    if review_text:
                        rating = review.get("rating")
                        confidence = review.get("confidence")

                        if isinstance(rating, str) and ":" in rating:
                            try:
                                rating = int(rating.split(":", 1)[0].strip())
                            except ValueError:
                                rating = -1

                        if isinstance(confidence, str) and ":" in confidence:
                            try:
                                confidence = int(confidence.split(":", 1)[0].strip())
                            except ValueError:
                                confidence = -1

                        review_info = {
                            "review_text": review_text,
                            "rating": rating,
                            "confidence": confidence,
                            "year": item.get("year"),
                            "conference": item.get("conference"),
                        }

                        paper_reviews[title].append(review_info)

        if not paper_reviews:
            print("No valid papers found in the dataset.")
            return 0

        print(f"Found {len(paper_reviews)} papers with valid reviews.")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 读取现有的聚合结果
        existing_results_path = os.path.join(output_dir, "aggregated_reviews.json")
        if os.path.exists(existing_results_path):
            with open(existing_results_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        else:
            existing_results = []

        # 获取已处理的标题集合
        titles_in_existing = {result["title"] for result in existing_results}

        # 过滤出需要处理的论文
        papers_to_process = [
            (title, reviews, paper_ids[title])
            for title, reviews in paper_reviews.items()
            if title not in titles_in_existing
        ]

        if not papers_to_process:
            print("所有论文已处理完毕。")
            return 0

        print(
            f"准备处理 {len(papers_to_process)} 篇新论文，并发数: {self.max_concurrent}"
        )

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(title, reviews, paper_id):
            async with semaphore:
                return await self._process_paper_async(title, reviews, paper_id)

        # 创建所有任务
        tasks = [
            process_with_semaphore(title, reviews, paper_id)
            for title, reviews, paper_id in papers_to_process
        ]

        # 并发执行并显示进度
        processed_count = 0
        error_count = 0

        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing papers"
        ):
            success, result, error_title = await coro

            if success and result:
                # 保存结果
                self._save_result(result, existing_results_path, output_dir)
                processed_count += 1
            else:
                error_count += 1

        print(
            f"处理完成，共处理了 {processed_count} 篇新论文，{error_count} 篇处理失败。"
        )
        return processed_count

    # ========== 保留原有同步接口作为备用 ==========

    def aggregate_reviews(self, reviews):
        """使用统一英文prompt处理单条或多条review，处理长评论不截断（同步版本）"""
        if not reviews:
            print("No reviews to process.")
            return ""

        total_chars = sum(len(review) for review in reviews)
        estimated_tokens = total_chars // 4

        if estimated_tokens > 6000:
            print(f"评论内容过长 (估计 {estimated_tokens} tokens)，将分批处理")

            batches = []
            current_batch = []
            current_chars = 0

            for review in reviews:
                review_chars = len(review)

                if current_chars + review_chars > 24000:
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [review]
                    current_chars = review_chars
                else:
                    current_batch.append(review)
                    current_chars += review_chars

            if current_batch:
                batches.append(current_batch)

            print(f"将评论分成 {len(batches)} 批处理")

            summaries = []
            for i, batch in enumerate(batches):
                print(f"处理第 {i + 1}/{len(batches)} 批评论...")
                batch_summary = self._process_single_batch(batch)
                summaries.append(batch_summary)

            if len(summaries) > 1:
                print("合并所有批次的摘要...")
                final_summary = self._merge_summaries(summaries)
                return final_summary
            else:
                return summaries[0]
        else:
            return self._process_single_batch(reviews)

    def _process_single_batch(self, reviews):
        """处理单批次评论（同步版本）"""
        numbered_reviews = "\n\n".join(
            [f"Review {i + 1}: {review}" for i, review in enumerate(reviews)]
        )

        prompt = f"""
        Please generate a concise summary based on the following review(s) with the following three sections:
        1. Key points
        2. Strengths and weaknesses
        3. Suggestions for improvement
        
        Critical Requirements:
        - Strictly maintain the exact core viewpoints of the original review(s)
        - Absolute prohibition on adding any new information
        - Summarize ONLY using the provided review content
        - Preserve verbatim any specific technical suggestions or terms
        - Ensure the summary captures the full depth and nuance of the original reviews
        - Match the original review's academic tone and technical precision
        - Reduce length by 20-30% without losing critical information
        - When making suggestions, write as if reviewers are directly addressing the authors (e.g., "We recommend that the authors improve..." rather than "The reviewers suggest..." or "We should improve...")
        - When discussing the paper in the Key points section, use phrases like "This paper presents..." or "The authors propose..." rather than "We present..."

        Format Requirements:
        - Use "### Key Points", "### Strengths and Weaknesses", and "### Suggestions for Improvement" as section headers
        - In the Strengths and Weaknesses section, write "Strengths:" and "Weaknesses:" on separate lines
        
        Review(s):
        {numbered_reviews}
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            if not response.choices:
                print("API call failed or returned no choices.")
                return ""

            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            return f"处理评论时出错: {str(e)}"

    def _merge_summaries(self, summaries):
        """合并多个批次的摘要（同步版本）"""
        combined_summaries = "\n\n".join(
            [f"Summary {i + 1}:\n{summary}" for i, summary in enumerate(summaries)]
        )

        prompt = f"""
        Please merge the following summaries into a single comprehensive summary with the following three sections:
        1. Key points
        2. Strengths and weaknesses
        3. Suggestions for improvement
        
        Strict Merging Requirements:
        - Preserve the exact core viewpoints from ALL summaries
        - Completely eliminate redundant information
        - Maintain 100% fidelity to the original review contents
        - Ensure the final summary is logically coherent
        - Keep the original academic and technical precision
        - Do NOT introduce any new interpretations or insights
        - When making suggestions, write as if reviewers are directly addressing the authors (e.g., "We recommend that the authors improve..." rather than "The reviewers suggest..." or "We should improve...")
        - When discussing the paper in the Key points section, use phrases like "This paper presents..." or "The authors propose..." rather than "We present..."

        Format Requirements:
        - Use "### Key Points", "### Strengths and Weaknesses", and "### Suggestions for Improvement" as section headers
        - In the Strengths and Weaknesses section, write "Strengths:" and "Weaknesses:" on separate lines
        
        Summaries to merge:
        {combined_summaries}
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            if not response.choices:
                print("API call failed or returned no choices.")
                return ""

            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"合并摘要时出错: {str(e)}"
            print(error_msg)
            self._log_error(error_msg)
            return "\n\n===== 摘要分割线 =====\n\n".join(summaries)

    def process_openreview_dataset(self, dataset_path, output_dir):
        """处理数据集并聚合评审意见（同步版本，兼容旧代码）"""

        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return []

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            print("Dataset is empty.")
            return []

        print(f"Loaded {len(data)} items from the dataset.")

        paper_reviews = {}
        paper_ids = {}

        for item in data:
            title = item.get("title")
            paper_id = item.get("id", f"paper_{len(paper_ids) + 1}")
            reviews_array = item.get("reviews", [])

            if title and reviews_array:
                if title not in paper_reviews:
                    paper_reviews[title] = []
                    paper_ids[title] = paper_id

                for review in reviews_array:
                    review_text = review.get("review_text")
                    if review_text:
                        rating = review.get("rating")
                        confidence = review.get("confidence")

                        if isinstance(rating, str) and ":" in rating:
                            try:
                                rating = int(rating.split(":", 1)[0].strip())
                            except ValueError:
                                rating = -1

                        if isinstance(confidence, str) and ":" in confidence:
                            try:
                                confidence = int(confidence.split(":", 1)[0].strip())
                            except ValueError:
                                confidence = -1

                        review_info = {
                            "review_text": review_text,
                            "rating": rating,
                            "confidence": confidence,
                            "year": item.get("year"),
                            "conference": item.get("conference"),
                        }

                        paper_reviews[title].append(review_info)

        if not paper_reviews:
            print("No valid papers found in the dataset.")
            return []

        print(f"Found {len(paper_reviews)} papers with valid reviews.")

        os.makedirs(output_dir, exist_ok=True)

        existing_results_path = os.path.join(output_dir, "aggregated_reviews.json")
        if os.path.exists(existing_results_path):
            with open(existing_results_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        else:
            existing_results = []

        titles_in_existing = {result["title"] for result in existing_results}

        processed_count = 0
        error_count = 0
        for title, reviews in tqdm(paper_reviews.items(), desc="Processing papers"):
            try:
                if title in titles_in_existing:
                    continue

                print(f"Processing paper: {title}")
                review_texts = [r["review_text"] for r in reviews]
                processed_review = self.aggregate_reviews(review_texts)

                if processed_review.startswith(
                    "处理评论时出错"
                ) or processed_review.startswith("合并摘要时出错"):
                    error_msg = f"处理论文 '{title}' 时出错: {processed_review}"
                    self._log_error(error_msg)
                    error_count += 1
                    continue

                paper_id = paper_ids[title]

                result = {
                    "id": paper_id,
                    "title": title,
                    "reviews_count": len(reviews),
                    "conference": reviews[0]["conference"],
                    "year": reviews[0]["year"],
                    "aggregated_review": processed_review,
                    "original_ratings": [
                        r["rating"] for r in reviews if r["rating"] is not None
                    ],
                    "original_confidences": [
                        r["confidence"] for r in reviews if r["confidence"] is not None
                    ],
                }

                self._save_result(result, existing_results_path, output_dir)
                processed_count += 1

            except Exception as e:
                error_msg = f"处理论文 '{title}' 时发生异常: {str(e)}"
                print(error_msg)
                self._log_error(error_msg)
                error_count += 1

        print(
            f"处理完成，共处理了 {processed_count} 篇新论文，{error_count} 篇处理失败。"
        )
        return processed_count


def main():
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "draft_data", "paper_reviews.json")
    output_dir = os.path.join(base_dir, "aggregated_reviews")

    # 从环境变量读取并发数，默认 10
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "10"))

    aggregator = ReviewAggregator("DASHSCOPE_API_KEY", max_concurrent=max_concurrent)

    print(f"开始处理，并发数: {max_concurrent}")

    # 使用异步版本
    processed_count = asyncio.run(
        aggregator.process_openreview_dataset_async(dataset_path, output_dir)
    )
    print(f"处理完成，共处理了 {processed_count} 篇新论文。")


if __name__ == "__main__":
    main()
