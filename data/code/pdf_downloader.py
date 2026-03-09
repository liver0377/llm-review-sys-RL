#!/usr/bin/env python3
"""
OpenReview PDF并发下载工具
支持多线程并发下载、进度条显示、全局限流
"""

import os
import json
import requests
import time
import argparse
import threading
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class RateLimiter:
    """全局限流器（线程安全）"""

    def __init__(self, min_delay=2.0):
        """
        初始化限流器

        Args:
            min_delay: 两次请求之间的最小间隔（秒）
        """
        self.min_delay = min_delay
        self.last_request_time = 0
        self.lock = threading.Lock()

    def acquire(self):
        """获取请求许可（阻塞到可以发送请求）"""
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


def download_single_pdf(args, rate_limiter):
    """
    下载单个PDF（线程安全）

    Args:
        args: (pdf_url, output_path, pdf_filename, paper_title, conference)
        rate_limiter: RateLimiter实例

    Returns:
        (success: bool, pdf_filename: str, error_msg: str or None)
    """
    pdf_url, output_path, pdf_filename, paper_title, conference = args

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # 全局限流
            rate_limiter.acquire()

            # 下载PDF
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Accept": "application/pdf",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://openreview.net/",
            }

            response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存文件
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return (True, pdf_filename, None)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # 请求过多，指数退避
                wait_time = 10 * (2**attempt)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    return (False, pdf_filename, f"HTTP 429: Too many requests")
            else:
                if attempt == max_retries - 1:
                    return (False, pdf_filename, f"HTTP {e.response.status_code}")
                else:
                    time.sleep(retry_delay)

        except Exception as e:
            if attempt == max_retries - 1:
                return (False, pdf_filename, str(e))
            else:
                time.sleep(retry_delay)

    return (False, pdf_filename, "Max retries exceeded")


def collect_papers_from_data(
    data_dir, pdf_dir, target_conferences=None, skip_existing=False
):
    """
    从data目录收集所有论文的PDF链接

    Args:
        data_dir: 数据目录（filtered_data或raw_data）
        pdf_dir: PDF保存目录
        target_conferences: 指定的会议列表
        skip_existing: 是否跳过已存在的PDF

    Returns:
        [(pdf_url, output_path, filename, title, conference), ...]
    """
    papers_list = []
    data_path = Path(data_dir)

    # 查找所有results.json
    if target_conferences:
        conference_dirs = [data_path / conf for conf in target_conferences]
    else:
        conference_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    for conf_dir in conference_dirs:
        results_file = conf_dir / "results.json"

        if not results_file.exists():
            continue

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                venues = json.load(f)
        except Exception as e:
            print(f"⚠️  跳过 {conf_dir.name}: 无法读取results.json - {str(e)}")
            continue

        conf_pdf_dir = Path(pdf_dir) / conf_dir.name

        for venue in venues:
            for paper in venue.get("papers", []):
                if "pdf_url" not in paper or not paper["pdf_url"]:
                    continue

                pdf_url = paper["pdf_url"]

                # 提取paper ID
                if "id=" in pdf_url:
                    paper_id = pdf_url.split("id=")[-1]
                else:
                    paper_id = f"paper_{hash(paper.get('title', ''))}"

                pdf_filename = f"{paper_id}.pdf"
                output_path = conf_pdf_dir / pdf_filename

                # 检查是否已存在
                if skip_existing and output_path.exists():
                    continue

                papers_list.append(
                    (
                        pdf_url,
                        str(output_path),
                        pdf_filename,
                        paper.get("title", "Unknown"),
                        conf_dir.name,
                    )
                )

    return papers_list


def download_pdfs_concurrent(papers_list, max_workers=5, rate_limiter=None):
    """
    并发下载PDF

    Args:
        papers_list: [(pdf_url, output_path, filename, title, conf), ...]
        max_workers: 最大并发数
        rate_limiter: 限流器实例

    Returns:
        (success_count, failed_count)
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter(min_delay=2.0)

    success_count = 0
    failed_count = 0
    progress_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(download_single_pdf, paper, rate_limiter): paper
            for paper in papers_list
        }

        # 使用tqdm显示进度
        with tqdm(
            total=len(futures), desc="Downloading PDFs", unit="file", ncols=120
        ) as pbar:
            for future in as_completed(futures):
                try:
                    success, filename, error = future.result()

                    with progress_lock:
                        if success:
                            success_count += 1
                        else:
                            failed_count += 1
                            pbar.write(f"❌ Failed: {filename} - {error}")

                        # 更新进度条后缀
                        pbar.set_postfix_str(f"✅ {success_count} | ❌ {failed_count}")
                        pbar.update(1)

                except Exception as e:
                    with progress_lock:
                        failed_count += 1
                        pbar.write(f"❌ Exception: {str(e)}")
                        pbar.update(1)

    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="并发下载OpenReview PDF文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置（5个并发，filtered_data）
  python pdf_downloader.py
  
  # 指定10个并发
  python pdf_downloader.py --max-workers 10
  
  # 只下载NeurIPS的论文
  python pdf_downloader.py --conferences NeurIPS
  
  # 使用原始数据
  python pdf_downloader.py --data-dir raw_data
  
  # 跳过已存在的PDF
  python pdf_downloader.py --skip-existing
  
  # 自定义限流（1秒间隔）
  python pdf_downloader.py --delay 1.0
        """,
    )

    # 参数配置
    parser.add_argument(
        "--data-dir", default="filtered_data", help="数据源目录（默认: filtered_data）"
    )
    parser.add_argument("--pdf-dir", default="pdfs", help="PDF保存目录（默认: pdfs）")
    parser.add_argument(
        "--conferences", nargs="+", help="指定会议列表（默认: 所有会议）"
    )
    parser.add_argument(
        "--max-workers", type=int, default=5, help="最大并发下载数（默认: 5）"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="请求间隔秒数（默认: 2.0）"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="跳过已存在的PDF（默认：重新下载）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="试运行模式，只扫描不下载"
    )

    args = parser.parse_args()

    # 验证参数
    if args.max_workers < 1:
        print("❌ 错误: max-workers 必须大于等于 1")
        return

    if args.delay < 0:
        print("❌ 错误: delay 必须大于等于 0")
        return

    # 打印配置
    print("=" * 70)
    print("PDF并发下载工具")
    print("=" * 70)
    print("\n配置:")
    print(f"  数据源: {args.data_dir}")
    print(f"  保存目录: {args.pdf_dir}")
    print(f"  最大并发: {args.max_workers} 线程")
    print(f"  请求间隔: {args.delay} 秒")
    print(f"  跳过已存在: {'是' if args.skip_existing else '否'}")
    if args.conferences:
        print(f"  指定会议: {', '.join(args.conferences)}")
    else:
        print(f"  指定会议: 所有会议")
    print()

    # 验证数据目录
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ 错误: 数据目录不存在: {data_path}")
        return

    # 收集论文链接
    print("📂 扫描数据...")
    papers_list = collect_papers_from_data(
        args.data_dir, args.pdf_dir, args.conferences, args.skip_existing
    )

    if not papers_list:
        print("⚠️  没有找到需要下载的论文")
        print("   提示: 使用 --skip-existing 跳过已下载的论文")
        return

    print(f"📊 总计: {len(papers_list):,} 篇论文待下载")

    # 按会议统计
    conf_stats = {}
    for _, _, _, _, conf in papers_list:
        conf_stats[conf] = conf_stats.get(conf, 0) + 1

    for conf, count in sorted(conf_stats.items()):
        print(f"  ✅ {conf}: {count:,} 篇")

    print()

    # 试运行模式
    if args.dry_run:
        print("=" * 70)
        print("试运行模式 - 只扫描不下载")
        print("=" * 70)
        print(f"📊 总计: {len(papers_list):,} 篇论文将被下载")
        print(f"   使用 --max-workers {args.max_workers} 并发下载")
        print(f"   请求间隔 {args.delay} 秒")
        print()
        print("💡 去掉 --dry-run 参数以开始实际下载")
        print("=" * 70)
        return

    # 创建限流器
    rate_limiter = RateLimiter(min_delay=args.delay)

    # 开始并发下载
    print("🚀 开始并发下载...")
    print(f"[{time.strftime('%H:%M:%S')}] 下载开始")
    start_time = time.time()

    success_count, failed_count = download_pdfs_concurrent(
        papers_list, max_workers=args.max_workers, rate_limiter=rate_limiter
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印结果
    print()
    print("=" * 70)
    print("下载完成！")
    print("=" * 70)
    print(f"成功下载: {success_count:,} 篇")
    print(f"下载失败: {failed_count:,} 篇")
    print(f"总计: {success_count + failed_count:,} 篇")

    if success_count + failed_count > 0:
        success_rate = success_count / (success_count + failed_count) * 100
        print(f"成功率: {success_rate:.1f}%")

    print(f"总耗时: {elapsed_time / 60:.1f} 分钟 ({elapsed_time:.0f} 秒)")

    if elapsed_time > 0:
        speed = (success_count + failed_count) / elapsed_time
        print(f"平均速度: {speed:.2f} file/s")

    print(f"保存位置: {args.pdf_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
