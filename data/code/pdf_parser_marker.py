import os
import gc
import json
import sys
import hashlib
import subprocess
import multiprocessing as mp
import queue
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()


# -----------------------------
# nohup 友好：强制 flush
# -----------------------------
class FlushFile:
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.f.flush()

    def isatty(self):
        return hasattr(self.f, "isatty") and self.f.isatty()


DEBUG = os.getenv("MARKER_DEBUG", "false").lower() == "true"


# -----------------------------
# ID / metadata utils
# -----------------------------
def stable_id_from_title(title: str) -> str:
    if not title:
        title = "unknown"
    return hashlib.sha1(title.encode("utf-8")).hexdigest()


def extract_paper_id_from_url(pdf_url: str, title: str) -> str:
    try:
        parsed = urlparse(pdf_url)
        qs = parse_qs(parsed.query)
        pid = qs.get("id", [None])[0]
        if pid:
            pid = "".join(c for c in str(pid) if c.isalnum() or c in ("-", "_"))
            if pid:
                return pid
    except Exception:
        pass
    return f"paper_{stable_id_from_title(title)}"


# -----------------------------
# GPU utils
# -----------------------------
def _parse_visible_devices() -> list[str] | None:
    """
    返回 CUDA_VISIBLE_DEVICES 中的原始列表（可能是数字或 UUID）。
    如果没设置，返回 None。
    """
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd.strip() == "":
        return None
    return [x.strip() for x in cvd.split(",") if x.strip() != ""]


def get_num_gpus() -> int:
    """
    不 import torch 的情况下获取 GPU 数量。
    优先 CUDA_VISIBLE_DEVICES，其次 nvidia-smi -L。
    """
    cvd_list = _parse_visible_devices()
    if cvd_list is not None:
        return len(cvd_list)

    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        return len(lines)
    except Exception:
        return 0


# -----------------------------
# output clean
# -----------------------------
def clean_output_dir(output_dir: Path, force: bool = False):
    if not output_dir.exists():
        return

    if not force:
        clean_env = os.getenv("MARKER_CLEAN_OUTPUT", "false").lower() == "true"
        if not clean_env:
            print("ℹ️  保留现有输出文件（跳过清空）", flush=True)
            return

    print(f"🧹 清空输出目录: {output_dir}", flush=True)
    total_removed = 0
    import shutil

    for conf_dir in output_dir.iterdir():
        if not conf_dir.is_dir():
            continue
        for paper_dir in conf_dir.iterdir():
            if paper_dir.is_dir():
                try:
                    shutil.rmtree(paper_dir)
                    total_removed += 1
                except Exception as e:
                    print(f"⚠️  删除失败 {paper_dir}: {e}", flush=True)

    print(f"✅ 已清空 {total_removed} 个论文目录\n", flush=True)


# -----------------------------
# collect tasks
# -----------------------------
def collect_papers_from_filtered_data(filtered_data_dir, pdfs_dir, output_root_dir, target_conferences=None):
    papers_list = []
    filtered_data_path = Path(filtered_data_dir)
    missing_pdfs = []

    if target_conferences:
        conference_dirs = [filtered_data_path / conf for conf in target_conferences]
    else:
        conference_dirs = [d for d in filtered_data_path.iterdir() if d.is_dir()]

    for conf_dir in conference_dirs:
        results_file = conf_dir / "results.json"
        if not results_file.exists():
            print(f"⚠️  跳过 {conf_dir.name}: results.json 不存在", flush=True)
            continue

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                venues = json.load(f)
        except Exception as e:
            print(f"⚠️  跳过 {conf_dir.name}: 无法读取 results.json - {e}", flush=True)
            continue

        conf_pdf_dir = Path(pdfs_dir) / conf_dir.name
        conf_output_dir = Path(output_root_dir) / conf_dir.name
        conf_output_dir.mkdir(parents=True, exist_ok=True)

        paper_count = 0
        for venue in venues:
            for paper in venue.get("papers", []):
                pdf_url = paper.get("pdf_url")
                if not pdf_url:
                    continue
                title = paper.get("title", "Unknown")
                paper_id = extract_paper_id_from_url(pdf_url, title)

                pdf_path = conf_pdf_dir / f"{paper_id}.pdf"
                if pdf_path.exists():
                    paper_output_dir = conf_output_dir / paper_id
                    papers_list.append((str(pdf_path), str(paper_output_dir), conf_dir.name, title, paper_id))
                    paper_count += 1
                else:
                    missing_pdfs.append(str(pdf_path))

        print(f"✓ {conf_dir.name}: 找到 {paper_count} 个论文", flush=True)

    if missing_pdfs:
        print(f"\n⚠️  {len(missing_pdfs)} 个 PDF 文件不存在（已跳过）:", flush=True)
        for p in missing_pdfs[:5]:
            print(f"   - {p}", flush=True)
        if len(missing_pdfs) > 5:
            print(f"   ... 还有 {len(missing_pdfs) - 5} 个", flush=True)

    return papers_list


# -----------------------------
# fast page count (prefer pdfium2; fallback to pypdf)
# -----------------------------
@lru_cache(maxsize=32768)
def get_num_pages_fast(pdf_path: str) -> int:
    # 1) pdfium (快且 marker 依赖里就有)
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_path)
        n = len(doc)
        doc.close()
        return int(n)
    except Exception:
        pass

    # 2) fallback to pypdf (慢一些，且可能打印 warning)
    try:
        from pypdf import PdfReader
        r = PdfReader(pdf_path)
        return len(r.pages)
    except Exception:
        return -1


# -----------------------------
# worker
# -----------------------------
def worker_loop(worker_id: int, gpu_visible_value: str, task_q: mp.Queue, result_q: mp.Queue, base_cli_options: dict):
    """
    一个 worker 绑定一张 GPU（通过 CUDA_VISIBLE_DEVICES 设为单个值）。
    关键优化：model + converter 初始化只做一次，循环里只做 converter(pdf)。
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_visible_value)

        import torch
        from marker.config.parser import ConfigParser
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.renderers.markdown import MarkdownOutput

        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # ---- 1) 模型只加载一次（每卡一份）
        model_refs = create_model_dict()

        # ---- 2) ConfigParser / converter 只构建一次
        # output_dir 只是 parser 必要字段；我们自己写 md，所以这里用 dummy
        parser_options = dict(base_cli_options)
        parser_options["output_dir"] = parser_options.get("output_dir", ".")

        config_parser = ConfigParser(parser_options)
        config_dict = config_parser.generate_config_dict()
        config_dict["disable_tqdm"] = True
        config_dict["dtype"] = "float16"

        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_dict,
            artifact_dict=model_refs,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        processed = 0

        while True:
            item = task_q.get()
            if item is None:
                break

            pdf_path, output_dir, paper_title = item
            try:
                os.makedirs(output_dir, exist_ok=True)
                base_name = Path(pdf_path).stem
                md_path = Path(output_dir) / f"{base_name}.md"

                # O(1) 跳过
                if base_cli_options.get("skip_existing", True) and md_path.exists():
                    result_q.put((True, paper_title))
                    continue

                # ---- 3) 每个 PDF 动态设置前 20 页（list[int]），并裁剪
                n_pages = get_num_pages_fast(pdf_path)
                if n_pages and n_pages > 0:
                    config_dict["page_range"] = list(range(min(20, n_pages)))
                else:
                    # 页数读不到：不设 page_range，避免越界；会处理全量（少数坏 PDF）
                    config_dict.pop("page_range", None)

                # ---- 4) 转换
                rendered = converter(pdf_path)

                if isinstance(rendered, MarkdownOutput):
                    markdown_text = rendered.markdown
                else:
                    text, ext, images = text_from_rendered(rendered)
                    markdown_text = text

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(markdown_text)

                del rendered
                processed += 1

                # 不要每篇 gc/empty_cache（非常拖慢）
                if processed % 200 == 0:
                    gc.collect()

                result_q.put((True, paper_title))

            except RuntimeError as e:
                # OOM：清一次 cache 重试一次
                msg = str(e).lower()
                if "out of memory" in msg or "cuda out of memory" in msg:
                    print(f"⚠️  [Worker {worker_id} GPU {gpu_visible_value}] OOM on '{paper_title[:60]}', retry once...", flush=True)
                    try:
                        torch.cuda.empty_cache()
                        gc.collect()

                        # retry once
                        n_pages = get_num_pages_fast(pdf_path)
                        if n_pages and n_pages > 0:
                            config_dict["page_range"] = list(range(min(20, n_pages)))
                        else:
                            config_dict.pop("page_range", None)

                        rendered = converter(pdf_path)
                        if isinstance(rendered, MarkdownOutput):
                            markdown_text = rendered.markdown
                        else:
                            text, ext, images = text_from_rendered(rendered)
                            markdown_text = text

                        with open(md_path, "w", encoding="utf-8") as f:
                            f.write(markdown_text)

                        del rendered
                        result_q.put((True, paper_title))
                        continue
                    except Exception as e2:
                        print(f"❌ [Worker {worker_id} GPU {gpu_visible_value}] 解析失败(OOM after retry): {paper_title[:60]} - {e2}", flush=True)
                        result_q.put((False, paper_title))
                        continue

                print(f"❌ [Worker {worker_id} GPU {gpu_visible_value}] 解析失败: {paper_title[:60]} - {e}", flush=True)
                result_q.put((False, paper_title))

            except Exception as e:
                print(f"❌ [Worker {worker_id} GPU {gpu_visible_value}] 解析失败: {paper_title[:60]} - {e}", flush=True)
                result_q.put((False, paper_title))

        # cleanup
        del converter, model_refs
        gc.collect()

    except Exception as e:
        print(f"❌ worker_init_failed (Worker {worker_id}, GPU {gpu_visible_value}): {e}", flush=True)
        result_q.put((False, f"worker_init_failed_gpu_{gpu_visible_value}"))


# -----------------------------
# master
# -----------------------------
def parse_pdfs(
    filtered_data_dir,
    pdfs_dir,
    output_root_dir,
    target_conferences=None,
    use_llm=False,
    desired_workers_per_gpu=1,  # 8×A100：建议先固定 1
):
    num_gpus = get_num_gpus()
    if num_gpus <= 0:
        print("❌ 未检测到 GPU", flush=True)
        return
    print(f"✅ 检测到 {num_gpus} 个 GPU", flush=True)

    output_dir_path = Path(output_root_dir)
    clean_output_dir(output_dir_path, force=False)

    # ---- 关键配置（按你需求）
    cli_options = {
        "output_dir": output_root_dir,          # dummy
        "output_format": "markdown",
        "skip_existing": True,

        # ✅ 不用 disable_multiprocessing（它会把 pdftext_workers 压成 1）
        "pdftext_workers": 12,                   # 先用 4；CPU 很多的话可以试 8
        "disable_image_extraction": True,       # ✅ 官方支持项

        # page_range 这里不再传字符串；worker 内对每篇 PDF 设置 list[int]
        # "page_range": "0-19",
    }
    # cli_options["processors"] = ",".join([
    #     "marker.processors.order.OrderProcessor",
    #     "marker.processors.block_relabel.BlockRelabelProcessor",
    #     "marker.processors.line_merge.LineMergeProcessor",
    #     "marker.processors.list.ListProcessor",
    #     "marker.processors.sectionheader.SectionHeaderProcessor",
    #     "marker.processors.code.CodeProcessor",
    #     "marker.processors.text.TextProcessor",
    #     "marker.processors.blank_page.BlankPageProcessor",
    # ])

    if use_llm:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        api_base = os.getenv("DASHSCOPE_API_BASE")
        if not api_key:
            print("❌ DASHSCOPE_API_KEY 未找到", flush=True)
            return
        print("✅ LLM 增强: 使用 DASHSCOPE API", flush=True)
        cli_options.update(
            {
                "use_llm": True,
                "llm_service": "marker.services.openai.OpenAIService",
                "openai_api_key": api_key,
                "openai_base_url": api_base,
                "openai_model": "qwen-max",
            }
        )

    print(f"\n📂 扫描 filtered_data 目录: {filtered_data_dir}", flush=True)
    papers_list = collect_papers_from_filtered_data(filtered_data_dir, pdfs_dir, output_root_dir, target_conferences)
    if not papers_list:
        print("\nℹ️  没有需要处理的 PDF", flush=True)
        return

    tasks = [(pdf_path, output_dir, title) for (pdf_path, output_dir, _conf, title, _paper_id) in papers_list]
    total = len(tasks)
    print(f"\n📊 共 {total} 个 PDF 需要处理", flush=True)

    # mp spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context("spawn")

    # ---- 设备映射：如果用户设置了 CUDA_VISIBLE_DEVICES=4,5,6,7 这种，必须按列表映射
    visible_list = _parse_visible_devices()
    if visible_list is None:
        visible_list = [str(i) for i in range(num_gpus)]
    else:
        # 这里 num_gpus == len(visible_list)
        pass

    workers_per_gpu = max(1, int(desired_workers_per_gpu))
    total_workers = min(len(tasks), len(visible_list) * workers_per_gpu)

    print(f"🚀 启动 {total_workers} 个 Worker（每GPU {workers_per_gpu} 个进程）", flush=True)

    task_q = ctx.Queue(maxsize=total_workers * 4)
    result_q = ctx.Queue()
    workers = []

    for wid in range(total_workers):
        gpu_visible_value = visible_list[wid % len(visible_list)]
        p = ctx.Process(target=worker_loop, args=(wid, gpu_visible_value, task_q, result_q, cli_options))
        p.start()
        workers.append(p)

    # ---- 关键：边投喂边消费，避免 put 卡死
    task_iter = iter(tasks)
    prefill = total_workers * 4
    sent = 0
    for _ in range(prefill):
        try:
            task_q.put(next(task_iter))
            sent += 1
        except StopIteration:
            break

    print(f"\n{'=' * 80}", flush=True)
    print(f"⏳ 开始处理 {total} 个 PDF（只处理前20页；只保存md）", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    done = 0
    success = 0
    fail = 0
    start_time = time.time()
    last_print_time = start_time
    last_title = ""

    while done < total:
        got = False
        try:
            ok, title = result_q.get(timeout=5)
            got = True
        except queue.Empty:
            got = False

        if got:
            done += 1
            last_title = title
            if ok:
                success += 1
            else:
                fail += 1

            # 每完成一个任务，再投喂一个任务
            try:
                task_q.put(next(task_iter))
                sent += 1
            except StopIteration:
                pass

        alive = sum(1 for p in workers if p.is_alive())
        if alive == 0 and done < total:
            print("\n❌ 错误: 所有 Worker 已退出，但任务未完成。", flush=True)
            break

        now = time.time()
        if (now - last_print_time) >= 10 or done == total:
            elapsed = now - start_time
            speed = done / max(elapsed, 1e-6)
            remaining = total - done
            eta_sec = remaining / max(speed, 1e-6)

            def fmt_hm(seconds: float) -> str:
                m = int(seconds // 60)
                h = m // 60
                m = m % 60
                return f"{h}h {m}m" if h > 0 else f"{m}m"

            print(
                f"[状态] 进度: {done}/{total} | 成功: {success} | 失败: {fail} | "
                f"速度: {speed:.2f} pdf/s | 已用: {fmt_hm(elapsed)} | 预计剩余: {fmt_hm(eta_sec)} | "
                f"workers_alive={alive}/{len(workers)} | last='{last_title[:60]}'",
                flush=True,
            )
            last_print_time = now

    # 发送退出信号
    for _ in workers:
        task_q.put(None)

    for p in workers:
        p.join(timeout=30)

    total_time = time.time() - start_time
    avg_speed = done / total_time if total_time > 0 else 0.0

    print(f"\n{'=' * 80}", flush=True)
    print("✅ 全部处理完成！", flush=True)
    print(f"   完成: {done}/{total}", flush=True)
    print(f"   成功: {success}", flush=True)
    if fail > 0:
        print(f"   失败: {fail}", flush=True)
    print(f"   总耗时: {total_time/60:.1f} 分钟", flush=True)
    print(f"   平均速度: {avg_speed:.2f} pdf/s", flush=True)
    print(f"   输出目录: {output_root_dir}", flush=True)
    print(f"{'=' * 80}", flush=True)


def main():
    if not sys.stdout.isatty():
        sys.stdout = FlushFile(sys.stdout)
        sys.stderr = FlushFile(sys.stderr)
        print("📝 Non-interactive mode detected: force flushing stdout/stderr", flush=True)

    base_dir = os.path.dirname(__file__)
    filtered_data_root = os.path.join(base_dir, "filtered_data")
    pdfs_root = os.path.join(base_dir, "pdfs")
    output_root = os.path.join(base_dir, "parsed_pdf_texts_marker")
    TARGET_CONFERENCES = ["AAAI", "NeurIPS", "ICML"]

    USE_LLM = os.getenv("MARKER_USE_LLM", "false").lower() == "true"

    print("=" * 70, flush=True)
    print("📄 PDF 转 Markdown - Marker (FAST: reuse converter per GPU)", flush=True)
    print("=" * 70, flush=True)
    print(f"数据目录: {filtered_data_root}", flush=True)
    print(f"PDF 目录: {pdfs_root}", flush=True)
    print(f"输出目录: {output_root}", flush=True)
    print(f"目标会议: {TARGET_CONFERENCES}", flush=True)
    print(f"LLM 增强: {'✅' if USE_LLM else '❌'}", flush=True)
    print("策略: 每GPU 1进程 + converter复用 + pdftext_workers=4 + disable_image_extraction + 前20页裁剪", flush=True)
    print("=" * 70, flush=True)

    parse_pdfs(
        filtered_data_root,
        pdfs_root,
        output_root,
        target_conferences=TARGET_CONFERENCES,
        use_llm=USE_LLM,
        desired_workers_per_gpu=1,
    )


if __name__ == "__main__":
    main()

