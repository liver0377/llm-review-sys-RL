#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import time
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Precompiled regex patterns
# ----------------------------

# 过滤 Marker/HTML 页码锚点：<span id="page-5-0"></span>
HTML_PAGE_SPAN = re.compile(
    r"""<span\s+[^>]*\bid\s*=\s*["']page-\d+(?:-\d+)*["'][^>]*>\s*</span\s*>""",
    re.IGNORECASE,
)

# 更宽泛：空的 span/a 锚点（避免把有内容的 span 删掉）
HTML_EMPTY_SPAN = re.compile(r"""<span\b[^>]*>\s*</span\s*>""", re.IGNORECASE)
HTML_EMPTY_A = re.compile(r"""<a\b[^>]*>\s*</a\s*>""", re.IGNORECASE)

CUT_HEADERS = re.compile(
    r"^\s{0,3}(#{1,6}\s+|\*\*)(appendix|supplementary|supporting information|references|bibliography|acknowledg(e)?ments?|附录|参考文献|致谢)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

MARKER_TOKENS = re.compile(r"【\d+†[^】]*】")
NUM_CIT = re.compile(r"\[(\d+(\s*[-–]\s*\d+)?)(\s*,\s*\d+(\s*[-–]\s*\d+)?)*\]")
LATEX_CITE_REF = re.compile(r"\\(cite|citet|citep|ref|eqref|autoref)\s*\{[^}]*\}")

MATH_BLOCK_1 = re.compile(r"\$\$[\s\S]*?\$\$", re.MULTILINE)
MATH_BLOCK_2 = re.compile(r"\\\[[\s\S]*?\\\]", re.MULTILINE)
MATH_BLOCK_3 = re.compile(r"\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\}", re.MULTILINE)
MATH_BLOCK_4 = re.compile(r"\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}", re.MULTILINE)

MATH_INLINE_1 = re.compile(r"\\\([\s\S]*?\\\)")
MATH_INLINE_2 = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")

LATEX_FIG = re.compile(r"\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}", re.MULTILINE)
LATEX_TAB = re.compile(r"\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}", re.MULTILINE)
LATEX_TABULAR = re.compile(r"\\begin\{tabular\}[\s\S]*?\\end\{tabular\}", re.MULTILINE)

MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MD_TABLE_BLOCK = re.compile(r"(?:^\s*\|.*\|\s*$\n)+", re.MULTILINE)

FIGTAB_MENTION = re.compile(
    r"\b(Fig\.?|Figure|Table|Eq\.?|Equation)\s*\(?\d+[a-z]?\)?",
    re.IGNORECASE,
)

MULTI_BLANK_LINES = re.compile(r"\n{3,}")
TRAILING_SPACES = re.compile(r"[ \t]+\n")


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CleanOptions:
    remove_inline_math: bool = False
    remove_figtab_mentions: bool = False
    skip_existing: bool = True
    remove_empty_html_anchors: bool = True  # 过滤空 span/a


@dataclass
class FileReport:
    input_path: str
    output_path: str
    ok: bool
    skipped: bool
    error: Optional[str]
    bytes_in: int
    bytes_out: int
    cut_triggered: bool
    removed_html_page_spans: int
    removed_html_empty_spans: int
    removed_html_empty_as: int
    removed_marker_tokens: int
    removed_num_citations: int
    removed_latex_cite_ref: int
    removed_math_blocks: int
    removed_inline_math: int
    removed_figures_tables: int
    removed_md_images: int
    removed_md_tables: int
    removed_figtab_mentions: int
    seconds: float


# ----------------------------
# Helpers
# ----------------------------

def cut_after_headers(text: str) -> Tuple[str, bool]:
    m = CUT_HEADERS.search(text)
    if not m:
        return text, False
    return text[: m.start()].rstrip() + "\n", True


def _count_sub(pattern: re.Pattern, repl: str, text: str) -> Tuple[str, int]:
    new_text, n = pattern.subn(repl, text)
    return new_text, n


# ----------------------------
# Cleaning pipeline
# ----------------------------

def clean_markdown(text: str, opts: CleanOptions) -> Tuple[str, Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "cut_triggered": False,
        "removed_html_page_spans": 0,
        "removed_html_empty_spans": 0,
        "removed_html_empty_as": 0,
        "removed_marker_tokens": 0,
        "removed_num_citations": 0,
        "removed_latex_cite_ref": 0,
        "removed_math_blocks": 0,
        "removed_inline_math": 0,
        "removed_figures_tables": 0,
        "removed_md_images": 0,
        "removed_md_tables": 0,
        "removed_figtab_mentions": 0,
    }

    # 0) Remove marker HTML anchors (do this early so headers become clean)
    text, n = _count_sub(HTML_PAGE_SPAN, "", text)
    stats["removed_html_page_spans"] += n

    if opts.remove_empty_html_anchors:
        text, n = _count_sub(HTML_EMPTY_SPAN, "", text)
        stats["removed_html_empty_spans"] += n
        text, n = _count_sub(HTML_EMPTY_A, "", text)
        stats["removed_html_empty_as"] += n

    # 1) Cut after Appendix/Refs/Ack/Supplementary
    text, cut = cut_after_headers(text)
    stats["cut_triggered"] = cut

    # 2) Remove tokens/citations
    text, n = _count_sub(MARKER_TOKENS, "", text)
    stats["removed_marker_tokens"] += n

    text, n = _count_sub(NUM_CIT, "", text)
    stats["removed_num_citations"] += n

    text, n = _count_sub(LATEX_CITE_REF, "", text)
    stats["removed_latex_cite_ref"] += n

    # 3) Remove figures/tables content
    for p in (LATEX_FIG, LATEX_TAB, LATEX_TABULAR):
        text, n = _count_sub(p, "", text)
        stats["removed_figures_tables"] += n

    text, n = _count_sub(MD_IMAGE, "", text)
    stats["removed_md_images"] += n

    text, n = _count_sub(MD_TABLE_BLOCK, "", text)
    stats["removed_md_tables"] += n

    # 4) Remove math blocks
    for p in (MATH_BLOCK_1, MATH_BLOCK_2, MATH_BLOCK_3, MATH_BLOCK_4):
        text, n = _count_sub(p, "", text)
        stats["removed_math_blocks"] += n

    # 5) Optional inline math
    if opts.remove_inline_math:
        text, n = _count_sub(MATH_INLINE_1, "", text)
        stats["removed_inline_math"] += n
        text, n = _count_sub(MATH_INLINE_2, "", text)
        stats["removed_inline_math"] += n

    # 6) Optional remove mentions like "Fig. 1"
    if opts.remove_figtab_mentions:
        text, n = _count_sub(FIGTAB_MENTION, "", text)
        stats["removed_figtab_mentions"] += n

    # 7) Normalize whitespace
    text = TRAILING_SPACES.sub("\n", text)
    text = MULTI_BLANK_LINES.sub("\n\n", text)
    text = text.strip() + "\n"

    return text, stats


# ----------------------------
# File processing
# ----------------------------

def compute_output_path(input_file: str, input_dir: str, output_dir: str) -> str:
    rel_path = os.path.relpath(input_file, input_dir)
    return os.path.join(output_dir, rel_path)


def should_skip(output_path: str) -> bool:
    try:
        p = Path(output_path)
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def process_one_file(args: Tuple[str, str, str, CleanOptions]) -> FileReport:
    input_file, input_dir, output_dir, opts = args
    t0 = time.time()

    out_path = compute_output_path(input_file, input_dir, output_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if opts.skip_existing and should_skip(out_path):
        return FileReport(
            input_path=input_file,
            output_path=out_path,
            ok=True,
            skipped=True,
            error=None,
            bytes_in=0,
            bytes_out=0,
            cut_triggered=False,
            removed_html_page_spans=0,
            removed_html_empty_spans=0,
            removed_html_empty_as=0,
            removed_marker_tokens=0,
            removed_num_citations=0,
            removed_latex_cite_ref=0,
            removed_math_blocks=0,
            removed_inline_math=0,
            removed_figures_tables=0,
            removed_md_images=0,
            removed_md_tables=0,
            removed_figtab_mentions=0,
            seconds=float(time.time() - t0),
        )

    try:
        raw = Path(input_file).read_text(encoding="utf-8", errors="ignore")
        bytes_in = len(raw.encode("utf-8", errors="ignore"))

        cleaned, stats = clean_markdown(raw, opts)
        bytes_out = len(cleaned.encode("utf-8", errors="ignore"))

        Path(out_path).write_text(cleaned, encoding="utf-8")

        return FileReport(
            input_path=input_file,
            output_path=out_path,
            ok=True,
            skipped=False,
            error=None,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            cut_triggered=bool(stats["cut_triggered"]),
            removed_html_page_spans=int(stats["removed_html_page_spans"]),
            removed_html_empty_spans=int(stats["removed_html_empty_spans"]),
            removed_html_empty_as=int(stats["removed_html_empty_as"]),
            removed_marker_tokens=int(stats["removed_marker_tokens"]),
            removed_num_citations=int(stats["removed_num_citations"]),
            removed_latex_cite_ref=int(stats["removed_latex_cite_ref"]),
            removed_math_blocks=int(stats["removed_math_blocks"]),
            removed_inline_math=int(stats["removed_inline_math"]),
            removed_figures_tables=int(stats["removed_figures_tables"]),
            removed_md_images=int(stats["removed_md_images"]),
            removed_md_tables=int(stats["removed_md_tables"]),
            removed_figtab_mentions=int(stats["removed_figtab_mentions"]),
            seconds=float(time.time() - t0),
        )
    except Exception as e:
        return FileReport(
            input_path=input_file,
            output_path=out_path,
            ok=False,
            skipped=False,
            error=str(e),
            bytes_in=0,
            bytes_out=0,
            cut_triggered=False,
            removed_html_page_spans=0,
            removed_html_empty_spans=0,
            removed_html_empty_as=0,
            removed_marker_tokens=0,
            removed_num_citations=0,
            removed_latex_cite_ref=0,
            removed_math_blocks=0,
            removed_inline_math=0,
            removed_figures_tables=0,
            removed_md_images=0,
            removed_md_tables=0,
            removed_figtab_mentions=0,
            seconds=float(time.time() - t0),
        )


def find_md_files(input_dir: str) -> List[str]:
    pattern = os.path.join(input_dir, "**", "*.md")
    return glob.glob(pattern, recursive=True)


def assert_safe_paths(input_dir: str, output_dir: str) -> None:
    in_p = Path(input_dir).resolve()
    out_p = Path(output_dir).resolve()
    if out_p == in_p or str(out_p).startswith(str(in_p) + os.sep):
        raise ValueError(
            f"Unsafe output_dir: output_dir ({out_p}) must NOT be the same as or inside input_dir ({in_p})."
        )


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--remove_inline_math", type=int, default=0, choices=[0, 1])
    parser.add_argument("--remove_figtab_mentions", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--remove_empty_html_anchors", type=int, default=1, choices=[0, 1])
    parser.add_argument("--report_name", default="clean_report.jsonl")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    assert_safe_paths(input_dir, output_dir)

    opts = CleanOptions(
        remove_inline_math=bool(args.remove_inline_math),
        remove_figtab_mentions=bool(args.remove_figtab_mentions),
        skip_existing=bool(args.skip_existing),
        remove_empty_html_anchors=bool(args.remove_empty_html_anchors),
    )

    md_files = find_md_files(input_dir)
    print(f"Found {len(md_files)} markdown files under: {input_dir}")
    if not md_files:
        print("No .md files found. Exiting.")
        return

    report_path = os.path.join(output_dir, args.report_name)
    work_items = [(f, input_dir, output_dir, opts) for f in md_files]

    ok_count = 0
    err_count = 0
    skip_count = 0
    t0 = time.time()

    with open(report_path, "w", encoding="utf-8") as rep_f:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one_file, item) for item in work_items]

            total = len(futures)
            done = 0
            last_print = time.time()

            for fut in as_completed(futures):
                r: FileReport = fut.result()
                rep_f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

                done += 1
                if r.ok:
                    ok_count += 1
                    if r.skipped:
                        skip_count += 1
                else:
                    err_count += 1

                now = time.time()
                if now - last_print >= 1.0 or done == total:
                    print(f"[{done}/{total}] ok={ok_count} skipped={skip_count} err={err_count}", end="\r")
                    last_print = now

    print()
    print(f"Done. ok={ok_count}, skipped={skip_count}, err={err_count}, seconds={time.time() - t0:.2f}")
    print(f"Report: {report_path}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
