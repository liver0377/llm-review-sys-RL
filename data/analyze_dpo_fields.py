import argparse
import json
import os
import re
from collections import Counter

try:
    import pandas as pd
except ImportError:
    pd = None


OVERALL_RE = re.compile(r"\*?\*?Overall Quality:\*?\*?\s*([0-9]+(?:\.[0-9]+)?)")
CONF_RE = re.compile(r"\*?\*?Review Confidence:\*?\*?\s*([0-9]+(?:\.[0-9]+)?)")


def has_both_fields(text: str):
    if not isinstance(text, str):
        return False, False, False
    has_overall = OVERALL_RE.search(text) is not None
    has_conf = CONF_RE.search(text) is not None
    return (has_overall and has_conf), has_overall, has_conf


def iter_records(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    raise ValueError(f"Bad JSON on line {line_no}: {e}") from e

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of records (or {data: [...]})")
        for rec in data:
            yield rec

    elif ext == ".parquet":
        if pd is None:
            raise ImportError(
                "pandas is required to read parquet: pip install pandas pyarrow"
            )
        df = pd.read_parquet(path)
        for rec in df.to_dict(orient="records"):
            yield rec

    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Use .jsonl/.json/.parquet"
        )


def get_text(rec: dict, key_candidates):
    for k in key_candidates:
        if k in rec and isinstance(rec[k], str):
            return rec[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path", required=True, help="Path to DPO dataset file (.jsonl/.json/.parquet)"
    )
    ap.add_argument(
        "--chosen_key",
        default="chosen",
        help="Override chosen field name (default: auto-detect)",
    )
    ap.add_argument(
        "--rejected_key",
        default="rejected",
        help="Override rejected field name (default: auto-detect)",
    )
    ap.add_argument(
        "--print_missing",
        type=int,
        default=0,
        help="Print N examples where chosen/rejected missing fields",
    )
    args = ap.parse_args()

    # Common DPO field variants
    chosen_candidates = (
        [args.chosen_key]
        if args.chosen_key
        else ["chosen", "accept", "preferred", "output_chosen", "chosen_response"]
    )
    rejected_candidates = (
        [args.rejected_key]
        if args.rejected_key
        else [
            "rejected",
            "reject",
            "dispreferred",
            "output_rejected",
            "rejected_response",
        ]
    )

    total = 0

    chosen_ok = 0
    rejected_ok = 0

    chosen_reason = Counter()
    rejected_reason = Counter()

    missing_examples = []

    for rec in iter_records(args.path):
        total += 1

        chosen_text = get_text(rec, chosen_candidates)
        rejected_text = get_text(rec, rejected_candidates)

        # If keys not found, count as missing
        if chosen_text is None:
            chosen_reason["missing_chosen_field"] += 1
        else:
            ok, has_o, has_c = has_both_fields(chosen_text)
            if ok:
                chosen_ok += 1
            else:
                if not has_o and not has_c:
                    chosen_reason["missing_both"] += 1
                elif not has_o:
                    chosen_reason["missing_overall"] += 1
                elif not has_c:
                    chosen_reason["missing_confidence"] += 1

        if rejected_text is None:
            rejected_reason["missing_rejected_field"] += 1
        else:
            ok, has_o, has_c = has_both_fields(rejected_text)
            if ok:
                rejected_ok += 1
            else:
                if not has_o and not has_c:
                    rejected_reason["missing_both"] += 1
                elif not has_o:
                    rejected_reason["missing_overall"] += 1
                elif not has_c:
                    rejected_reason["missing_confidence"] += 1

        if args.print_missing > 0:
            # Save some examples where either chosen or rejected fails
            ch_ok = chosen_text is not None and has_both_fields(chosen_text)[0]
            rj_ok = rejected_text is not None and has_both_fields(rejected_text)[0]
            if (not ch_ok) or (not rj_ok):
                if len(missing_examples) < args.print_missing:
                    missing_examples.append(
                        {
                            "idx": total,
                            "chosen_preview": (
                                chosen_text[:400]
                                if isinstance(chosen_text, str)
                                else str(chosen_text)
                            ),
                            "rejected_preview": (
                                rejected_text[:400]
                                if isinstance(rejected_text, str)
                                else str(rejected_text)
                            ),
                        }
                    )

    def pct(x):
        return (100.0 * x / total) if total else 0.0

    print(f"\n📄 File: {args.path}")
    print(f"Total samples: {total}")

    print("\n=== Coverage (both fields present) ===")
    print(f"chosen:   {chosen_ok}/{total}  ({pct(chosen_ok):.2f}%)")
    print(f"rejected: {rejected_ok}/{total}  ({pct(rejected_ok):.2f}%)")

    print("\n=== Missing breakdown (chosen) ===")
    if chosen_reason:
        for k, v in chosen_reason.most_common():
            print(f"{k:>22}: {v} ({pct(v):.2f}%)")
    else:
        print("No missing issues detected for chosen.")

    print("\n=== Missing breakdown (rejected) ===")
    if rejected_reason:
        for k, v in rejected_reason.most_common():
            print(f"{k:>22}: {v} ({pct(v):.2f}%)")
    else:
        print("No missing issues detected for rejected.")

    if missing_examples:
        print("\n=== Examples (missing) ===")
        for ex in missing_examples:
            print(f"\n--- sample #{ex['idx']} ---")
            print("chosen_preview:\n", ex["chosen_preview"])
            print("rejected_preview:\n", ex["rejected_preview"])

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
