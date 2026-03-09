import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


OVERALL_RE = re.compile(r"\*?\*?Overall Quality:\*?\*?\s*([0-9]+(?:\.[0-9]+)?)")
CONF_RE = re.compile(r"\*?\*?Review Confidence:\*?\*?\s*([0-9]+(?:\.[0-9]+)?)")


def has_both_fields(text: str):
    if not isinstance(text, str):
        return False, False, False
    has_overall = OVERALL_RE.search(text) is not None
    has_conf = CONF_RE.search(text) is not None
    return (has_overall and has_conf), has_overall, has_conf


def clean_dpo_dataset(
    input_path: str,
    output_path: str = None,
    backup: bool = True,
    remove_incomplete: bool = False,
):
    """
    Remove entries where rejected field is failed or incomplete.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (default: input_path with '_cleaned' suffix)
        backup: Whether to create a backup of the original file
        remove_incomplete: If True, also remove entries missing Overall Quality or Review Confidence
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = (
            input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    print(f"📂 Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of records")

    original_count = len(data)

    # Track removal reasons
    removal_reasons = Counter()
    removed_indices = []

    # Filter out failed/incomplete entries
    cleaned_data = []
    for i, rec in enumerate(data):
        rejected = rec.get("rejected", "")

        # Check for API failures
        if rejected == "PLACEHOLDER_FOR_API_FAILURE":
            removal_reasons["PLACEHOLDER_FOR_API_FAILURE"] += 1
            removed_indices.append(i)
            continue
        elif rejected == "GENERATION_FAILED":
            removal_reasons["GENERATION_FAILED"] += 1
            removed_indices.append(i)
            continue

        # Check for incomplete ratings if flag is enabled
        if remove_incomplete:
            ok, has_o, has_c = has_both_fields(rejected)
            if not ok:
                if not has_o and not has_c:
                    removal_reasons["missing_both_ratings"] += 1
                elif not has_o:
                    removal_reasons["missing_overall_quality"] += 1
                elif not has_c:
                    removal_reasons["missing_review_confidence"] += 1
                removed_indices.append(i)
                continue

        cleaned_data.append(rec)

    removed_count = original_count - len(cleaned_data)

    # Create backup if requested
    if backup and output_path.exists():
        backup_path = output_path.parent / f"{output_path.stem}.bak{output_path.suffix}"
        shutil.copy2(output_path, backup_path)
        print(f"💾 Backed up existing output to: {backup_path}")

    # Write cleaned data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Cleaning complete!")
    print(f"   Original samples: {original_count}")
    print(f"   Removed samples:  {removed_count}")
    print(f"   Final samples:    {len(cleaned_data)}")
    print(f"\n📄 Saved to: {output_path}")

    # Show removal breakdown
    if removal_reasons:
        print(f"\n📊 Removal breakdown:")
        for reason, count in removal_reasons.most_common():
            print(f"   {reason}: {count}")

    # Show removed indices if any
    if removed_indices:
        print(f"\n🗑️  Removed indices (first 20): {removed_indices[:20]}")
        if len(removed_indices) > 20:
            print(f"   ... and {len(removed_indices) - 20} more")


def main():
    ap = argparse.ArgumentParser(
        description="Remove failed/incomplete entries from DPO dataset"
    )
    ap.add_argument("--input", required=True, help="Input JSON file path")
    ap.add_argument(
        "--output", help="Output JSON file path (default: <input>_cleaned.json)"
    )
    ap.add_argument(
        "--no-backup", action="store_true", help="Skip backup if output file exists"
    )
    ap.add_argument(
        "--remove-incomplete",
        action="store_true",
        help="Also remove entries missing Overall Quality or Review Confidence in rejected field",
    )

    args = ap.parse_args()

    clean_dpo_dataset(
        input_path=args.input,
        output_path=args.output,
        backup=not args.no_backup,
        remove_incomplete=args.remove_incomplete,
    )


if __name__ == "__main__":
    main()
