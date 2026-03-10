#!/usr/bin/env python3
"""
Convert OpenReview GRPO data to veRL parquet format
"""

import argparse
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def convert_grpo_json_to_parquet(
    input_json_path: str, output_parquet_path: str, data_source: str = "openreview"
):
    """
    Convert GRPO JSON format to veRL parquet format

    Input JSON format:
    [
        {
            "messages": [
                {"role": "user", "content": "..."}
            ]
        }
    ]

    Output Parquet format:
    - data_source: str
    - prompt: list[dict] (conversation messages)
    - ability: str
    - reward_model: dict
    - extra_info: dict
    """
    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    converted_data = []
    for idx, item in enumerate(data):
        messages = item.get("messages", [])

        prompt = []
        for msg in messages:
            prompt.append({"role": msg["role"], "content": msg["content"]})

        converted_item = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "review_generation",
            "reward_model": {
                "style": "external",
                "ground_truth": "",
            },
            "extra_info": {
                "index": idx,
            },
        }

        converted_data.append(converted_item)

    print(f"Converting to parquet format...")
    df = pd.DataFrame(converted_data)

    table = pa.Table.from_pandas(df)

    print(f"Saving to {output_parquet_path}...")
    pq.write_table(table, output_parquet_path)

    print(f"Done! Saved {len(converted_data)} samples to {output_parquet_path}")

    return len(converted_data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenReview GRPO data to veRL parquet format"
    )
    parser.add_argument(
        "--input_json", type=str, required=True, help="Input JSON file path"
    )
    parser.add_argument(
        "--output_parquet", type=str, required=True, help="Output parquet file path"
    )
    parser.add_argument(
        "--data_source", type=str, default="openreview", help="Data source identifier"
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    convert_grpo_json_to_parquet(args.input_json, args.output_parquet, args.data_source)


if __name__ == "__main__":
    main()
