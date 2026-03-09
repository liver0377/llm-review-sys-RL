#!/usr/bin/env python3
"""
Download Qwen3.5-35B-A3B model from HuggingFace.

Usage:
    python scripts/download_qwen35_35b_a3b.py
    python scripts/download_qwen35_35b_a3b.py --output-dir /path/to/output
    python scripts/download_qwen35_35b_a3b.py --model-id Qwen/Qwen3.5-35B-A3B --revision main
"""

import os
import argparse
from huggingface_hub import snapshot_download


def download_model(
    model_id: str = "Qwen/Qwen3.5-35B-A3B",
    output_dir: str = "models/qwen3.5-35b-a3b",
    revision: str = "main",
    token: str = None,
):
    """
    Download Qwen3.5-35B-A3B model from HuggingFace.

    Args:
        model_id: HuggingFace model ID
        output_dir: Local directory to save the model
        revision: Model revision (branch, tag or commit hash)
        token: HuggingFace authentication token (if needed)
    """
    print(f"========================================")
    print(f"Downloading Qwen3.5-35B-A3B Model")
    print(f"========================================")
    print(f"Model ID: {model_id}")
    print(f"Output Dir: {output_dir}")
    print(f"Revision: {revision}")
    print(f"========================================\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download model
    print("Starting download...")
    print("Note: This model is ~70GB, so it may take a while depending on your connection.\n")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            revision=revision,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"\n========================================")
        print(f"Download completed successfully!")
        print(f"Model saved to: {output_dir}")
        print(f"========================================\n")

        # Print model size
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(output_dir)
            for filename in filenames
        )
        size_gb = total_size / (1024**3)
        print(f"Total model size: {size_gb:.2f} GB\n")

    except Exception as e:
        print(f"\nError downloading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5-35B-A3B model from HuggingFace"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3.5-35B-A3B",
        help="HuggingFace model ID (default: Qwen/Qwen3.5-35B-A3B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/qwen3.5-35b-a3b",
        help="Local directory to save the model (default: models/qwen3.5-35b-a3b)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch (default: main)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace authentication token (if needed)",
    )

    args = parser.parse_args()

    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Please install it using: pip install huggingface_hub")
        return 1

    download_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        revision=args.revision,
        token=args.token,
    )

    return 0


if __name__ == "__main__":
    exit(main())
