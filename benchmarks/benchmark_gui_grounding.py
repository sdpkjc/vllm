#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone GUI Grounding Benchmark script.

This script provides a standalone entry point for the GUI grounding benchmark
using the OSWorld-G dataset. For the recommended usage, use:
    vllm bench gui-grounding [options]

Example usage:
    # 1. Start vLLM server
    vllm serve Qwen/Qwen3-VL-8B-Instruct --trust-remote-code

    # 2. Run benchmark (auto-downloads dataset from HuggingFace)
    python benchmarks/benchmark_gui_grounding.py \\
        --base-url http://localhost:8000 \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --num-samples 100 \\
        --output-file results.json

    # Or use local dataset
    python benchmarks/benchmark_gui_grounding.py \\
        --base-url http://localhost:8000 \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --dataset-path /path/to/OSWorld-G_refined.json \\
        --image-dir /path/to/images \\
        --num-samples 100 \\
        --output-file results.json
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to path to allow importing vllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm.benchmarks.gui_grounding import add_cli_args, main  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="GUI Grounding Benchmark using OSWorld-G dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (auto-download from HuggingFace):
    python benchmarks/benchmark_gui_grounding.py \\
        --base-url http://localhost:8000 \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --num-samples 100 \\
        --output-file results.json

Example (local dataset):
    python benchmarks/benchmark_gui_grounding.py \\
        --base-url http://localhost:8000 \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --dataset-path /path/to/OSWorld-G_refined.json \\
        --image-dir /path/to/images \\
        --num-samples 100 \\
        --output-file results.json

For more information, see the vLLM documentation.
        """,
    )
    add_cli_args(parser)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
