# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.gui_grounding import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkGUIGroundingSubcommand(BenchmarkSubcommandBase):
    """The `gui-grounding` subcommand for `vllm bench`."""

    name = "gui-grounding"
    help = "Benchmark GUI grounding accuracy using the OSWorld-G dataset."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
