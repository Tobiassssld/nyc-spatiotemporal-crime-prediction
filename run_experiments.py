"""Command-line entry point for the NYC crime-prediction experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments import run_experiments


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TCP baseline, extension, and ablation experiments."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "smoke"),
        default="full",
        help=(
            "Run the complete experiment configuration or a lightweight "
            "end-to-end smoke test."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Omit it to keep the default stochastic behavior.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip exploratory temporal and spatial diagnostic plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        project_root=PROJECT_ROOT,
        mode=args.mode,
        device_name=args.device,
        seed=args.seed,
        run_diagnostics=not args.skip_diagnostics,
    )


if __name__ == "__main__":
    main()
