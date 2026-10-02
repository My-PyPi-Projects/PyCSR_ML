"""Command-line interface for PyCSR_ML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .api import generate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PyCSR_ML",
        description="Create a business-ready profiling and ML comparison report from CSV or TXT data.",
        epilog="Example: PyCSR_ML --input test.csv --target churn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to a .csv or .txt dataset")
    parser.add_argument("--output", "-o", help="Destination HTML path; defaults beside the input file")
    parser.add_argument("--target", "-t", help="Prediction target; if omitted, safe inference is attempted")
    parser.add_argument("--no-ml", action="store_true", help="Generate profiling without model comparison")
    parser.add_argument("--max-model-rows", type=int, default=20000, help="Maximum rows used for modeling")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_model_rows < 30:
        print("error: --max-model-rows must be at least 30", file=sys.stderr)
        return 2
    try:
        print(f"[1/3] Reading and profiling: {Path(args.input)}")
        print("[2/3] Running automatic baseline ML comparison when a target is available")
        destination = generate_report(
            args.input,
            args.output,
            args.target,
            not args.no_ml,
            args.random_state,
            args.max_model_rows,
        )
        print(f"[3/3] Report ready: {destination}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
