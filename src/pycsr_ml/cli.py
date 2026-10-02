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
        epilog="Example: pycsr-ml data.csv --target churn --cv 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_path", nargs="?", help="Path to a .csv or .txt dataset")
    parser.add_argument(
        "--input", "-i", dest="input_option", metavar="INPUT", help="Path to a .csv or .txt dataset"
    )
    parser.add_argument("--output", "-o", help="Destination HTML path; defaults beside the input file")
    parser.add_argument("--target", "-t", help="Prediction target; if omitted, safe inference is attempted")
    parser.add_argument("--no-ml", action="store_true", help="Generate profiling without model comparison")
    parser.add_argument("--max-model-rows", type=int, default=20000, help="Maximum rows used for modeling")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducibility seed")
    parser.add_argument(
        "--cv",
        type=int,
        metavar="FOLDS",
        help="Enable cross-validation with 2-20 folds instead of a single holdout",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_path and args.input_option:
        parser.error("provide the dataset either positionally or with --input, not both")
    input_path = args.input_option or args.input_path
    if not input_path:
        parser.error("a dataset path is required (positional or --input PATH)")
    if args.max_model_rows < 30:
        print("error: --max-model-rows must be at least 30", file=sys.stderr)
        return 2
    if args.cv is not None and not 2 <= args.cv <= 20:
        print("error: --cv must be between 2 and 20", file=sys.stderr)
        return 2
    try:
        print(f"[1/3] Reading and profiling: {Path(input_path)}")
        print("[2/3] Running automatic baseline ML comparison when a target is available")
        destination = generate_report(
            input_path=input_path,
            output_path=args.output,
            target=args.target,
            run_ml=not args.no_ml,
            random_state=args.random_state,
            max_model_rows=args.max_model_rows,
            cv=args.cv,
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
