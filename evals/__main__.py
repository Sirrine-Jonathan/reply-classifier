"""CLI entry point: `python -m evals run`.

Examples:
    python -m evals run                                  # heuristic vs real gold set
    python -m evals run --classifier llm                 # LLM (needs a real client)
    python -m evals run --data evals/data/fixture_tiny.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.classifier import get_classifier
from evals.runner import load_gold_set, run_eval, save_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "evals" / "data" / "gold_set.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "reports"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description="Reply Classifier eval harness")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a classifier over the gold set and print a report")
    run.add_argument("--classifier", default="heuristic", choices=["heuristic", "llm"])
    run.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    run.add_argument("--min-accuracy", type=float, default=0.90)
    run.add_argument("--min-recall", type=float, default=0.80)
    args = parser.parse_args(argv)

    if args.command == "run":
        if not args.data.exists():
            print(
                f"Gold set not found: {args.data}\n"
                "Export it first:  python evals/export_gold_set.py",
                file=sys.stderr,
            )
            return 2
        records = load_gold_set(args.data)
        classifier = get_classifier(args.classifier)
        report = run_eval(
            classifier,
            records,
            dataset_name=str(args.data),
            min_accuracy=args.min_accuracy,
            min_recall=args.min_recall,
        )
        json_path, md_path = save_report(report, args.out_dir)
        print(report.to_markdown())
        print(f"Saved: {json_path}")
        print(f"Saved: {md_path}")
        return 0 if report.passed else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
