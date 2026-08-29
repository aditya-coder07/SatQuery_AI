import argparse
import sys
from pathlib import Path

from satquery.cli import evaluate as eval_cmd
from satquery.cli.matrix_validate import validate_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satquery")
    subparsers = parser.add_subparsers(dest="command")

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument(
        "--validate", action="store_true", help="Validate the capability matrix"
    )
    matrix_parser.add_argument(
        "--matrix-path",
        type=Path,
        default=Path("configs/capability_matrix.yaml"),
        help="Path to matrix file",
    )

    eval_cmd.add_parser(subparsers)

    ask_parser = subparsers.add_parser("ask", help="Run one query against images")
    ask_parser.add_argument("images", type=Path, nargs="+", help="one or two rasters")
    ask_parser.add_argument("--query", required=True)
    ask_parser.add_argument("--trace", action="store_true", help="print the full trace")
    ask_parser.add_argument(
        "--evidence", type=Path,
        help="write an evidence pack (ZIP) to this directory",
    )

    return parser


def _run_ask(args) -> int:
    import json

    from satquery.controller.pipeline import Controller

    trace = Controller().run(args.images, args.query)

    if args.evidence:
        from satquery.report import export

        archive = export(trace, args.evidence, artifact_dir="artifacts")
        print(f"Evidence pack: {archive}")

    if args.trace:
        print(trace.model_dump_json(indent=2))
    else:
        print(f"Task      : {trace.routing.selected_task}")
        print(f"Answer    : {trace.answer}")
        print(
            f"Confidence: {trace.confidence.final:.3f} ({trace.confidence.band})"
        )
        if trace.abstained:
            print(f"Abstained : {trace.abstain_reason}")
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "matrix":
        if args.validate:
            if validate_matrix(args.matrix_path):
                print("Matrix validation successful.")
                sys.exit(0)
            print("Matrix validation failed.")
            sys.exit(1)
        parser.parse_args([args.command, "--help"])
        sys.exit(1)

    if args.command == "eval":
        sys.exit(eval_cmd.run(args))

    if args.command == "ask":
        sys.exit(_run_ask(args))

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
