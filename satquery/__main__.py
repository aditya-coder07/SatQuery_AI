import sys
import argparse
from pathlib import Path
from satquery.cli.matrix_validate import validate_matrix

def main():
    parser = argparse.ArgumentParser(prog="satquery")
    subparsers = parser.add_subparsers(dest="command")
    
    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--validate", action="store_true", help="Validate the capability matrix")
    matrix_parser.add_argument("--matrix-path", type=Path, default=Path("configs/capability_matrix.yaml"), help="Path to matrix file")

    args = parser.parse_args()

    if args.command == "matrix":
        if args.validate:
            if validate_matrix(args.matrix_path):
                print("Matrix validation successful.")
                sys.exit(0)
            else:
                print("Matrix validation failed.")
                sys.exit(1)
        else:
            matrix_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
