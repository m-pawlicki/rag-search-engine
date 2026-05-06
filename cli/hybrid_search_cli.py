import argparse
import lib.hybrid_search as hs

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="A list of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hs.normalize_scores_command(args.scores)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()