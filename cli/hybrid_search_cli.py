import argparse
import lib.hybrid_search as hs
from lib.search_utils import DEFAULT_SEARCH_LIMIT, ALPHA

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="A list of scores to normalize")

    weighred_search_parser = subparsers.add_parser("weighted-search", help="Search using ")
    weighred_search_parser.add_argument("query", type=str, help="Search query")
    weighred_search_parser.add_argument("--alpha", type=float, default=ALPHA, help="Weight of keyword to semantic search, 1.0 = 100%% keyword, 0.0 = 100%% semantic")
    weighred_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="How many search results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hs.normalize_scores_command(args.scores)
        case "weighted-search":
            hs.weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()