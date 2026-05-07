import argparse
import lib.hybrid_search as hs
import lib.query_enhancement as qe
from lib.search_utils import DEFAULT_SEARCH_LIMIT, ALPHA, K_VALUE

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="A list of scores to normalize")

    weighred_search_parser = subparsers.add_parser("weighted-search", help="Search using ")
    weighred_search_parser.add_argument("query", type=str, help="Search query")
    weighred_search_parser.add_argument("--alpha", type=float, default=ALPHA, help="Weight of keyword to semantic search, 1.0 = 100%% keyword, 0.0 = 100%% semantic")
    weighred_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="How many search results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Search using Reciprocal Rank Fusion")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=K_VALUE, help="How much weight higher-ranked results have vs lower-ranked ones, lower k = more weight, higher k = less weight")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="How many search results to return")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hs.normalize_scores_command(args.scores)
        case "weighted-search":
            hs.weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            
            match args.enhance:
                case "spell":
                    enhanced_query = qe.spell_enhance_command(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                    hs.rrf_search_command(enhanced_query, args.k, args.limit)

                case "rewrite":
                    enhanced_query = qe.rewrite_enhance_command(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                    hs.rrf_search_command(enhanced_query, args.k, args.limit)

                case _:
                    hs.rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()