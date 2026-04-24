import argparse, math
import keyword_search
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the cache")

    tf_parser = subparsers.add_parser("tf", help="Return frequency of a term for a given id")
    tf_parser.add_argument("tf_id", type=int, help="Document ID")
    tf_parser.add_argument("tf_term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="Return inverse document frequency of a term")
    idf_parser.add_argument("idf_term", type=str, help="Search term")

    args = parser.parse_args()
    indexer = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_result = keyword_search.keyword_search(args.query)
            if len(search_result) == 0:
                print("No results found!")
            else:
                for i in range(len(search_result)):
                    print(f"{i+1}. [{search_result[i]["id"]}] {search_result[i]["title"]}")
        case "build":
            print("Building the cache...")
            indexer.build()
            indexer.save()
            print("Done!")
        case "tf":
            indexer.load()
            freq = indexer.get_tf(args.tf_id, args.tf_term)
            print(f"Frequency of '{args.tf_term}' in doc ID {args.tf_id}: {freq}")
        case "idf":
            indexer.load()
            idf = indexer.get_idf(args.idf_term)
            print(f"Inverse document frequency of '{args.idf_term}': {idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
