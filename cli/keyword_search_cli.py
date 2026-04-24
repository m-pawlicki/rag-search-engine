import argparse
import keyword_search
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the cache")

    tf_parser = subparsers.add_parser("tf", help="Return frequency of a term for a given id")
    tf_parser.add_argument("docid", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

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
            freq = indexer.get_tf(args.docid, args.term)
            print(f"Frequency of {args.term} in doc ID {args.docid}: {freq}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
