import argparse
import keyword_search
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the cache")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score of a given document ID and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

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
            freq = indexer.get_tf(args.doc_id, args.term)
            print(f"Frequency of '{args.term}' in document '{args.doc_id}': {freq}")

        case "idf":
            indexer.load()
            idf = indexer.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            indexer.load()
            tf_idf = indexer.get_tf_idf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
