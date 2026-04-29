#!/usr/bin/env python3

import argparse
import lib.semantic_search as ss

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the currently loaded semantic model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to get embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify the current embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a given query")
    embed_query_parser.add_argument("query", type=str, help="Query to get embedding for")

    args = parser.parse_args()
    
    match args.command:
        case "verify":
            ss.verify_model()
        case "embed_text":
            ss.embed_text(args.text)
        case "verify_embeddings":
            ss.verify_embeddings()
        case "embed_query":
            ss.embed_query_text(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()