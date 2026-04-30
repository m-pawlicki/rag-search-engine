#!/usr/bin/env python3

import argparse
import lib.semantic_search as ss
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT, 
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_SEMANTIC_CHUNK_SIZE, 
    load_movies)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the currently loaded semantic model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to get embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify the current embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a given query")
    embed_query_parser.add_argument("query", type=str, help="Query to get embedding for")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="How many results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk long text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of each chunk in characters")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="How much overlap in each chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk long text into smaller pieces")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Size of each chunk in sentences")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="How much overlap in each chunk")

    subparsers.add_parser("embed_chunks", help="Embed document chunks")    

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

        case "search":
            search_instance = ss.SemanticSearch()
            movies = load_movies()
            search_instance.load_or_create_embeddings(movies)
            search_result = search_instance.search(args.query, args.limit)
            for i, res in enumerate(search_result, start=1):
                print(f"{i}. {res["title"]} (score: {res["score"]})\n{res["description"][:140]}...\n")

        case "chunk":
            result = ss.chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, res in enumerate(result, start=1):
                print(f"{i}. {res}")

        case "semantic_chunk":
            result = ss.semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, res in enumerate(result, start=1):
                print(f"{i}. {res}")

        case "embed_chunks":
            ss.embed_chunks()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()