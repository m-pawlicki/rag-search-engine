from sentence_transformers import SentenceTransformer
from lib.search_utils import (CACHE_PATH, 
                              DEFAULT_SEARCH_LIMIT, 
                              DEFAULT_CHUNK_SIZE, 
                              DEFAULT_CHUNK_OVERLAP,
                              DEFAULT_SEMANTIC_CHUNK_SIZE,
                              SCORE_PRECISION, 
                              load_movies)
import numpy as np
import os, re, json

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input cannot be empty.")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents

        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            entry = f"{doc["title"]}: {doc["description"]}"
            doc_list.append(entry)
        
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        with open(f"{CACHE_PATH}/movie_embeddings.npy", "wb") as me:
            np.save(me, self.embeddings)
        
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(f"{CACHE_PATH}/movie_embeddings.npy"):
            with open(f"{CACHE_PATH}/movie_embeddings.npy", "rb") as me:
                self.embeddings = np.load(me)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embedding = self.generate_embedding(query)
        similarity_scores = []

        for index, document in enumerate(self.embeddings):
            cosine = cosine_similarity(embedding, document)
            similarity_scores.append((cosine, self.documents[index]))

        sorted_scores = sorted(similarity_scores, key=lambda item:item[0], reverse=True)

        results = []
        for score, doc in sorted_scores[:limit]:
            entry = {"score": score, "title": doc["title"], "description": doc["description"]}
            results.append(entry)
        
        return results
    
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = []
        chunk_meta = []

        for  doc_idx, doc in enumerate(self.documents):

            self.document_map[doc["id"]] = doc

            if not doc["description"]:
                continue

            text_chunks = semantic_chunk_text(text = doc["description"], size = 4, overlap = 1)

            for chunk_idx, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                entry = {"movie_idx": doc_idx, "chunk_idx": chunk_idx, "total_chunks": len(text_chunks)}
                chunk_meta.append(entry)

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_meta

        with open(f"{CACHE_PATH}/chunk_embeddings.npy", "wb") as ce:
            np.save(ce, self.chunk_embeddings)

        with open(f"{CACHE_PATH}/chunk_metadata.json", "w") as md:
            json.dump({"chunks": chunk_meta, "total_chunks": len(chunks)}, md, indent=2)


        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents

        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(f"{CACHE_PATH}/chunk_embeddings.npy") and os.path.exists(f"{CACHE_PATH}/chunk_metadata.json"):
            with open(f"{CACHE_PATH}/chunk_embeddings.npy", "rb") as ce:
                self.chunk_embeddings = np.load(ce)

            with open(f"{CACHE_PATH}/chunk_metadata.json", "r") as md:
                data = json.load(md)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []

        for embed_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            cosine = cosine_similarity(chunk_embedding, query_embedding)
            meta = self.chunk_metadata[embed_idx]
            entry = {"chunk_idx": meta["chunk_idx"], "movie_idx": meta["movie_idx"], "score": cosine}
            chunk_scores.append(entry)

        movie_scores = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        
        sorted_scores = sorted(movie_scores.items(), key = lambda item: item[1], reverse=True)

        results = []
        for movie_idx, score in sorted_scores[:limit]:
            doc = self.documents[movie_idx]
            entry = {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": doc.get("metadata") or {}
            }
            results.append(entry)
        
        return results

def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    model = SemanticSearch()
    documents = load_movies()
    embeddings = model.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(text: str, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    split_text = text.split()
    chunks = []

    if len(text) == 0:
        raise ValueError("No text to chunk!")
    
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    while len(split_text) > (0 + overlap):
        chunk = " ".join(split_text[:size])
        chunks.append(chunk)
        split_text = split_text[(size - overlap):]

    return chunks

def semantic_chunk_text(text: str, size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    split_text = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    if len(text) == 0:
        raise ValueError("No text to chunk!")
    
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    while len(split_text) > (0 + overlap):
        chunk = " ".join(split_text[:size])
        chunks.append(chunk)
        split_text = split_text[(size - overlap):]

    return chunks

def embed_chunks():
    model = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = model.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def chunk_command(text: str, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    result = chunk_text(text, size, overlap)

    print(f"Chunking {len(text)} characters")
    for i, res in enumerate(result, start=1):
        print(f"{i}. {res}")

def semantic_chunk_command(text: str, size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    result = semantic_chunk_text(text, size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, res in enumerate(result, start=1):
        print(f"{i}. {res}")

def search_command(query: str, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    search_results = search_instance.search(query, limit)

    for i, res in enumerate(search_results, start=1):
        print(f"\n{i}. {res["title"]} (score: {res["score"]:.4f})")
        print(f"    {res["description"][:100]}...")

def search_chunked_command(query: str, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = ChunkedSemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_chunk_embeddings(movies)
    search_results = search_instance.search_chunks(query, limit)
    for i, res in enumerate(search_results, start=1):
        print(f"\n{i}. {res["title"]} (score: {res["score"]:.4f})")
        print(f"   {res["document"]}...")