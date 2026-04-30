from sentence_transformers import SentenceTransformer
from lib.search_utils import (CACHE_PATH, 
                              DEFAULT_SEARCH_LIMIT, 
                              DEFAULT_CHUNK_SIZE, 
                              DEFAULT_CHUNK_OVERLAP,
                              DEFAULT_SEMANTIC_CHUNK_SIZE, 
                              load_movies)
import numpy as np
import os, re

class SemanticSearch:
    def __init__(self, name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
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