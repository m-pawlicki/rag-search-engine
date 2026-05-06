import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize_scores(scores):
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        result = [1.0] * len(scores)
        return result
    
    result = []
    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        result.append(normalized_score)
    return result


def normalize_scores_command(scores):
    result = normalize_scores(scores)
    if not result:
        return
    for score in result:
        print(f"* {score:.4f}")