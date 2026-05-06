import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import DEFAULT_SEARCH_LIMIT, ALPHA, K_VALUE, load_movies

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

    def weighted_search(self, query, alpha=ALPHA, limit=DEFAULT_SEARCH_LIMIT):
        bm25_results = self._bm25_search(query, limit = (limit*500))
        semantic_results = self.semantic_search.search_chunks(query, limit = (limit*500))

        bm25_scores = []
        for res in bm25_results:
            bm25_scores.append(res["score"])

        semantic_scores = []
        for res in semantic_results:
            semantic_scores.append(res["score"])

        normalized_bm25 = normalize_scores(bm25_scores)
        normalized_semantic = normalize_scores(semantic_scores)

        merged_scores = {}

        for i, res in enumerate(bm25_results):
            doc_id = res["id"]
            doc = self.idx.docmap[doc_id]
            if doc_id not in merged_scores:
                merged_scores[doc_id] = {"id": doc_id, "title": doc["title"],"document": doc["description"][:100], "bm25_score": 0.0, "semantic_score": 0.0}
            if normalized_bm25[i] > merged_scores[doc_id]["bm25_score"]:
                merged_scores[doc_id]["bm25_score"] = normalized_bm25[i]

        for i, res in enumerate(semantic_results):
            doc_id = res["id"]
            doc = self.idx.docmap[doc_id]
            if doc_id not in merged_scores:
                merged_scores[doc_id] = {"id": doc_id, "title": doc["title"], "document": doc["description"][:100], "bm25_score": 0.0, "semantic_score": 0.0}
            if normalized_semantic[i] > merged_scores[doc_id]["semantic_score"]:
                merged_scores[doc_id]["semantic_score"] = normalized_semantic[i]

        for entry in merged_scores.values():
            entry["hybrid_score"] = hybrid_score(entry["bm25_score"], entry["semantic_score"], alpha)

        sorted_scores = sorted(merged_scores.values(), key = lambda entry: entry["hybrid_score"], reverse=True)

        return sorted_scores[:limit]



    def rrf_search(self, query, k=K_VALUE, limit=DEFAULT_SEARCH_LIMIT):
        bm25_results = self._bm25_search(query, limit = (limit*500))
        semantic_results = self.semantic_search.search_chunks(query, limit = (limit*500))

        merged_scores = {}

        for i, res in enumerate(bm25_results, start=1):
            doc_id = res["id"]
            doc = self.idx.docmap[doc_id]
            if doc_id not in merged_scores:
                merged_scores[doc_id] = {"id": doc_id, "title": doc["title"], "document": doc["description"][:100], "bm25_rank": i, "semantic_rank": None}
            if merged_scores[doc_id]["bm25_rank"] is None:
                merged_scores[doc_id]["bm25_rank"] = i

        for i, res in enumerate(semantic_results, start=1):
            doc_id = res["id"]
            doc = self.idx.docmap[doc_id]
            if doc_id not in merged_scores:
                merged_scores[doc_id] = {"id": doc_id, "title": doc["title"], "document": doc["description"][:100], "bm25_rank": None, "semantic_rank": i}
            if merged_scores[doc_id]["semantic_rank"] is None:
                merged_scores[doc_id]["semantic_rank"] = i

        for entry in merged_scores.values():
            score = 0.0
            if entry["bm25_rank"] is not None:
                score += rrf_score(entry["bm25_rank"],k)
            if entry["semantic_rank"] is not None:
                score += rrf_score(entry["semantic_rank"], k)
                
            entry["rrf_score"] = score

        sorted_scores = sorted(merged_scores.values(), key = lambda entry: entry["rrf_score"], reverse=True)

        return sorted_scores
    
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

def hybrid_score(bm25_score, semantic_score, alpha=ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search_command(query, alpha=ALPHA, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    search_instance = HybridSearch(movies)
    search_results = search_instance.weighted_search(query, alpha, limit)
    for i, res in enumerate(search_results[:limit], start=1):
        print(f"{i}. {res["title"]}")
        print(f"    Hybrid Score: {res["hybrid_score"]:.3f}")
        print(f"    BM25: {res["bm25_score"]:.3f}, Semantic: {res["semantic_score"]:.3f}")
        print(f"    {res["document"]}...")

def rrf_score(rank, k=K_VALUE):
    return 1 / (k + rank)

def rrf_search_command(query, k=K_VALUE, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    search_instance = HybridSearch(movies)
    search_results = search_instance.rrf_search(query, k, limit)
    for i, res in enumerate(search_results[:limit], start=1):
        print(f"{i}. {res["title"]}")
        print(f"    RRF Score: {res["rrf_score"]:.3f}")
        print(f"    BM25 Rank: {res["bm25_rank"]}, Semantic Rank: {res["semantic_rank"]}")
        print(f"    {res["document"]}...")