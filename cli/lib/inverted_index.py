import lib.keyword_search as keyword_search
from lib.search_utils import CACHE_PATH, load_movies, BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT
import os, pickle, sys, math
from collections import defaultdict, Counter

class InvertedIndex:
    def __init__(self):
        # {token:str, doc_ids: set(int)}
        self.index = defaultdict(set)
        # {doc_id: int, doc_obj: dict}
        self.docmap: dict[int, dict] = {}
        # {doc_id: int, count: Counter()}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = keyword_search.tokenize_text(text)
        total_tokens = len(tokens)

        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

        self.doc_lengths[doc_id] = total_tokens

    def __get_avg_doc_length(self) -> float:
        num_docs = len(self.doc_lengths)
        if num_docs == 0:
            return 0.0
        total = 0.0
        for doc in self.doc_lengths:
                total += self.doc_lengths[doc]
        return total/num_docs

    def get_documents(self, term: str):
        lowercase_term = term.lower()
        if lowercase_term in self.index:
            return sorted(list(self.index[lowercase_term]))
        else:
            return []

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_id = movie["id"]
            movie_desc = f"{movie["title"]} {movie["description"]}"
            self.__add_document(movie_id, movie_desc)
            self.docmap[movie_id] = movie

    def save(self):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)

        with open(f"{CACHE_PATH}/index.pkl", "wb") as idx:
            pickle.dump(self.index, idx)

        with open(f"{CACHE_PATH}/docmap.pkl", "wb") as doc:
            pickle.dump(self.docmap, doc)

        with open(f"{CACHE_PATH}/term_frequencies.pkl", "wb") as tf:
            pickle.dump(self.term_frequencies, tf)

        with open(f"{CACHE_PATH}/doc_lengths.pkl", "wb") as dl:
            pickle.dump(self.doc_lengths, dl)

    def load(self):
        try:
            with open(f"{CACHE_PATH}/index.pkl", "rb") as idx:
                self.index = pickle.load(idx)

            with open(f"{CACHE_PATH}/docmap.pkl", "rb") as doc:
                self.docmap = pickle.load(doc)

            with open(f"{CACHE_PATH}/term_frequencies.pkl", "rb") as tf:
                self.term_frequencies = pickle.load(tf)

            with open(f"{CACHE_PATH}/doc_lengths.pkl", "rb") as dl:
                self.doc_lengths = pickle.load(dl)

        except FileNotFoundError:
            print("Error loading file(s)! Do they exist?")
            sys.exit(1)

    def is_single_token(self, token: list[str]):
        if len(token) > 1:
            return False
        return True

    def get_tf(self, doc_id: int, term: str):
        token = keyword_search.tokenize_text(term)
        if not self.is_single_token(token):
            raise Exception("Error: More than one token detected.")
        if doc_id in self.term_frequencies:
            if token[0] in self.term_frequencies[doc_id]:
                return self.term_frequencies[doc_id][token[0]]
            else:
                return 0
        else:
            return 0
        
    def tf_command(self, doc_id: int, term: str):
        self.load()
        tf = self.get_tf(doc_id, term)
        return tf
        
    def get_idf(self, term:str):
        token = keyword_search.tokenize_text(term)
        if not self.is_single_token(token):
            raise Exception("Error: More than one token detected.")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf
    
    def idf_command(self, term: str):
        self.load()
        idf = self.get_idf(term)
        return idf
    
    def get_tf_idf(self, doc_id: int, term: str):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def tf_idf_command(self, doc_id: int, term: str):
        self.load()
        tf_idf = self.get_tf_idf(doc_id, term)
        return tf_idf
    
    def get_bm25_idf(self, term: str) -> float:
        token = keyword_search.tokenize_text(term)
        if not self.is_single_token(token):
            raise Exception("Error: More than one token detected.")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        bm25_idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
        return bm25_idf
    
    def bm25_idf_command(self, term: str):
        self.load()
        bm25_idf = self.get_bm25_idf(term)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        token = keyword_search.tokenize_text(term)
        if not self.is_single_token:
            raise Exception("Error: More than one token detected.")
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def bm25_tf_command(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        self.load()
        bm25_tf = self.get_bm25_tf(doc_id, term, k1, b)
        return bm25_tf
    
    def bm25(self, doc_id: int, term: str):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        score = bm25_tf * bm25_idf
        return score

    def bm25_search(self, query: str, limit=DEFAULT_SEARCH_LIMIT):
        query_tokens = keyword_search.tokenize_text(query)
        scores = {}
        for doc in self.docmap:
            doc_score = 0
            for token in query_tokens:
                doc_score += self.bm25(doc, token)
            if doc_score > 0:
                scores[doc] = doc_score
        sorted_scores = sorted(scores.items(), key=lambda item:item[1], reverse=True)
        score_limit = dict(sorted_scores[:limit])
        return score_limit
    
    def bm25_search_command(self, query: str, limit=DEFAULT_SEARCH_LIMIT):
        self.load()
        results = self.bm25_search(query, limit)
        return results