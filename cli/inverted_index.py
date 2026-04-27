import keyword_search
from search_utils import CACHE_PATH, load_movies, BM25_K1
import os, pickle, sys, math
from collections import defaultdict, Counter

class InvertedIndex():
    def __init__(self):
        # {token:str, doc_ids: set(int)}
        self.index = defaultdict(set)
        # {doc_id: int, doc_obj: dict}
        self.docmap: dict[int, dict] = {}
        # {doc_id: int, count: Counter()}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str):
        tokens = keyword_search.tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

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

    def load(self):
        try:
            with open(f"{CACHE_PATH}/index.pkl", "rb") as idx:
                self.index = pickle.load(idx)

            with open(f"{CACHE_PATH}/docmap.pkl", "rb") as doc:
                self.docmap = pickle.load(doc)

            with open(f"{CACHE_PATH}/term_frequencies.pkl", "rb") as tf:
                self.term_frequencies = pickle.load(tf)

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
        bm25idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
        return bm25idf
    
    def bm25_idf_command(self, term: str):
        self.load()
        bm25_idf = self.get_bm25_idf(term)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1):
        token = keyword_search.tokenize_text(term)
        if not self.is_single_token:
            raise Exception("Error: More than one token detected.")
        tf = self.get_tf(doc_id, term)
        saturated_tf = (tf * (k1 + 1)) / (tf + k1)
        return saturated_tf

    def bm25_tf_command(self, doc_id: int, term: str, k1=BM25_K1):
        self.load()
        bm25_tf = self.get_bm25_tf(doc_id, term, k1)
        return bm25_tf