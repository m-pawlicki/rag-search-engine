import keyword_search
from search_utils import CACHE_PATH, load_movies
import os, pickle, sys
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

    def get_tf(self, doc_id: int, term: str):
        token = keyword_search.tokenize_text(term)
        if len(token) > 1:
            raise Exception("Error: More than one token detected.")
        if doc_id in self.term_frequencies:
            if token[0] in self.term_frequencies[doc_id]:
                return self.term_frequencies[doc_id][token[0]]
        else:
            return 0