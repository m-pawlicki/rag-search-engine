from keyword_search import tokenize_text
from search_utils import CACHE_PATH, load_movies
import os, pickle

class InvertedIndex():
    # {token:str, doc_ids: set(int)}
    index = {}
    # {doc_id: int, doc_obj: obj?}
    docmap = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

    def get_documents(self, term: str):
        lowercase_term = term.lower()
        if lowercase_term in self.index:
            return sorted(list(self.index[lowercase_term]))
        else:
            return []

    def build(self):
        movies = load_movies()
        for movie in movies:
            index_data = f"{movie["title"]} {movie["description"]}"
            self.__add_document(movie["id"], index_data)
            self.docmap[movie["id"]] = movie

    def save(self):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)

        with open(f"{CACHE_PATH}/index.pkl", "wb") as idx:
            pickle.dump(self.index, idx)

        with open(f"{CACHE_PATH}/docmap.pkl", "wb") as doc:
            pickle.dump(self.docmap, doc)