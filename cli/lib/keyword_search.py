from lib.search_utils import DEFAULT_SEARCH_LIMIT, CACHE_PATH, BM25_K1, BM25_B, load_movies, load_stopwords
from nltk.stem import PorterStemmer
import os, pickle, sys, math, string
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

        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_len_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
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

        with open(self.index_path, "wb") as idx:
            pickle.dump(self.index, idx)

        with open(self.docmap_path, "wb") as doc:
            pickle.dump(self.docmap, doc)

        with open(self.term_freq_path, "wb") as tf:
            pickle.dump(self.term_frequencies, tf)

        with open(self.doc_len_path, "wb") as dl:
            pickle.dump(self.doc_lengths, dl)

    def load(self):
        try:
            with open(self.index_path, "rb") as idx:
                self.index = pickle.load(idx)

            with open(self.docmap_path, "rb") as doc:
                self.docmap = pickle.load(doc)

            with open(self.term_freq_path, "rb") as tf:
                self.term_frequencies = pickle.load(tf)

            with open(self.doc_len_path, "rb") as dl:
                self.doc_lengths = pickle.load(dl)

        except FileNotFoundError:
            print("Error loading file(s)! Do they exist?")
            sys.exit(1)

    def is_single_token(self, token: list[str]):
        if len(token) > 1:
            return False
        return True

    def get_tf(self, doc_id: int, term: str):
        token = tokenize_text(term)
        if not self.is_single_token(token):
            raise ValueError("Error: More than one token detected.")
        if doc_id in self.term_frequencies:
            if token[0] in self.term_frequencies[doc_id]:
                return self.term_frequencies[doc_id][token[0]]
            else:
                return 0
        else:
            return 0
        
    def get_idf(self, term:str):
        token = tokenize_text(term)
        if not self.is_single_token(token):
            raise ValueError("Error: More than one token detected.")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf
    
    def get_tf_idf(self, doc_id: int, term: str):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if not self.is_single_token(token):
            raise ValueError("Error: More than one token detected.")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        bm25_idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        token = tokenize_text(term)
        if not self.is_single_token(token):
            raise ValueError("Error: More than one token detected.")
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
    
    def bm25(self, doc_id: int, term: str):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        score = bm25_tf * bm25_idf
        return score

    def bm25_search(self, query: str, limit=DEFAULT_SEARCH_LIMIT):
        query_tokens = tokenize_text(query)
        scores = {}

        for doc in self.docmap:
            doc_score = 0
            for token in query_tokens:
                doc_score += self.bm25(doc, token)
            if doc_score > 0:
                scores[doc] = doc_score

        sorted_scores = sorted(scores.items(), key=lambda item:item[1], reverse=True)
        results = []

        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            entry = {"id": doc_id, "title": doc["title"], "score": score}
            results.append(entry)

        return results
    
def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    
def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    return tf

def idf_command(term: str):
    idx = InvertedIndex
    idx.load()
    idf = idx.get_idf(term)
    return idf

def tf_idf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tf_idf(doc_id, term)
    return tf_idf

def bm25_idf_command(term: str):
    idx = InvertedIndex()
    idx.load()
    bm25_idf = idx.get_bm25_idf(term)
    return bm25_idf

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.get_bm25_tf(doc_id, term, k1, b)
    return bm25_tf

def bm25_search_command(query: str, limit=DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    results = idx.bm25_search(query, limit)
    return results

def keyword_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    results = []
    indexer = InvertedIndex()
    indexer.load()
    query_tokens = tokenize_text(query)
    seen = set()

    for token in query_tokens:
        doc_matches = indexer.get_documents(token)

        if len(results) >= limit:
            break

        for match in doc_matches:
            if match in seen:
                continue
            seen.add(match)
            entry = {"id": match, "title": indexer.docmap[match]["title"]}
            results.append(entry)
            if len(results) >= limit:
                return results
            
    return results

def preprocess_text(text: str) -> str:
    strip_punctuation = str.maketrans("", "", string.punctuation)
    text = text.lower()
    text = text.translate(strip_punctuation)
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_tokens = remove_stopwords(valid_tokens)
    stemmed_tokens = stem_tokens(filtered_tokens)
    return stemmed_tokens

def has_token_match(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def remove_stopwords(tokens: list[str]) -> list[str]:
    stopwords = load_stopwords()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stems = []
    for token in tokens:
        stem = stemmer.stem(token)
        stems.append(stem)
    return stems