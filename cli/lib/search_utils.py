import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

def load_movies():
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH) as file:
        data = file.read()
        lines = data.splitlines()
    return lines