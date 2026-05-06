import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

# Controls diminishing returns/saturation
BM25_K1 = 1.5
# Normalization strength/how much we care about document length
BM25_B = 0.75
# Weighting of keyword to semantic results, 1.0 = 100% keyword, 0.0 = 100% semantic
ALPHA = 0.5
# k value for RRF (reciprocal rank fusion), how much weight we give to higher-ranked results vs lower-ranked results, eg. lower k = more weight, higher k = less weight
K_VALUE = 60

def load_movies():
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH) as file:
        data = file.read()
        lines = data.splitlines()
    return lines