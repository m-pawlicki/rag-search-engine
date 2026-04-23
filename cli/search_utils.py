import json, io

DEFAULT_SEARCH_LIMIT = 5
DATA_PATH = "./data/movies.json"
STOPWORDS_PATH = "./data/stopwords.txt"

def load_movies():
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH) as file:
        data = file.read()
        lines = data.splitlines()
    return lines