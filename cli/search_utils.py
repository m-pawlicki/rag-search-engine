import json

DEFAULT_SEARCH_LIMIT = 5
DATA_PATH = "./data/movies.json"

def load_movies():
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]