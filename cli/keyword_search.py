from search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def keyword_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    results = []
    movies = load_movies()

    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_token_match(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    string_out = f"Searching for: {query}"

    if len(results) == 0:
        string_out += "\nNo results found!"
    else:
        for i in range(len(results)):
            string_out += f"\n{i+1}. {results[i]["title"]}"

    print(string_out)

def preprocess_text(text: str):
    strip_punctuation = str.maketrans("", "", string.punctuation)
    text = text.lower()
    text = text.translate(strip_punctuation)
    return text

def tokenize_text(text: str):
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens

def has_token_match(query_tokens, title_tokens):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False