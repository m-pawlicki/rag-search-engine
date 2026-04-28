from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from lib.inverted_index import InvertedIndex
from nltk.stem import PorterStemmer
import string

def keyword_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[str]:
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