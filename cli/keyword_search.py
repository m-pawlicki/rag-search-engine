from search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def keyword_search(query, limit = DEFAULT_SEARCH_LIMIT):
    results = []
    movies = load_movies()

    for entry in movies:
        processed_query = process_text(query)
        processed_title = process_text(entry["title"])
        if  processed_query in processed_title:
            results.append(entry)
            if len(results) >= limit:
                break

    string_out = f"Searching for: {query}"

    if len(results) == 0:
        string_out += "\nNo results found!"
    else:
        for i in range(len(results)):
            string_out += f"\n{i+1}. {results[i]["title"]}"

    print(string_out)

def process_text(text):
    text = text.lower()
    return text