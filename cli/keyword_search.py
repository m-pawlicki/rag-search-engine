import json

def keyword_search(term):
    results = []
    data_path = "./data/movies.json"

    with open(data_path, "r") as file:
        data = json.load(file)

    for entry in data["movies"]:
        if term in entry["title"]:
            results.append(entry)
    
    string_out = f"Searching for: {term}"
    search_limit = 5

    if len(results) == 0:
        string_out += "\nNo results found!"
    else:
        for i in range(search_limit):
            string_out += f"\n{i+1}. {results[i]["title"]}"

    print(string_out)