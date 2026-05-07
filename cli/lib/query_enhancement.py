from lib.search_utils import CLIENT, MODEL

def spell_enhancement(query):
    client = CLIENT
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text

def rewrite_enhancement(query):
    client = CLIENT
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """
    
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text

def spell_enhance_command(query):
    enhanced_query = spell_enhancement(query)
    return enhanced_query

def rewrite_enhance_command(query):
    enhanced_query = rewrite_enhancement(query)
    return enhanced_query