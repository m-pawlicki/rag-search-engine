from lib.search_utils import CLIENT, MODEL

def spell_enhance(query):
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

def spell_enhance_command(query):
    enhanced_query = spell_enhance(query)
    return enhanced_query