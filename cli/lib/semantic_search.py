from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input cannot be empty.")
        embedding = self.model.encode([text])
        return embedding[0]


def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")