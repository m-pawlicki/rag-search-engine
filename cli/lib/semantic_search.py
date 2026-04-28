from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)

def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")