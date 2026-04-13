from sentence_transformers import SentenceTransformer


def load_model(model_name):
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def generate_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
