import os
from dotenv import load_dotenv
from loader import load_and_chunk
from embedder import load_model, generate_embeddings
from indexer import get_collection, index_chunks

load_dotenv()

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "velorath"


def main():
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    collection = get_collection(CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME)

    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks, skipping indexing.")
        return

    print("Loading and chunking documents...")
    all_chunks = load_and_chunk(KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Total chunks: {len(all_chunks)}\n")

    model = load_model(EMBEDDING_MODEL)
    embeddings = generate_embeddings(all_chunks, model)
    print(f"Embeddings shape: {embeddings.shape}\n")

    index_chunks(all_chunks, embeddings, collection)


if __name__ == "__main__":
    main()
