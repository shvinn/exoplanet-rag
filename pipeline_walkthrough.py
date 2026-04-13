import os
from dotenv import load_dotenv
from loader import load_and_chunk
from embedder import load_model, generate_embeddings
from indexer import get_collection, index_chunks
from retriever import retrieve
from prompt_builder import build_prompt, SYSTEM_PROMPT
from llm import get_client, generate_answer

load_dotenv()

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "velorath"


def main():
    # Step 1 - Load and chunk
    print("Loading and chunking documents...")
    all_chunks = load_and_chunk(KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Total chunks: {len(all_chunks)}\n")

    # Step 2 - Generate embeddings
    model = load_model(EMBEDDING_MODEL)
    embeddings = generate_embeddings(all_chunks, model)
    print(f"Embeddings shape: {embeddings.shape}\n")

    # Step 3 - Index into ChromaDB
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    collection = get_collection(CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME)

    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks, skipping indexing.")
    else:
        index_chunks(all_chunks, embeddings, collection)

    # Step 4 - Retrieve relevant chunks
    question = "What is the gravity on Velorath?"
    chunks = retrieve(question, collection, model, top_k=3)

    # Step 5 - Build prompt
    user_message = build_prompt(question, chunks)

    # Step 6 - Generate answer
    print("\n--- Step 6: LLM Answer ---")
    client = get_client(os.getenv("OPENAI_API_KEY"))
    answer = generate_answer(client, SYSTEM_PROMPT, user_message)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
