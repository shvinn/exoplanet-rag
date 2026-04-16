import os
from dotenv import load_dotenv
from embedder import load_model
from indexer import get_collection
from retriever import retrieve
from prompt_builder import build_prompt, SYSTEM_PROMPT
from llm import get_client, generate_answer

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "velorath"
TOP_K = 2


def main():
    print("Loading embedding model...")
    model = load_model(EMBEDDING_MODEL)

    print(f"Connecting to ChromaDB...")
    collection = get_collection(CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME)

    client = get_client(os.getenv("OPENAI_API_KEY"))

    print("\nVelorath RAG — ask anything. Type 'exit' to quit.\n")

    history = []

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        chunks = retrieve(question, collection, model, top_k=TOP_K)
        user_message = build_prompt(question, chunks)
        answer = generate_answer(client, SYSTEM_PROMPT, user_message, history)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})

        print(f"Velorath: {answer}\n")


if __name__ == "__main__":
    main()
