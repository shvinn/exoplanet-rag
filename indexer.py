import chromadb


def get_collection(host, port, collection_name):
    client = chromadb.HttpClient(host=host, port=port)
    collection = client.get_or_create_collection(collection_name)
    print(f"Collection '{collection_name}' ready")
    return collection


def index_chunks(chunks, embeddings, collection):
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"filename": chunk["filename"]} for chunk in chunks],
        embeddings=embeddings.tolist()
    )
    print(f"Indexed {collection.count()} chunks into ChromaDB")
