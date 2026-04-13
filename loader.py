import os


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                text = f.read()
            documents.append({"filename": filename, "text": text})
    return documents


def chunk_document(doc, chunk_size, chunk_overlap):
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    text = doc["text"]
    filename = doc["filename"]
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "chunk_id": f"{filename}_chunk_{chunk_index}",
            "filename": filename,
            "text": chunk_text
        })
        chunk_index += 1
        start = end - chunk_overlap

    return chunks


def load_and_chunk(directory, chunk_size, chunk_overlap):
    documents = load_documents(directory)
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"{doc['filename']} → {len(chunks)} chunks")
    return all_chunks
