# Exoplanet RAG

A Retrieval-Augmented Generation (RAG) pipeline that lets you query a knowledge base about the fictional exoplanet **Velorath** using natural language.

## What is RAG?

RAG is a technique that grounds LLM responses to a specific set of documents. Instead of relying on the model's training data, it:

1. Searches your knowledge base for relevant context
2. Injects that context into the prompt
3. The LLM answers based only on what was retrieved

This prevents hallucinations and allows the LLM to answer questions about private or fictional data it was never trained on.

## Pipeline

```
Knowledge Base (.txt files)
        ↓
   loader.py        — load and split documents into chunks
        ↓
   embedder.py      — convert chunks to vectors (sentence-transformers)
        ↓
   indexer.py       — store vectors + text in ChromaDB
        ↓
   [ChromaDB]       — persistent vector store (Docker)
        ↓
   retriever.py     — embed user question, fetch top-k similar chunks
        ↓
   prompt_builder.py — assemble context + question into enriched prompt
        ↓
   llm.py           — send prompt to OpenAI gpt-4o-mini, return answer
```

## Tech Stack

| Component | Technology |
|---|---|
| Document chunking | Plain Python |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (Docker) |
| LLM | OpenAI `gpt-4o-mini` |
| Dependency management | `uv` |

## Knowledge Base

9 documents about the exoplanet Velorath covering atmosphere, fauna, flora, habitability, terrain, resources, weather, and exploration history.

## Setup

1. Start ChromaDB:
```bash
docker compose up -d
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

3. Install dependencies:
```bash
uv sync
```

4. Index the knowledge base (run once):
```bash
uv run index.py
```

5. Start the interactive CLI:
```bash
uv run query.py
```

## Evaluation

The pipeline is evaluated using RAGAS-style metrics over a manually written test dataset of 18 Q&A pairs covering all 9 knowledge base documents.

### Metrics

| Metric | What it measures | Method |
|---|---|---|
| Faithfulness | Are all answer claims supported by the retrieved context? | LLM-as-judge |
| Answer Relevancy | Does the answer address the question asked? | Embedding similarity |
| Context Recall | Does the retrieved context contain enough to answer the ground truth? | LLM-as-judge |
| Context Precision | Of the retrieved chunks, how many were actually useful? | LLM-as-judge |

### Baseline Results (top-k=2)

| Metric | Score |
|---|---|
| Faithfulness | 0.917 |
| Answer Relevancy | 0.728 |
| Context Recall | 0.903 |
| Context Precision | 0.639 |

### Running Evaluation

```bash
uv run eval/evaluate.py
```

Results are saved to `eval/results.json`.

## Project Structure

```
exoplanet-rag/
  ├── index.py              ← run once to index the knowledge base
  ├── query.py              ← interactive CLI (entry point)
  ├── loader.py             ← document loading and chunking
  ├── embedder.py           ← embedding model wrapper
  ├── indexer.py            ← ChromaDB connection and indexing
  ├── retriever.py          ← similarity search
  ├── prompt_builder.py     ← prompt assembly
  ├── llm.py                ← LLM client and answer generation
  ├── pipeline_walkthrough.py ← step-by-step build walkthrough
  ├── docker-compose.yml    ← ChromaDB service
  ├── knowledge_base/       ← source documents
  └── eval/
        ├── test_dataset.json ← 18 Q&A pairs for evaluation
        ├── evaluate.py       ← evaluation script
        └── results.json      ← latest scores (generated)
```
