import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
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
DATASET_PATH = os.path.join(os.path.dirname(__file__), "test_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def llm_judge(client, prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def score_faithfulness(client, context, answer):
    """Fraction of answer sentences that are supported by the retrieved context."""
    result = llm_judge(client, f"""You are evaluating whether an AI-generated answer is faithful to its source context.

Context:
{context}

Answer:
{answer}

For each sentence in the Answer, determine whether it is fully supported by information present in the Context.
Return JSON with:
- "supported": integer — number of sentences supported by the context
- "total": integer — total number of sentences in the answer""")
    total = result.get("total", 0)
    return result.get("supported", 0) / total if total > 0 else 0.0


def score_answer_relevancy(client, embed_model, question, answer, n=3):
    """Mean cosine similarity between hypothetical questions generated from the answer and the original question."""
    result = llm_judge(client, f"""Given the following answer, generate {n} different questions that this answer could be responding to.

Answer:
{answer}

Return JSON with:
- "questions": list of {n} question strings""")
    hypothetical_qs = result.get("questions", [])
    if not hypothetical_qs:
        return 0.0

    q_emb = embed_model.encode([question])
    h_embs = embed_model.encode(hypothetical_qs)
    sims = np.dot(h_embs, q_emb.T).flatten() / (
        np.linalg.norm(h_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8
    )
    return float(np.mean(sims))


def score_context_recall(client, context, ground_truth):
    """Fraction of ground truth sentences that can be attributed to the retrieved context."""
    result = llm_judge(client, f"""You are evaluating whether retrieved context contains enough information to produce a ground truth answer.

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

For each sentence in the Ground Truth Answer, determine whether it can be attributed to the Retrieved Context.
Return JSON with:
- "attributable": integer — number of ground truth sentences attributable to the context
- "total": integer — total number of sentences in the ground truth""")
    total = result.get("total", 0)
    return result.get("attributable", 0) / total if total > 0 else 0.0


def score_context_precision(client, question, chunks):
    """Fraction of retrieved chunks that are useful for answering the question."""
    useful = 0
    for chunk in chunks:
        result = llm_judge(client, f"""You are evaluating whether a retrieved context chunk is useful for answering a question.

Question:
{question}

Retrieved Chunk:
{chunk['text']}

Is this chunk useful for answering the question?
Return JSON with:
- "useful": boolean""")
        if result.get("useful", False):
            useful += 1
    return useful / len(chunks) if chunks else 0.0


def main():
    print("Loading components...")
    embed_model = load_model(EMBEDDING_MODEL)
    collection = get_collection(CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME)
    client = get_client(os.getenv("OPENAI_API_KEY"))

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    results = []

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"\n[{i + 1}/{len(dataset)}] {question}")

        chunks = retrieve(question, collection, embed_model, top_k=TOP_K)
        user_message = build_prompt(question, chunks)
        answer = generate_answer(client, SYSTEM_PROMPT, user_message, history=[])
        context = "\n\n".join(chunk["text"] for chunk in chunks)

        faithfulness = score_faithfulness(client, context, answer)
        answer_relevancy = score_answer_relevancy(client, embed_model, question, answer)
        context_recall = score_context_recall(client, context, ground_truth)
        context_precision = score_context_precision(client, question, chunks)

        print(f"  Faithfulness:       {faithfulness:.2f}")
        print(f"  Answer Relevancy:   {answer_relevancy:.2f}")
        print(f"  Context Recall:     {context_recall:.2f}")
        print(f"  Context Precision:  {context_precision:.2f}")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision,
        })

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    aggregate = {m: float(np.mean([r[m] for r in results])) for m in metrics}

    print("\n" + "=" * 50)
    print("AGGREGATE SCORES")
    print("=" * 50)
    for metric, score in aggregate.items():
        print(f"  {metric:<22} {score:.3f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"per_question": results, "aggregate": aggregate}, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
