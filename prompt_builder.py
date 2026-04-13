SYSTEM_PROMPT = "You are an expert on the exoplanet Velorath. Answer the user's question using only the context provided. If the answer is not in the context, say you don't have enough information."


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source: {chunk['filename']}]\n{chunk['text'].strip()}" for chunk in chunks
    )

    user_message = f"""Context:
{context}

Question: {question}"""

    return user_message
