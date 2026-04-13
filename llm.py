from openai import OpenAI


def get_client(api_key):
    return OpenAI(api_key=api_key)


def generate_answer(client, system_prompt, user_message, history, model="gpt-4o-mini"):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content
    return answer
