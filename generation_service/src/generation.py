from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def generate_response(query: str, context: str) -> str:
    """
    Generate a response based on the provided query and context using LLM.
    """
    prompt = f"Answer the following question based on the context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
    )

    return response.choices[0].message.content
