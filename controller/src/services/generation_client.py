import requests
from config.config import config


def get_answer(query: str, context: str):
    """Send the query and context to the generation service to generate a response.

    Args:
        query (str): The user's query text.
        context (str): The context created using retrieval service.

    Returns:
        dict: The response from the generation service.
    """
    url = f"{config['endpoint']['generation']}/generate"
    data = {"query": query, "context": context}
    response = requests.post(url, json=data)
    return response.json()
