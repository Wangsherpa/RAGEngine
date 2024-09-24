import requests
from config.config import config


def send_document_to_retrieval_service(
    document_text: str, chunking_strategy: str = "character"
):
    """
    Send the extracted document text to the retrieval service for processing.

    Args:
        document_text (str): The extracted text from the document.
        chunking_strategy (str): The chunking strategy (default is 'character').
    Returns:
        dict: The response from the retrieval service.
    """

    url = f"{config['endpoint']['retrieval']}/process-document"
    data = {"document": document_text, "chunking_strategy": chunking_strategy}
    response = requests.post(url, json=data)
    return response.json()


def query_retrieval_service(query: str, top_k: int):
    """
    Query the retrieval service to get the best-matching chunks for the user's query.

    Args:
        query (str): The user's query text.
    Returns:
        dict: The retrieved chunks from the retrieval service.
    """
    url = f"{config['endpoint']['retrieval']}/query"
    data = {"query": query, "top_k": top_k}
    response = requests.post(url, json=data)
    return response.json()
