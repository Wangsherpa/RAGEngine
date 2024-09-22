import sys

sys.path.append("../../")
import uuid
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from tqdm import tqdm
from src.vector_stores.base_store import VectorStore


class ChromaDB(VectorStore):
    def __init__(self, config: dict, embedding_fn: chromadb.EmbeddingFunction = None):
        self.client = chromadb.PersistentClient(config["db_path"])
        self.collection = self.client.get_or_create_collection(
            config["collection_name"], embedding_function=embedding_fn
        )

    def store_vectors(
        self, chunks: List[str], vectors: List[List[float]], metadata: List[Dict]
    ) -> None:

        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        self.collection.add(
            documents=chunks, embeddings=vectors, metadatas=metadata, ids=ids
        )

    def search_vectors(self, vector: str, top_k: int = 3) -> List[Dict]:
        search_results = self.collection.query(query_embeddings=vector, n_results=top_k)
        return search_results
