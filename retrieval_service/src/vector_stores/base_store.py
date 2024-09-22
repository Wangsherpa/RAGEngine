from abc import ABC, abstractmethod
from typing import List, Dict


class VectorStore(ABC):
    @abstractmethod
    def store_vectors(
        self, chunks: List[str], vectors: List[List[float]], metadata: List[Dict]
    ) -> None:
        """
        Store the list of chunks in the vector database.
        """
        pass

    @abstractmethod
    def search_vectors(self, vectors: List[List[float]], top_k: int = 3) -> List[Dict]:
        """Search the top-k closest vectors to the query vector.

        Args:
            query_vector (str): Query text.
            top_k (int, optional): Number of similar results to return. Defaults to 3.

        Returns:
            List[Dict]: top-k similar results.
        """
        pass
