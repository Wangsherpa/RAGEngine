from abc import ABC, abstractmethod
from typing import List

class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate_embedding(self, text: str) -> list:
        """Generate embeddings for the given text using a transformer model.

        Args:
            text (str): Text for which embeddings will be generated.

        Returns:
            list: Embeddings for a given text.
        """
        pass