from abc import ABC, abstractmethod
from typing import List

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: str) -> List[str]:
        """Abstract method for chunking a document into smaller chunks.

        Args:
            document (str): Document to be chunked.

        Returns:
            List[str]: Document chunks.
        """
        pass