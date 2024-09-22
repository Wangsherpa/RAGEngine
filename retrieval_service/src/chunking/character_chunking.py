from src.chunking.base_chunker import ChunkingStrategy
from typing import List


class CharacterChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split the document into chunks based on a specific character count and overlap.

        Args:
            text (str): text to chunk.

        Returns:
            list: list of chunked texts.
        """
        if len(text) < self.chunk_size:
            return [text]
        chunks = []
        index = 0
        while index < len(text):
            chunk = text[index : index + self.chunk_size]
            chunks.append(chunk)
            index += self.chunk_size - self.overlap
        return chunks
