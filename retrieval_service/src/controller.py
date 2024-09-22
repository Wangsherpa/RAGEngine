from src.chunking.base_chunker import ChunkingStrategy
from src.vector_stores.base_store import VectorStore
from src.embeddings.transformer_embedding import TransformerEmbeddingGenerator


class RetrievalEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        chunking_strategy: ChunkingStrategy,
        embedding_model_name: str,
    ):
        self.vector_store = vector_store
        self.chunking_strategy = chunking_strategy
        self.embedding_generator = TransformerEmbeddingGenerator(
            model_name=embedding_model_name
        )

    def process_document(self, document: str, document_name: str):
        """Chunk the document, generate embeddings, and store them in the vector store.

        Args:
            document (str): document to process.
        """
        chunks = self.chunking_strategy.chunk(document)
        embeddings = [
            self.embedding_generator.generate_embedding(chunk) for chunk in chunks
        ]
        metadata = [{"file_name": document_name} for _ in chunks]
        self.vector_store.store_vectors(chunks, embeddings, metadata)

    def retrieve_chunks(self, query_text: str, top_k: int = 3):
        """
        Retrieve the best matching document chunks for a given user query.
        """
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        results = self.vector_store.search_vectors(query_embedding, top_k=top_k)
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
        }
