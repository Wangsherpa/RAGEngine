import sys
import argparse

sys.path.append(".")

from chromadb import Documents, EmbeddingFunction, Embeddings
from src.document_loader.document_loader import DocumentLoader
from src.chunking.character_chunking import CharacterChunking
from src.vector_stores.chroma_db import ChromaDB
from src.embeddings.transformer_embedding import TransformerEmbeddingGenerator
from src.controller import RetrievalEngine
from config.vectordb_config import vectordb_config


# Using custom embedding function with chromadb
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return self.embedder.generate_embedding(input)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document",
        default="data/sample_data.txt",
        help="Path to the document to process",
    )
    parser.add_argument(
        "--query",
        default="PL Techniques",
        help="Query text for retrieval",
    )
    parser.add_argument(
        "--process_doc",
        type=bool,
        default=False,
        help="set as false if doc is already processed",
    )
    args = parser.parse_args()

    chroma_config = vectordb_config["chroma"]
    with open(args.document, "r", encoding="utf-8") as f:
        doc = f.read()

    char_chunker = CharacterChunking(chunk_size=500, overlap=100)
    embedding_model_name = "bert-base-uncased"
    # embedding_fn = MyEmbeddingFunction(embedder=embedding_generator)
    vector_store = ChromaDB(config=chroma_config, embedding_fn=None)

    retrieval_engine = RetrievalEngine(
        vector_store=vector_store,
        chunking_strategy=char_chunker,
        embedding_model_name=embedding_model_name,
    )
    if args.process_doc:
        retrieval_engine.process_document(doc, document_name="test_name")
    results = retrieval_engine.retrieve_chunks(query_text=args.query)
    for i, doc in enumerate(results["documents"]):
        print(f"### {i+1}")
        print(doc)
        print()


if __name__ == "__main__":
    main()
