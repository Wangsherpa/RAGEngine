from fastapi import APIRouter, HTTPException
from src.retrieval import RetrievalEngine
from src.chunking.character_chunking import CharacterChunking
from src.vector_stores.chroma_db import ChromaDB
from api.models import DocumentRequest, QueryRequest, QueryResponse
from config.vectordb_config import vectordb_config

router = APIRouter()

chroma_config = vectordb_config["chroma"]

# Initialize vector store
vector_store = ChromaDB(chroma_config, embedding_fn=None)

# Chunking strategies TODO: add more
chunking_strategies = {
    "character": CharacterChunking(chunk_size=1000, overlap=100),
    "semantic": CharacterChunking(chunk_size=1000, overlap=100),  # TODO: update
}

# Initialize Retrieval Engine with default values
retrieval_engine = RetrievalEngine(
    vector_store,
    chunking_strategy=chunking_strategies["character"],
    embedding_model_name="bert-base-uncased",
)


@router.post("/process-document")
async def process_document(doc_req: DocumentRequest):
    """Api endpoint to process a document, chunk it and store in the vector database."""
    if doc_req.chunking_strategy not in chunking_strategies:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")

    # Update Retrieval engine with the  chosen chunking strategy
    retrieval_engine.chunking_strategy = chunking_strategies[doc_req.chunking_strategy]

    # Process the document
    retrieval_engine.process_document(doc_req.document, "")

    return {"status": "Document processed successfully"}


@router.post("/query", response_model=QueryResponse)
async def retrieve_chunks(query_req: QueryRequest):
    """API endpoint to query the RAG engine and retrieve the best chunks."""
    top_results = retrieval_engine.retrieve_chunks(
        query_text=query_req.query, top_k=query_req.top_k
    )
    return QueryResponse(top_results=top_results)
