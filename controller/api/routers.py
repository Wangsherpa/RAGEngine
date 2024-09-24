from fastapi import APIRouter, UploadFile, File, HTTPException
from src.document_loaders.document_loader import DocumentLoader
from src.services.retrieval_client import (
    send_document_to_retrieval_service,
    query_retrieval_service,
)
from src.services.generation_client import get_answer

router = APIRouter()


@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...), pdf_processor: str = "pymupdf"):
    """Upload a document (TXT or PDF), process it, and send the text to the retrieval service.

    Args:
        file (UploadFile): The uploaded file (TXT or PDF).
        pdf_processor (str, optional): PDF processing tool ('pymupdf' or 'pdfminer'). Defaults to 'pymupdf'.

    Raises:
        HTTPException: If an unsupported file type is uploaded or an invalid PDF processor is selected.

    Returns:
        dict: The response from the retrieval service after processing the document.
    """
    try:
        loader = DocumentLoader(pdf_processor=pdf_processor)
        document_text = await loader.load_document(file)
        retrieval_response = send_document_to_retrieval_service(document_text)
        return retrieval_response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/process-query")
async def process_query(query: str, top_k: int = 3):

    # Retrieve the best-matching chunks
    retrieval_resp = query_retrieval_service(query=query, top_k=top_k)
    print(retrieval_resp)
    retrieved_chunks = retrieval_resp["top_results"].get("documents", [])

    if retrieved_chunks:
        # Create a context from the retrieve chunks
        context = "\n".join(retrieved_chunks)
        # Send query and context to the generation service
        generation_response = get_answer(query=query, context=context)
        return {
            "retrieved_chunks": retrieved_chunks,
            "generated_response": generation_response.get("answer"),
        }
    else:
        return {"error": "No relevant chunks found for the query."}
