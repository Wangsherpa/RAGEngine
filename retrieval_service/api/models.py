from pydantic import BaseModel
from typing import Optional


class DocumentRequest(BaseModel):
    document: str
    chunking_strategy: str = "character"


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    top_results: dict
