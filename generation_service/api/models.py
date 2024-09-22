from pydantic import BaseModel


class GenerationRequest(BaseModel):
    query: str
    context: str


class GenerationResponse(BaseModel):
    answer: str
