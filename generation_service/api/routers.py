from fastapi import APIRouter
from src.generation import generate_response
from api.models import GenerationRequest, GenerationResponse

router = APIRouter()


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(gen_req: GenerationRequest):
    """
    Endpoint to generate a response using a generative model based on the provided context.
    """
    response = generate_response(gen_req.query, gen_req.context)
    return GenerationResponse(answer=response)
