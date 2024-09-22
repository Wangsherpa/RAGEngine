from fastapi import FastAPI
from .routers import router

# Initialize the FastAPI app
app = FastAPI()

# Include API router for endpoints
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Engine!"}
