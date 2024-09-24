from fastapi import FastAPI
from api.routers import router

app = FastAPI()

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Welcome to the RAG service."}
