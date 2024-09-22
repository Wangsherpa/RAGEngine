import torch
from transformers import AutoModel, AutoTokenizer
from src.embeddings.base_embedder import EmbeddingGenerator


class TransformerEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_embedding(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling the output tokens to create single embedding vector
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding
