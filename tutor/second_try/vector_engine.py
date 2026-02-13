import torch
from sentence_transformers import SentenceTransformer
import faiss
from typing import List

class VectorEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing Vector Engine on device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.chunks = list()

    def build_index(self, chunks: List):
        self.chunks = chunks
        texts = [c['text'] for c in chunks]
        
        print("Generating embeddings")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        
        # Initialize FAISS
        dimension = embeddings.shape
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Index built. Total vectors: {self.index.ntotal}")

    def search(self, query: str, k: int = 5):
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = list()
        for idx, dist in zip(indices, distances):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]['text'],
                    "distance": float(dist),
                    "source": self.chunks[idx]['source']
                })
        return results