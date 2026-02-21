from typing import List
from FlagEmbedding import BGEM3FlagModel

class EmbedService:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        print(f"Loading embedding model: {model_name}...")
        # BGE-M3 supports dense, sparse, and multi-vector (colbert) embeddings.
        # We will load it to utilize both dense and sparse representations if needed.
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        print("Embedding model loaded.")

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text string into dense vectors.
        """
        # Return only dense embeddings for standard use
        embeddings = self.model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return embeddings['dense_vecs'].tolist()
        
    def embed_text_hybrid(self, texts: List[str]):
        """
        Embed a list of text into both dense and sparse vectors for advanced hybrid search.
        """
        embeddings = self.model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return embeddings
