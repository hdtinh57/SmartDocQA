import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
from core.config import settings

class VDBService:
    def __init__(self, collection_name: str = "smart_doc_qa"):
        self.collection_name = collection_name
        
        # Connect to local Qdrant memory/disk or URL if specified
        if "localhost" in settings.qdrant_url:
            # We can use memory mode for dev or connect to local docker
            print(f"Connecting to Local Qdrant: {settings.qdrant_url}")
            # Try connecting to url, fallback to memory if not running
            try:
                self.client = QdrantClient(url=settings.qdrant_url)
                self.client.get_collections() # test connection
            except Exception:
                print("Local Qdrant Server not found, falling back to memory/disk mode.")
                self.client = QdrantClient(path="./data/qdrant_storage")
        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
            
        self._ensure_collection()
        
    def _ensure_collection(self):
        """Create collection if it doesn't exist. BAAI/bge-m3 default dimension is 1024."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"Creating collection '{self.collection_name}' with dimension 1024...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            
    def upsert_chunks(self, chunks: List[str], embeddings_dense: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Insert extracted text chunks into Qdrant."""
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in chunks]
            
        points = []
        for i, (chunk, vector, meta) in enumerate(zip(chunks, embeddings_dense, metadatas)):
            # Tự thêm chunk text vào metadata để query có thể trả về
            meta_copy = meta.copy()
            meta_copy["text"] = chunk
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=meta_copy
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} chunks into {self.collection_name}.")
        
    def search(self, query_vector: List[float], limit: int = 5, allowed_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search similar vectors and return the payload."""
        search_filter = None
        if allowed_sources is not None and len(allowed_sources) > 0:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(any=allowed_sources)
                    )
                ]
            )
        elif allowed_sources is not None and len(allowed_sources) == 0:
            # If allowed_sources list is explicitly empty, return nothing
            return []
            
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=search_filter,
            score_threshold=0.2 # Filter out completely irrelevant vectors
        ).points
        
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload
            })
            
        return results

    def has_document(self, source_name: str) -> bool:
        """Check if a document has already been processed and saved."""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_name)
                        )
                    ]
                )
            )
            return result.count > 0
        except Exception as e:
            print(f"Error checking document existence: {e}")
            return False

    def delete_document(self, source_name: str) -> bool:
        """Delete all vectors associated with a specific document source."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_name)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def get_all_documents(self) -> List[str]:
        """Fetch all unique source document names from Qdrant."""
        try:
            sources = set()
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=["source"],
                with_vectors=False
            )
            for point in points:
                if point.payload:
                    source = point.payload.get("source")
                    if source and source != "unknown":
                        sources.add(source)
            return list(sources)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
