import os
from qdrant_client import QdrantClient
from core.config import settings

print(f"Connecting to {settings.qdrant_url}")
try:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )
    cols = client.get_collections()
    print("Success:", cols)
except Exception as e:
    print("Error:", e)
