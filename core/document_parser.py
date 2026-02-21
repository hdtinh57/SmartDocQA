from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def parse_and_chunk(self, raw_text: str, source_metadata: Dict[str, Any] = None) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Takes raw text extracted from OCR, splits it into chunks,
        and attaches relevant metadata to each chunk.
        """
        if not raw_text or not raw_text.strip():
            print("Warning: Received empty text for chunking.")
            return [], []
            
        chunks = self.text_splitter.split_text(raw_text)
        
        # Prepare metadata
        if source_metadata is None:
            source_metadata = {"source": "unknown"}
            
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = source_metadata.copy()
            meta["chunk_index"] = i
            metadatas.append(meta)
            
        return chunks, metadatas
