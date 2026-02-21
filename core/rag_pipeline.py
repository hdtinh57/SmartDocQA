from typing import List, Dict, Any
from core.ocr_service import MistralOCRService, QwenVLService
from core.embed_service import EmbedService
from core.vdb_service import VDBService
from core.document_parser import DocumentParser
from core.llm_service import LLMService

class RagPipeline:
    def __init__(self, use_local_vlm=False):
        # Initialize Core Services
        try:
            if use_local_vlm:
                self.ocr = QwenVLService()
            else:
                self.ocr = MistralOCRService()
        except Exception as e:
            print(f"Error initializing OCR: {e}. Falling back to Mistral API if possible.")
            self.ocr = MistralOCRService()

        self.embedder = EmbedService(model_name="BAAI/bge-m3")
        self.vdb = VDBService(collection_name="smart_doc_qa")
        self.parser = DocumentParser(chunk_size=1000, chunk_overlap=200)
        self.llm_service = LLMService(provider="gemini")
        
    def ingest_document(self, file_path: str, source_name: str) -> bool:
        """
        End-to-end ingestion pipeline:
        1. OCR Image/PDF -> Text
        2. Chunk Text
        3. Embed Chunks
        4. Store in VectorDB
        """
        print(f"--- Starting Ingestion for {source_name} ---")
        
        # 1. OCR Extraction
        text = self.ocr.extract_text(file_path)
        if not text:
            print("Failed to extract text from document.")
            return False
            
        print(f"OCR extracted {len(text)} characters.")
        
        # Save OCR text locally for later viewing
        import os
        os.makedirs("data/ocr_results", exist_ok=True)
        ocr_path = f"data/ocr_results/{source_name}.txt"
        with open(ocr_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # 2. Chunking
        chunks, metadatas = self.parser.parse_and_chunk(text, source_metadata={"source": source_name})
        if not chunks:
            print("No text chunks generated.")
            return False
            
        print(f"Created {len(chunks)} chunks.")
        
        # 3. Embedding
        dense_vectors = self.embedder.embed_text(chunks)
        
        # 4. Storage
        self.vdb.upsert_chunks(chunks, dense_vectors, metadatas)
        
        print("--- Ingestion Complete ---")
        return True
        
    def ask(self, query: str) -> str:
        """
        End-to-end QA Pipeline:
        1. Embed user query
        2. Search VectorDB for context
        3. Build prompt
        4. Generate LLM Answer
        """
        # 1. Embed query (using 1 element list since service expects list)
        query_vector = self.embedder.embed_text([query])[0]
        
        # 2. Retrieve context from Qdrant
        search_results = self.vdb.search(query_vector, limit=4)
        
        if not search_results:
            return "Xin lỗi, tôi không tìm thấy thông tin nào phù hợp trong tài liệu."
            
        # Combine context
        context_parts = []
        for i, res in enumerate(search_results):
            text = res.get("text", "")
            source = res.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"--- Đoạn {i+1} (Nguồn: {source}) ---\n{text}")
            
        context_str = "\n\n".join(context_parts)
        
        # 3. Build prompt template
        system_prompt = (
            "Bạn là một trợ lý AI phân tích tài liệu thông minh. "
            "Dựa trên các đoạn ngữ cảnh (context) được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng. "
            "Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là bạn không biết dựa trên tài liệu, đừng tự bịa thêm.\n\n"
            "NGỮ CẢNH TÀI LIỆU:\n"
            f"{context_str}"
        )
        
        # 4. Generate answer
        answer = self.llm_service.generate_response(system_prompt=system_prompt, user_query=query)
        
        return answer
