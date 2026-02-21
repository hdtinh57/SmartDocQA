import os
import base64
from typing import Optional
from mistralai import Mistral
from core.config import settings

class MistralOCRService:
    def __init__(self):
        self.api_key = settings.mistral_api_key
        # Ensure we have the API key
        if not self.api_key:
            print("Warning: MISTRAL_API_KEY is not set.")
        self.client = Mistral(api_key=self.api_key) if self.api_key else None
        self.model = "mistral-ocr-latest"

    def _encode_image(self, document_path: str) -> tuple[str, str]:
        with open(document_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode('utf-8')
            # Extract extension
            ext = os.path.splitext(document_path)[1][1:].lower()
            mime_type = "application/pdf" if ext == "pdf" else f"image/{ext}"
            if ext == 'jpg':
                mime_type = 'image/jpeg'
            return f"data:{mime_type};base64,{encoded_string}", ext

    def extract_text(self, image_path: str) -> Optional[str]:
        if not self.client:
            raise ValueError("Mistral Client not initialized. Check API key.")
        
        try:
            base64_data, ext = self._encode_image(image_path)
            
            document_type = "document_url" if ext == "pdf" else "image_url"
            document_key = "document_url" if ext == "pdf" else "image_url"
            
            # Using the Mistral OCR endpoint
            response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": document_type,
                    document_key: base64_data
                }
            )
            
            # Parse the text from response
            # Assuming the response object has pages as returned by mistralai
            extracted_text = ""
            for page in response.pages:
                extracted_text += page.markdown + "\n\n"
                
            return extracted_text.strip()
        except Exception as e:
            print(f"Mistral OCR Error: {e}")
            return None


class QwenVLService:
    def __init__(self):
        """
        Khởi tạo Qwen3-VL chạy local. 
        Requires: transformers >= 4.40.0, qwen-vl-utils, flash_attn (optional)
        """
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            # Use a smaller version for 8GB VRAM (2B model) or INT4 quantization
            self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
            
            print(f"Loading {self.model_id}...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            print("Qwen-VL Model loaded securely.")
            self.is_ready = True
        except ImportError:
            print("Warning: Missing libraries to run Qwen-VL. Run check requirements.")
            self.is_ready = False
            
    def extract_text(self, image_path: str, prompt: str = "Extract all text, tables, and contents from this image in markdown format.") -> Optional[str]:
        if not self.is_ready:
            raise RuntimeError("Qwen-VL model is not loaded.")
            
        try:
            from PIL import Image
            from qwen_vl_utils import process_vision_info
            import torch
            
            image = Image.open(image_path)
            
            # Construct message according to Qwen VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            inputs = inputs.to(self.device)
            
            # Generation bounds
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            print(f"Qwen-VL OCR Error: {e}")
            return None

# Factory logic simply exports standard interface
# Usage:
# ocr = MistralOCRService()
# text = ocr.extract_text("path/to/img.png")
