# Smart Document Q&A System

![Status](https://img.shields.io/badge/Status-Development-yellow)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-green)

A cutting-edge Retrieval-Augmented Generation (RAG) system for intelligent document querying. The system supports multi-modal document processing (PDFs, Images, and Text) using advanced Vision-Language Models (VLMs) and dense-sparse vector search.

Developed as part of an AI/ML engineering portfolio for 2025-2026.

## 🌟 Key Features

- **Multi-modal OCR Processing**: Utilizes **Qwen3-VL** (local) and **Mistral OCR API** to extract text, tables, and complex layouts with high precision.
- **Advanced Embeddings**: Uses **BGE-M3** for powerful multi-lingual and hybrid dense-sparse vector representations.
- **High-Performance Vector Database**: Integrates with **Qdrant** for low-latency similarity search.
- **RAG Pipeline**: orchestrated by **LangChain** and **LlamaIndex** to retrieve relevant context and generate accurate answers.
- **LLM Integration**: Supports both local LLMs (via Ollama/GGUF) and cloud APis (Gemini AI).
- **Interactive UI**: Built with **Streamlit** for a seamless user experience.

## 🛠️ Technology Stack

- **Backend & Pipeline**: Python 3.13, FastAPI, LangChain, LlamaIndex
- **Vision & OCR**: Qwen3-VL, Mistral OCR, HuggingFace Transformers
- **Database**: Qdrant (Vector DB)
- **Frontend**: Streamlit
- **Infrastructure**: Local CUDA 12.8 / RTX 3070 Optimization

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.13 installed along with an NVIDIA GPU supporting CUDA 12.8+.

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Project1_SmartDocQA
   ```

2. **Create and activate a virtual environment:**

   ```bash
   py -3.13 -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/MacOS
   source venv/bin/activate
   ```

3. **Install PyTorch (CUDA 12.8 support):**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Install remaining dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up Environment Variables:**
   - Create a `.env` file in the root directory.
   - Add your API keys:
     ```env
     MISTRAL_API_KEY="your_api_key_here"
     GEMINI_API_KEY="your_api_key_here"
     QDRANT_URL="http://localhost:6333"
     ```

### Verification

Run the hardware verification script to ensure PyTorch detects your GPU:

```bash
python scripts/verify_cuda.py
```

## 📄 License

This is a personal project developed for research and building an AI/ML portfolio.
