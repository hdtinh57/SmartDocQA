import streamlit as st
import os
import tempfile
from core.rag_pipeline import RagPipeline

# --- Page Config ---
st.set_page_config(
    page_title="Smart Document Q&A System",
    page_icon="🤖",
    layout="wide"
)

# --- Initialization ---
@st.cache_resource
def get_pipeline():
    # Cache the pipeline so models (like BGE-M3) are loaded only once
    return RagPipeline(use_local_vlm=False)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline()
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

initialize_session_state()

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("📂 Upload Documents")
    st.markdown("hỗ trợ định dạng PDF, PNG, JPG, JPEG.")
    
    uploaded_file = st.file_uploader("Upload file here", type=["pdf", "png", "jpg", "jpeg"])
    
    if st.button("Process Document", type="primary"):
        if uploaded_file is not None:
            if uploaded_file.name in st.session_state.processed_files:
                st.warning(f"File '{uploaded_file.name}' đã được xử lý trước đó!")
            else:
                with st.spinner(f"Đang xử lý {uploaded_file.name}... (OCR & VectorEmbedding)"):
                    # Save uploaded file to a temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    
                    # Run Ingestion Pipeline
                    success = st.session_state.pipeline.ingest_document(
                        file_path=tfile.name, 
                        source_name=uploaded_file.name
                    )
                    
                    # Clean up temp file
                    os.unlink(tfile.name)
                    
                    if success:
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.success(f"Xử lý thành công: {uploaded_file.name}")
                    else:
                        st.error(f"Xử lý thất bại: {uploaded_file.name}. Vui lòng kiểm tra API Key (Mistral) trong file .env")
        else:
            st.error("Vui lòng upload một file trước khi nhấn Process.")
            
    st.divider()
    st.markdown("### 📚 Tài liệu đã lưu")
    if st.session_state.processed_files:
        for file in st.session_state.processed_files:
            st.markdown(f"- 📄 `{file}`")
    else:
        st.markdown("*Chưa có tài liệu nào*")

# --- Main Layout ---
st.title("🤖 Trợ lý AI hỏi đáp tài liệu thông minh (RAG)")
st.markdown("Hãy upload tài liệu ở thanh bên trái (Sidebar) trước khi đặt câu hỏi để AI có ngữ cảnh.")

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Nhập câu hỏi của bạn về tài liệu..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm thông tin và suy nghĩ..."):
            response = st.session_state.pipeline.ask(prompt)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
