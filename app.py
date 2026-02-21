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
        # Lấy danh sách tài liệu đã có sẵn từ Qdrant để hiến thị
        docs = st.session_state.pipeline.vdb.get_all_documents()
        st.session_state.processed_files = docs

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
            # Save uploaded file permanently to view later
            os.makedirs("data/uploaded_docs", exist_ok=True)
            save_path = os.path.join("data/uploaded_docs", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Kiểm tra xem file đã có trong DB chưa (trên giao diện cache hoặc quét Qdrant trực tiếp)
            if uploaded_file.name in st.session_state.processed_files or st.session_state.pipeline.vdb.has_document(uploaded_file.name):
                st.warning(f"Tài liệu '{uploaded_file.name}' đã có sẵn trong cơ sở dữ liệu! Bạn có thể đặt câu hỏi hoặc xem tài liệu ngay.")
                if uploaded_file.name not in st.session_state.processed_files:
                    st.session_state.processed_files.append(uploaded_file.name)
            else:
                with st.spinner(f"Đang xử lý {uploaded_file.name}... (OCR & VectorEmbedding)"):
                    # Run Ingestion Pipeline
                    success = st.session_state.pipeline.ingest_document(
                        file_path=save_path, 
                        source_name=uploaded_file.name
                    )
                    
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
        for file in list(st.session_state.processed_files):
            col1, col2 = st.columns([8, 2])
            with col1:
                st.markdown(f"📄 `{file}`")
            with col2:
                if st.button("❌", key=f"del_{file}", help="Xóa tài liệu"):
                    # Xóa vector từ Qdrant
                    st.session_state.pipeline.vdb.delete_document(file)
                    st.session_state.processed_files.remove(file)
                    # Xóa file vật lý
                    try:
                        os.remove(os.path.join("data", "uploaded_docs", file))
                        os.remove(os.path.join("data", "ocr_results", f"{file}.txt"))
                    except:
                        pass
                    st.rerun()
    else:
        st.markdown("*Chưa có tài liệu nào*")

# --- Main Layout ---
st.title("🤖 Trợ lý AI hỏi đáp tài liệu thông minh (RAG)")
st.markdown("Hãy upload tài liệu ở thanh bên trái (Sidebar) trước khi đặt câu hỏi để AI có ngữ cảnh.")

tab1, tab2 = st.tabs(["💬 Chat & Q&A", "📄 Xem tài liệu (Bản gốc & OCR)"])

with tab1:
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
                response = st.session_state.pipeline.ask(prompt, allowed_sources=st.session_state.processed_files)
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("Chi tiết tài liệu")
    if not st.session_state.processed_files:
        st.info("Chưa có tài liệu nào trong hệ thống. Hãy upload tài liệu ở cột trái.")
    else:
        selected_doc = st.selectbox("Chọn tài liệu để xem:", st.session_state.processed_files)
        if selected_doc:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Bản gốc")
                doc_path = os.path.join("data", "uploaded_docs", selected_doc)
                if os.path.exists(doc_path):
                    ext = os.path.splitext(doc_path)[1].lower()
                    if ext == ".pdf":
                        # Display PDF using base64 embed
                        import base64
                        with open(doc_path, "rb") as f:
                            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf" />'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    elif ext in [".png", ".jpg", ".jpeg"]:
                        st.image(doc_path, use_container_width=True)
                else:
                    st.warning("Không tìm thấy file gốc trên máy chủ (Có thể đã bị xóa hoặc được xử lý từ thiết bị khác). Bạn có thể thử upload lại file đó để xem.")
            
            with col2:
                st.subheader("Kết quả OCR")
                ocr_path = os.path.join("data", "ocr_results", f"{selected_doc}.txt")
                if os.path.exists(ocr_path):
                    with open(ocr_path, "r", encoding="utf-8") as f:
                        ocr_text = f.read()
                    st.text_area("Văn bản nhận diện được (có thể chỉnh sửa để kiểm tra):", value=ocr_text, height=600)
                else:
                    st.warning("Không tìm thấy kết quả OCR lưu trữ cho tài liệu này (Có thể quá trình OCR lần trước bị lỗi hoặc file text đã bị xóa).")
