import streamlit as st
import os
import rag_backend

# Set up page configurations
st.set_page_config(page_title="Academic RAG Explorer", page_icon="📚", layout="wide")

st.title("Multi-PDF Research Paper RAG System")
st.write("Query architectural details from papers like Attention Is All You Need, LoRA, BERT, and more.")

# Sidebar for Setup & Document management
st.sidebar.header("📁 System Controls")

if st.sidebar.button("🔨 Build/Refresh Vector Database"):
    with st.spinner("Processing PDFs and generating embeddings..."):
        db = rag_backend.create_vector_db()
        if db is not None:
            st.sidebar.success("Database created and saved successfully!")
        else:
            st.sidebar.error("Failed to build DB. Check if your 'papers' directory contains PDFs.")

# Show currently loaded documents in sidebar
if os.path.exists(rag_backend.PDF_FOLDER):
    files = [f for f in os.listdir(rag_backend.PDF_FOLDER) if f.endswith('.pdf')]
    st.sidebar.subheader("Loaded Papers:")
    if files:
        for f in files:
            st.sidebar.text(f"📄 {f}")
    else:
        st.sidebar.warning("No PDFs found in 'papers/' folder.")
else:
    st.sidebar.info("Click 'Build Database' to initialize.")

# Main Interface Query Field
st.subheader("🤖 Ask your Research Questions")
query = st.text_input(
    "Enter a comparison or concept query:", 
    placeholder="e.g., How does LoRA reduce training cost?"
)

if query:
    if not os.path.exists(rag_backend.DB_FOLDER):
        st.error("Vector database not found. Please click 'Build/Refresh Vector Database' in the sidebar first.")
    else:
        with st.spinner("Searching documents and generating response..."):
            try:
                qa_chain = rag_backend.build_qa_chain()
                
                # We use "input" now instead of "query"
                response = qa_chain.invoke({"input": query})
                
                st.markdown("### 📝 Answer")
                # We use "answer" now instead of "result"
                st.info(response["answer"])
                
                st.markdown("### 🔍 Source Citations")
                # We look up "context" now instead of "source_documents"
                sources = response.get("context", [])
                if sources:
                    unique_sources = set([doc.metadata.get("source_paper", "Unknown") for doc in sources])
                    for source in unique_sources:
                        st.markdown(f"- **File:** `{source}`")
                else:
                    st.write("No source documents returned.")
            except Exception as e:
                st.error(f"An error occurred: {e}")