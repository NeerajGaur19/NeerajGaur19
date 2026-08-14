import streamlit as st
from rag_pipeline import create_rag_chain
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY and "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

import streamlit as st

# Hide the Deploy button, GitHub icon, and main menu
st.markdown(
    """
    <style>
    .stDeployButton { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

# Browser tab title
st.set_page_config(page_title="PDF RAG Chatbot")

# Main heading
st.subheader("📄 PDF RAG Chatbot (HuggingFace Embeddings)")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()


col1, col2, col3 = st.columns(3)

with col1:
    st.info("📚 **Multi-PDF Support**\nUpload and search across multiple documents.")

with col2:
    st.success("🔍 **Semantic Search**\nFind answers even when wording is different.")

with col3:
    st.warning("⚡ **Powered by Gemini**\nContext-aware answers using RAG + LLMs.")


#st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:18px;'>Upload a PDF and ask questions from it</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")
    st.caption("Supported: Multiple PDF documents") 

# Upload PDF
# uploaded_files = st.file_uploader("Choose a PDF file",
#    type="pdf",
#    accept_multiple_files=True
#)

if uploaded_files:
    temp_pdf_paths = []
    try:
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_paths.append(tmp_file.name)

        # Build RAG pipeline
        if temp_pdf_paths and "rag" not in st.session_state:
            with st.spinner("📄 Processing PDFs and building vector index..."):
                st.session_state.rag = create_rag_chain(temp_pdf_paths,GOOGLE_API_KEY)

        st.success(f"Processed {len(uploaded_files)} PDF files")

        # User question
        question = st.text_input("Ask a question from the PDF.")

        if question:
            with st.spinner("Searching document and generating answer..."):
                # Retrieve relevant chunks
                docs = st.session_state.rag["retriever"].invoke(question)

                # Combine retrieved text
                context = "\n\n".join([doc.page_content for doc in docs])

                # Create prompt
                prompt = f"""
                Answer the question using only the provided context.

                Context:
                {context}

                Question:
                {question}
                """
                # Generate answer
                response = st.session_state.rag["llm"].invoke(prompt)

            #st.subheader("Source Details")            
            #st.write(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")
            st.subheader("Answer")
            st.write(response.content[0]["text"])
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
