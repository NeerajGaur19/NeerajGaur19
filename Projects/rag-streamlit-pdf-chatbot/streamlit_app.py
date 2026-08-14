import streamlit as st
from rag_pipeline import create_rag_chain
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Browser tab title
st.set_page_config(page_title="PDF RAG Chatbot")

# Main heading
st.title("📄 PDF RAG Chatbot (HuggingFace Embeddings)")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

st.write("Upload a PDF and ask questions from it.")

# Upload PDF
uploaded_files = st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)

if uploaded_files:
    temp_pdf_paths = []
    try:
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_paths.append(tmp_file.name)

        # Build RAG pipeline
        with st.spinner("Processing PDFs"):
            rag = create_rag_chain(temp_pdf_paths,GOOGLE_API_KEY)

        st.success(f"Processed {len(uploaded_files)} PDF files")

        # User question
        question = st.text_input("Ask a question from the PDF")

        if question:
            with st.spinner("Searching document and generating answer..."):
                # Retrieve relevant chunks
                docs = rag["retriever"].invoke(question)

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
                response = rag["llm"].invoke(prompt)

            st.subheader("Answer")
            st.write(response.content[0]["text"])
            
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")