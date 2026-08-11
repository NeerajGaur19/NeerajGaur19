import os
import gradio as gr
from dotenv import load_dotenv
#from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =========================
# Load API Key
# =========================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# =========================
# LLM and Embeddings
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_document"
)

#embeddings = HuggingFaceEmbeddings(
#    model_name="sentence-transformers/all-MiniLM-L6-v2"
#)

rag_chain = None

# =========================
# Helper Function
# =========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =========================
# Load and Process Document
# =========================
def load_document(pdf_files=None):
    global rag_chain

    documents = []
    file_names = []

    # If no file uploaded, use default ISTQB PDF
    if not pdf_files:
        pdf_paths = ["ISTQB-CTAL.pdf"]       
    else:
        pdf_paths = [f.name for f in pdf_files]


    # Load PDF
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
        file_names.append(os.path.basename(pdf_path))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create prompt
    
    prompt = ChatPromptTemplate.from_template(""" You are a helpful AI assistant.

    Answer the question using ONLY the provided context.
    If the answer is not present in the context, say:
    "The document does not contain enough information to answer this question."
    
    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    # LCEL RAG Chain (LangChain 1.x style)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return (
        f"Loaded {len(file_names)} PDF(s): {', '.join(file_names)}\n"
        f"Loaded {len(chunks)} chunks."
    )

# =========================
# Ask Question
# =========================
def ask_question(question):
    global rag_chain

    if rag_chain is None:
        return "Please upload a PDF document first."

    response = rag_chain.invoke(question)
    return response

# =========================
# Gradio UI
# =========================
with gr.Blocks(title="RAG Document Assistant") as demo:

    gr.Markdown("""
    # 📄 RAG-Based Document Assistant

    Upload a PDF and ask questions about its contents.

    **Tech Stack:** LangChain • Gemini • ChromaDB • Gradio
    """)

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"],file_count="multiple")
        status_output = gr.Textbox(label="Status")

    load_btn = gr.Button("📥 Load Document")

    question_input = gr.Textbox(
        label="Ask a Question",
        placeholder="Example: What is risk-based testing?",
        lines=3
    )

    answer_output = gr.Textbox(
        label="Answer",
        lines=12
    )

    ask_btn = gr.Button("🤖 Get Answer")

    load_btn.click(
        fn=load_document,
        inputs=pdf_input,
        outputs=status_output
    )

    # Click button
    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

    # Press Enter
    question_input.submit(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )

