from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Function to build complete RAG pipeline
def create_rag_chain(pdf_path, google_api_key):

    # -------------------------------
    # STEP 1: LOAD PDF
    # -------------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # -------------------------------
    # STEP 2: SPLIT DOCUMENT
    # -------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(documents)

    # -------------------------------
    # STEP 3: CREATE EMBEDDINGS
    # -------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------------
    # STEP 4: CREATE VECTOR DATABASE
    # -------------------------------
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    # -------------------------------
    # STEP 5: CREATE RETRIEVER
    # -------------------------------
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # -------------------------------
    # STEP 6: LOAD LLM
    # -------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0,
        google_api_key=google_api_key
    )

     # Return both retriever and llm
    return {
        "retriever": retriever,
        "llm": llm
    }