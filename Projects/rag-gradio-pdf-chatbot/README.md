RAG Gradio PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain 1.x, all-MiniLM-L6-v2 HuggingFace, ChromaDB, and Gradio. The application allows users to upload single or multiple PDF documents (such as ISTQB study guides, test management books, or process documents) and ask context-aware questions through a simple web interface.

🚀 Features
📚 Multi-PDF document ingestion
✂️ Recursive text chunking
🔍 Gemini embeddings (gemini-embedding-001)
🗂️ ChromaDB vector database
🤖 Context-aware question answering
🌐 Gradio web interface
🧠 Semantic retrieval across multiple documents
🔐 Secure API key management using .env

---

🛠️ Tech Stack

    Technology	            Purpose
    Python	                Application development
    LangChain 1.x	        RAG orchestration
    Google Gemini	        LLM
    all-MiniLM-L6-v2        Embedding
    ChromaDB	            Vector storage and retrieval
    Gradio	                Web-based UI
    PyPDF	                PDF document loading

---

📁 Project Structure

    RAG-GRADIO-PDF-CHATBOT/
    │
    ├── app.py                  # Main Gradio application
    ├── requirements.txt        # Python dependencies
    ├── README.md               # Project documentation
    ├── .env                    # API key (not committed to GitHub)
    ├── .gitignore              # Git ignore rules
    └── venv/                   # Virtual environment

---

⚙️ Installation

1. Clone the repository
    git clone https://github.com/YOUR_USERNAME/RAG-GRADIO-PDF-CHATBOT.git

    cd rag-gradio-pdf-chatbot

3. Create a virtual environment

    Windows
        python -m venv venv
        venv\Scripts\activate
    Mac/Linux
        python -m venv venv
        source venv/bin/activate

4. Install dependencies
    python -m pip install -r requirements.txt

---

🔐 Configure Gemini API Key

    Create a .env file in the project root:

    GOOGLE_API_KEY=your_gemini_api_key_here

    ⚠️ Never commit .env to GitHub.

---

▶️ Run the Application

    python app.py

    You will see:

    * Running on local URL:  http://127.0.0.1:7860

    Open the URL in your browser.

---

🌐 Temporary Public URL

    For sharing a public demo, update the launch section:

    if __name__ == "__main__":
        demo.launch(share=True)

    Gradio will generate a temporary public URL such as:

    https://abcd1234.gradio.live

---

📚 Using the Chatbot

Default Local PDF

    If no file is uploaded, the app automatically loads:

    ISTQB-CTAL.pdf

Upload Multiple PDFs

    Use the Upload PDF Files option to upload several documents together. The chatbot creates a combined knowledge base from all uploaded PDFs.

Example questions:

    * What are the review types
    * What are the success factors for reviews
    * What are the test management activities 
    * What are the steps of a retrospective

---

🧠 How It Works

    PDF Upload
        ↓
    PyPDFLoader
        ↓
    RecursiveCharacterTextSplitter
        ↓
    Gemini Embeddings
        ↓
    ChromaDB Vector Store
        ↓
    Retriever
        ↓
    LangChain LCEL RAG Chain
        ↓
    Gemini LLM
        ↓
    Gradio UI


🖼️ Application Screenshot

    Add a screenshot here after running the application.

    Example:

<img width="1716" height="914" alt="image" src="https://github.com/user-attachments/assets/803400de-43a1-45cc-8e19-1bf4cafb9ee2" />

---

📦 Requirements

    Current requirements.txt:

    streamlit
    langchain
    langchain-community
    langchain-huggingface
    langchain-google-genai
    langchain-text-splitters
    langchain-chroma
    sentence-transformers==5.1.0
    chromadb
    gradio
    pypdf
    python-dotenv
    
    --extra-index-url https://download.pytorch.org/whl/cpu
    torch==2.13.0+cpu

---

🔧 Key Implementation Highlights

    * Uses LangChain 1.x LCEL (Runnable) architecture
    * Supports multi-document retrieval
    * Uses ChromaDB for semantic search
    * Combines retrieved chunks into a single context prompt
    * Returns grounded answers instead of relying only on the LLM’s internal knowledge

---

🙏 Acknowledgement

    This project was developed as a hands-on learning exercise to understand:

    Retrieval-Augmented Generation (RAG)
    Embeddings and vector databases
    Semantic search
    LangChain 1.x LCEL architecture
    Lightweight AI application deployment with Gradio

---

📄 License

    This project is intended for learning and portfolio purposes. Ensure that any uploaded PDF documents comply with their respective copyright and usage terms.

---

👨‍💻 Author

    Neeraj Gaur

    LinkedIn: https://www.linkedin.com/in/neerajgaur82
    GitHub: https://github.com/NeerajGaur19

⭐ If you found this project useful, consider giving it a star on GitHub!
