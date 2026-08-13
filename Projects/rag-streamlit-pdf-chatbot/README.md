# 📄 Streamlit RAG PDF Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** built using **LangChain 1.x**, **Google Gemini**, **Hugging Face embeddings**, **ChromaDB**, and **Streamlit**.

The application allows users to upload a PDF document (such as ISTQB study guides, test management books, telecom process documents, or any knowledge-base PDF) and ask **context-aware questions** through a simple and interactive Streamlit web interface.

---

## 🚀 Features

* 📄 PDF document upload
* ✂️ Recursive text chunking using LangChain
* 🤗 Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
* 🗂️ ChromaDB vector database for semantic retrieval
* 🤖 Google Gemini (`gemini-flash-lite-latest`) for answer generation
* 🌐 Streamlit web interface
* 🔍 Context-aware question answering from uploaded PDFs
* 🔐 Secure API key management using `.env`
* ⚡ Local vector persistence with `chroma_db/`

---

## 🛠️ Tech Stack

| Technology                             | Purpose                         |
| -------------------------------------- | ------------------------------- |
| **Python**                             | Application development         |
| **Streamlit**                          | Interactive web UI              |
| **LangChain 1.x**                      | RAG orchestration               |
| **Google Gemini**                      | Large Language Model (LLM)      |
| **Hugging Face Sentence Transformers** | Local text embeddings           |
| **ChromaDB**                           | Vector storage and retrieval    |
| **PyPDF**                              | PDF document loading            |
| **python-dotenv**                      | Environment variable management |

---

## 📁 Project Structure

```text
rag-streamlit-pdf-chatbot/
│
├── streamlit_app.py         # Main Streamlit application
├── rag_pipeline.py          # RAG pipeline implementation
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── .env                     # API key (not committed to GitHub)
├── .gitignore               # Git ignore rules
└── venv/                    # Virtual environment
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-streamlit-pdf-chatbot.git
cd rag-streamlit-pdf-chatbot
```

---

### 2. Create a Virtual Environment

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

#### Mac / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 🔐 Configure Gemini API Key

Create a **`.env`** file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

⚠️ **Never commit `.env` to GitHub.**

---

## ▶️ Run the Application

```bash
streamlit run streamlit_app.py
```

You should see output similar to:

```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser.

---

## 🌐 Streamlit Cloud Deployment

### 1. Push the Project to GitHub

```bash
git init
git add .
git commit -m "Initial Streamlit RAG app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rag-streamlit-pdf-chatbot.git
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud

1. Go to **https://streamlit.io/cloud**
2. Connect your GitHub account
3. Select the repository
4. Set the main file to:

```text
streamlit_app.py
```

### 3. Add Secrets

In **App Settings → Secrets**, add:

```toml
GOOGLE_API_KEY = "your_gemini_api_key_here"
```

---

## 📚 Using the Chatbot

### Upload a PDF

Use the **Choose a PDF file** button to upload any PDF document.

### Example Questions

* What is risk-based testing?
* Explain verification vs validation.
* What are the responsibilities of a Test Manager?
* Describe the ISTQB test process activities.
* What is equivalence partitioning?
* Summarize the uploaded document.

---

## 🧠 How It Works

```text
PDF Upload
     ↓
PyPDFLoader
     ↓
RecursiveCharacterTextSplitter
     ↓
HuggingFaceEmbeddings
     ↓
ChromaDB Vector Store
     ↓
Retriever
     ↓
Gemini Prompt Construction
     ↓
Gemini 2.5 Flash
     ↓
Streamlit UI
```

The application retrieves the **most relevant chunks** from the uploaded PDF and sends them along with the user’s question to Gemini, ensuring **grounded and context-aware responses**.

---

## 🖼️ Application Screenshot

Add a screenshot after running the application.

Example:

What are the steps of a retrospective
<img width="1907" height="969" alt="image" src="https://github.com/user-attachments/assets/cc620eec-a9c9-4e3a-a9e9-6f3e20f9a773" />

What are the success factors for reviews
<img width="1911" height="958" alt="image" src="https://github.com/user-attachments/assets/027153b9-6e54-429a-a023-a7e681e5630a" />


---

## 📦 Requirements

### `requirements.txt`

```text
streamlit
langchain
langchain-community
langchain-huggingface
sentence-transformers==5.7.0
transformers==4.55.4
chromadb
pypdf
python-dotenv
langchain-google-genai
torchvision
```

---

## 🔧 Key Implementation Highlights

* Uses **LangChain 1.x** modular architecture
* Supports **semantic retrieval** with ChromaDB
* Uses **local Hugging Face embeddings** (no embedding API cost)
* Retrieves the **top-k relevant chunks** for each question
* Combines retrieved context into a **grounded Gemini prompt**
* Returns **document-based answers** instead of relying only on the LLM’s internal knowledge
* Includes **API key validation** and **Streamlit-friendly error handling**

---

## 🔍 Example Retrieval Flow

```python
docs = rag["retriever"].invoke(question)

context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""

response = rag["llm"].invoke(prompt)
```

---

## 🙏 Acknowledgement

This project was developed as a **hands-on learning exercise** to understand:

* Retrieval-Augmented Generation (RAG)
* Embeddings and vector databases
* Semantic search
* LangChain 1.x architecture
* Google Gemini integration
* Streamlit-based AI application deployment
* Local vector persistence with ChromaDB

---

## 📄 License

This project is intended for **learning and portfolio purposes**. Ensure that any uploaded PDF documents comply with their respective copyright and usage terms.
 
---

## 👨‍💻 Author

**Neeraj Gaur**

* 💼 LinkedIn: https://www.linkedin.com/in/neerajgaur82
* 💻 GitHub: https://github.com/NeerajGaur19

---

## ⭐ Support

If you found this project useful, consider **starring the repository** on GitHub! ⭐

It helps others discover the project and supports continued improvements.

