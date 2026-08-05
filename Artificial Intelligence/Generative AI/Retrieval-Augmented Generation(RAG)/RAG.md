# Retrieval-Augmented Generation

RAG (Retrieval-Augmented Generation) is a Generative AI technique that improves an LLM's answers by retrieving relevant information from an external knowledge source before generating the response.

Without RAG, an LLM only relies on what it learned during training.

With RAG, it can also use:

    PDFs
    Word documents
    Databases
    Websites
    Company policies
    Research papers
    Emails
    SharePoint, Confluence, etc.

# Why do we need RAG?

Suppose ChatGPT was trained in 2025.

You ask:

What is my company's leave policy?

The model doesn't know because it has never seen your private HR document.

Instead of guessing, RAG searches your documents first.


# RAG Architecture

                 User Question
                        │
                        ▼
                Query Processing
                        │
                        ▼
                Embedding Model
                        │
                        ▼
                Vector Database
                        │
               Similarity Search
                        │
                 Top K Chunks
                        │
                        ▼
               Prompt Construction
                        │
                        ▼
                       LLM
                        │
                        ▼
                   Final Answer

---

# Components of RAG

There are 8 main components.

## 1. Knowledge Source

This is where your data lives.

Examples:

        PDF
        Word
        SQL Database
        CSV
        Website
        SharePoint
        Emails
        API

Example:

        Employee Handbook.pdf


## 2. Document Loader

The loader reads documents.

Example:

    PDF
    ↓
    Text

In LangChain:

    PyPDFLoader()
    TextLoader()
    CSVLoader()

Without loading, the LLM cannot read the document.


## 3. Text Chunking (Text Splitter)

LLMs cannot efficiently process a very large document in one go.

Example:

A PDF has

    500 pages

Split into

    Chunk 1
    Chunk 2
    Chunk 3
    ...
    Chunk 500

Typical chunk size:

    500–1000 characters

Sometimes overlapping chunks are used to preserve context across boundaries.

### Why do we chunk?

Suppose the sentence is

    The leave policy states employees receive
    12 casual leaves annually.

If you split badly:

Chunk 1

    The leave policy states employees receive

Chunk 2

    12 casual leaves annually.

Neither chunk has the complete meaning.

Using chunk overlap helps avoid losing important context.


## 4. Embedding Model

The chunk is converted into numbers.

Example

    Chunk
    
    ↓
    
    [0.25, -0.11, 0.72, ...]

This numerical representation is called an embedding.

Popular embedding models:

    OpenAI Embeddings
    Hugging Face sentence-transformers
    BAAI/bge models
    E5 models


### Why embeddings?

Computers cannot compare paragraphs directly.

Instead they compare vectors.

Example

    "What is leave policy?"
    
    ↓
    
    Vector
    
    ↓
    
    Compare with document vectors


## 5. Vector Database

Stores embeddings.

Instead of

    Document

It stores

    Embedding

Popular vector databases:

* FAISS
* Chroma
* Pinecone
* Weaviate
* Milvus
* Qdrant

Think of it as a specialized database optimized for similarity search.


## 6. Retriever (Similarity Search)

When the user asks a question

    "What is maternity leave?"

The question is also converted into an embedding.

Then

    Question Vector
    
    ↓
    
    Compare
    
    ↓
    
    Document Vectors
    
    ↓
    
    Top 5 most similar chunks

This is called vector similarity search.

## 7. Prompt Augmentation

Now LangChain (or your application) creates a prompt like:

    Context:
    
    Employees are entitled to
    12 casual leaves annually.
    
    Question:
    
    How many casual leaves are allowed?
    
    Answer:

Notice the retrieved context is inserted into the prompt.

This is the Augmented part of Retrieval-Augmented Generation.

## 8. Large Language Model (LLM)

Finally the LLM reads

* User Question
* Retrieved Context

and generates

    Employees receive 12 casual leaves annually according to the company handbook.

---

# Complete Flow
        
                        PDF
                         │
                         ▼
                  Document Loader
                         │
                         ▼
                  Text Chunking
                         │
                         ▼
                  Embedding Model
                         │
                         ▼
                 Vector Database
        ──────────────────────────────────────
                  User Question
                         │
                         ▼
              Question Embedding
                         │
                         ▼
                Similarity Search
                         │
                         ▼
                Top Relevant Chunks
                         │
                         ▼
              Prompt Augmentation
                         │
                         ▼
                       LLM
                         │
                         ▼
                  Final Response


## Definition of each component

        Component	                            Purpose
    Knowledge Source	            Stores original data (PDFs, databases, websites, etc.)
    Document Loader	                Reads data into the application
    Chunking	                    Splits large documents into manageable pieces
    Embedding Model	                Converts text into numerical vectors
    Vector Database	                Stores and indexes embedding vectors
    Retriever	                    Finds the most relevant chunks using similarity search
    Prompt Augmentation	            Combines retrieved context with the user's question
    LLM	                            Generates the final natural-language answer



<img width="1536" height="1024" alt="ChatGPT Image Aug 5, 2026, 10_57_27 AM" src="https://github.com/user-attachments/assets/30d2d79a-69ca-41ce-b97e-cf89cded164b" />


