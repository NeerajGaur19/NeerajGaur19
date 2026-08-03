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

