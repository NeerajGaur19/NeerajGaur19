
## 1. Data Source -> 
   
   ### 1.1 Data Ingestion with document Loaders


---

## 2. Data Transformation -> 

### 2.1 Text Splitting Technique    

#### 2.1.1 Recursive Character Text Splitter 
   (Data into text chunks)
   
   ### What is RecursiveCharacterTextSplitter?

      RecursiveCharacterTextSplitter splits a large document into smaller chunks while trying to preserve the document's natural structure.
      Instead of cutting text at an exact character count, it recursively tries different separators until the chunk size is reached.

   ### Sample Program

      from langchain_text_splitters import RecursiveCharacterTextSplitter
      
      text = """
      Artificial Intelligence is transforming industries.
      
      Machine Learning is a subset of AI.
      
      Deep Learning is a subset of Machine Learning.
      """
      
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=50,
          chunk_overlap=10
      )
      
      chunks = splitter.split_text(text)
      
      print(chunks)
   
   ### Where it fits in a RAG pipeline
   
            PDF / DOCX / TXT
                    │
                    ▼
            Document Loader
                    │
                    ▼
            RecursiveCharacterTextSplitter
                    │
                    ▼
            Small Chunks
                    │
                    ▼
            Embeddings
                    │
                    ▼
            Vector Database
                    │
                    ▼
            Retriever
                    │
                    ▼
            LLM (GPT, Llama, Gemini, etc.)

#### 2.1.2 Character Text Splitter

CharacterTextSplitter is the simplest text splitter in LangChain. It divides text into chunks based on a specified separator and chunk size, without trying to preserve the document's structure intelligently.

   ### What is CharacterTextSplitter?
   
   It splits text into chunks based on:
   
   * A separator (such as \n or a space)
   * A chunk_size
   * A chunk_overlap

   Unlike RecursiveCharacterTextSplitter, it does not try multiple separators.

   ### Example:

      from langchain_text_splitters import CharacterTextSplitter
      
      splitter = CharacterTextSplitter(
          separator="\n",
          chunk_size=40,
          chunk_overlap=10
      )
      
      chunks = splitter.split_text(text)
      
      print(chunks)

   ### Visual Comparison

      Suppose your text is:
      
      Paragraph 1
      
      Paragraph 2
      
      Paragraph 3

   ### CharacterTextSplitter

      Uses only one separator
              │
              ▼
      Paragraph 1
      
      Paragraph 2
      
      Paragraph 3

   If one paragraph is longer than the chunk_size, it doesn't intelligently try a different separator.

   ### Which one should you use?

      <img width="683" height="307" alt="image" src="https://github.com/user-attachments/assets/8b566bc7-0c34-4d1f-8807-9c2618a96b67" />

 
## Summary

      CharacterTextSplitter uses one separator (such as \n) and splits text into chunks based on chunk_size and chunk_overlap.
      It is simple and predictable, but it does not adapt if a chunk is still too large.
      RecursiveCharacterTextSplitter is generally preferred for production RAG systems because it preserves the document's structure more effectively by 
      trying multiple separators in order.


#### 2.1.3 HTML Header Text Splitter

   HTMLHeaderTextSplitter is a LangChain text splitter specifically designed for HTML documents. 
   Instead of splitting by characters, it splits the document based on HTML heading tags (<h1>, <h2>, <h3>, etc.).
   This is useful because headings naturally define sections of a document.

   ### Why do we need it?
   
   Consider this HTML page:
      
         <h1>Machine Learning</h1>
         
         <p>Machine Learning is a subset of AI.</p>
         
         <h2>Supervised Learning</h2>
         
         <p>Uses labeled data.</p>
         
         <h2>Unsupervised Learning</h2>
         
         <p>Uses unlabeled data.</p>
   
   If you use CharacterTextSplitter, it may split in the middle of a section.
   
   If you use HTMLHeaderTextSplitter, each section stays together.

#### 2.1.4 Recursive Json Splitter

---

## 3. Embedding
   (Text to Vectors)

                 
---   

## 4. VectorStore DB (FAISS, CHROMA DB, ASTRA DB)

We can query from vector database. Output will be context information.

## 
 
<img width="732" height="306" alt="image" src="https://github.com/user-attachments/assets/7eb91b95-ee25-4843-b2e1-02adde994ea4" />

