
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


   ### Example

         from langchain_text_splitters import HTMLHeaderTextSplitter
         
         headers_to_split_on = [
             ("h1", "Header 1"),
             ("h2", "Header 2"),
             ("h3", "Header 3"),
         ]
         
         splitter = HTMLHeaderTextSplitter(
             headers_to_split_on=headers_to_split_on
         )
         
         documents = splitter.split_text(html_string)


   ### Where it fits in a RAG pipeline
      
         HTML Web Page
               │
               ▼
         HTMLHeaderTextSplitter
               │
               ▼
         Sections based on <h1>, <h2>, <h3>
               │
               ▼
         Document Chunks + Metadata
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
         LLM

### Summary

         Purpose: Split HTML documents by heading tags rather than by character count.
         Key benefit: Keeps each logical section together and stores heading information as metadata.
         Ideal for: HTML documentation, blogs, technical manuals, and web pages.
         Advantage in RAG: The metadata (for example, the page title and section name) can improve retrieval quality and provide better 
                           context when answering questions.


#### 2.1.4 Recursive Json Splitter

   RecursiveJsonSplitter is a LangChain splitter designed specifically for JSON data.
   
   Unlike text splitters that work on characters or paragraphs, it recursively traverses a JSON object and breaks it into smaller JSON chunks while preserving 
   the JSON structure.
   
   ### Why do we need it?

      Suppose you have a large JSON file:

               {
                 "company": "ABC Ltd",
                 "employees": [
                   {
                     "id": 101,
                     "name": "John",
                     "department": "IT",
                     "salary": 80000
                   },
                   {
                     "id": 102,
                     "name": "Alice",
                     "department": "HR",
                     "salary": 70000
                   }
                 ],
                 "projects": [
                   {
                     "project_id": 1,
                     "name": "AI Chatbot"
                   },
                   {
                     "project_id": 2,
                     "name": "Recommendation System"
                   }
                 ]
               }

   If you use CharacterTextSplitter, it may split the JSON like this:
            
            {
              "company": "ABC Ltd",
              "employees": [
                {
                  "id": 101,

   This creates invalid JSON because braces and objects are cut in half.
   
   RecursiveJsonSplitter avoids this by keeping each chunk as valid JSON.

   ### How it works

   It recursively explores the JSON tree.

            JSON
            │
            ├── company
            │
            ├── employees
            │      ├── Employee 1
            │      └── Employee 2
            │
            └── projects
                   ├── Project 1
                   └── Project 2
            
   If the JSON is too large, it splits at a lower level while preserving the hierarchy.

## Example

         from langchain_text_splitters import RecursiveJsonSplitter
         
         json_data = {
             "company": "ABC Ltd",
             "employees": [
                 {"id": 1, "name": "John"},
                 {"id": 2, "name": "Alice"},
                 {"id": 3, "name": "Bob"}
             ]
         }
         
         splitter = RecursiveJsonSplitter(max_chunk_size=100)
         
         chunks = splitter.split_json(json_data)
         
         print(chunks)

   Possible output:

         [
             {
                 "company": "ABC Ltd"
             },
             {
                 "employees": [
                     {"id": 1, "name": "John"},
                     {"id": 2, "name": "Alice"}
                 ]
             },
             {
                 "employees": [
                     {"id": 3, "name": "Bob"}
                 ]
             }
         ]

Each chunk is still valid JSON.

### Important Parameter

   max_chunk_size

            splitter = RecursiveJsonSplitter(
                max_chunk_size=500
            )

   This specifies the approximate maximum size of each JSON chunk.
   
   If a chunk exceeds this size, the splitter recursively breaks it into smaller JSON objects or arrays.

### Why is it called "Recursive"?

   Consider this JSON:

         Company
         │
         ├── Employees
         │      ├── Employee 1
         │      ├── Employee 2
         │      └── Employee 3
         │
         └── Projects
                ├── Project 1
                └── Project 2
       

   If the entire JSON is too large:

      It first tries to keep the whole object together.
      If that's too big, it splits at the top-level keys (employees, projects).
      If employees is still too large, it splits the employee list.
      If an employee object is still too large, it can split deeper into nested objects.
      
   It keeps descending the JSON tree until each chunk fits the size limit.

---

## 3. Embedding
   (Text to Vectors)

                 
---   

## 4. VectorStore DB (FAISS, CHROMA DB, ASTRA DB)

We can query from vector database. Output will be context information.

## 
 
<img width="732" height="306" alt="image" src="https://github.com/user-attachments/assets/7eb91b95-ee25-4843-b2e1-02adde994ea4" />

