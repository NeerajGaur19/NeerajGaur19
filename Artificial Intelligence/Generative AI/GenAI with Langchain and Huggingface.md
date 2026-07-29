
## 1. Data Source -> 
   
   ### 1.1 Data Ingestion with document Loaders


---

## 2. Data Transformation -> 

### 2.1 Text Splitting Technique    

#### 2.1.1 Recursive Character Text Splitter 
   (Data into text chunks)
   
   What is RecursiveCharacterTextSplitter?

      RecursiveCharacterTextSplitter splits a large document into smaller chunks while trying to preserve the document's natural structure.
      Instead of cutting text at an exact character count, it recursively tries different separators until the chunk size is reached.

Sample Program

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

#### 2.1.2 Character Text Splitter





#### 2.1.3 HTML Header Text Splitter


#### 2.1.4 Recursive Json Splitter


---

## 3. Embedding
   (Text to Vectors)

                 
---   

## 4. VectorStore DB (FAISS, CHROMA DB, ASTRA DB)

We can query from vector database. Output will be context information.

## 
 
<img width="732" height="306" alt="image" src="https://github.com/user-attachments/assets/7eb91b95-ee25-4843-b2e1-02adde994ea4" />

