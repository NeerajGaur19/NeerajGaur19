# LangChain Ecosystem


                     ┌──────────────────────────┐
                     │      LLM Providers       │
                     │ OpenAI, Gemini, Claude   │
                     │ Llama, Mistral, etc.     │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      LangChain Core      │
                     │ Prompts                  │
                     │ Models                   │
                     │ Output Parsers           │
                     │ Runnables                │
                     │ Tools                    │
                     └─────────────┬────────────┘
                                   │
               ┌───────────────────┼────────────────────┐
               ▼                   ▼                    ▼
        Chains & LCEL         Memory             Document Loaders
                                                    │
                                                    ▼
                                             Text Splitters
                                                    │
                                                    ▼
                                               Embeddings
                                                    │
                                                    ▼
                                               Vector DB
                                      (FAISS, Chroma, Pinecone)
                                                    │
                                                    ▼
                                                Retriever
                                                    │
                                                    ▼
                                                   RAG
                                                    │
                                                    ▼
                                                  Agent
                                                    │
                              ┌─────────────────────┴───────────────────┐
                              ▼                                         ▼
                           Search API                             Python Tool
                           SQL Database                           Calculator
                           Weather API                            Custom APIs
                                                    │
                                                    ▼
                                                Final Answer



