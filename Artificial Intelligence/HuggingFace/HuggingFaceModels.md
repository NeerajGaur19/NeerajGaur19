# HuggingFace Models

    dslim/bert-base-NER
    distilbert/distilbert-base-uncased-finetuned-sst-2-english
    mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis



---

# HuggingFace Classes


                    Class	                                                  Task
              AutoModel	                                              Base transformer (embeddings/hidden states)
              AutoModelForSequenceClassification	                    Text classification
              AutoModelForTokenClassification	                        Named Entity Recognition
              AutoModelForQuestionAnswering	                          Question answering
              AutoModelForMaskedLM	                                  Masked language modeling (e.g., BERT)
              AutoModelForCausalLM	                                  Text generation (e.g., GPT)


---

# Supported Pipelines

    pipeline("text-classification")
    
    pipeline("text-generation")
    
    pipeline("summarization")
    
    pipeline("translation")
    
    pipeline("question-answering")
    
    pipeline("fill-mask")
    
    pipeline("ner")

    pipeline("sentiment-analysis")

  
