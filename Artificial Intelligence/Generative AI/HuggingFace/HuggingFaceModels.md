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
        
    pipeline("fill-mask")
    
    pipeline("ner")

    pipeline("sentiment-analysis")

  
## Available tasks are 

    'any-to-any', 
    'audio-classification', 
    'automatic-speech-recognition', 
    'depth-estimation', 
    'document-question-answering', 
    'feature-extraction', 
    'fill-mask', 
    'image-classification', 
    'image-feature-extraction', 
    'image-segmentation', 
    'image-text-to-text', 
    'keypoint-matching', 
    'mask-generation', 
    'ner', 
    'object-detection', 
    'sentiment-analysis', 
    'table-question-answering', 
    'text-classification', 
    'text-generation', 
    'text-to-audio', 
    'text-to-speech', 
    'token-classification', 
    'video-classification', 
    'zero-shot-audio-classification', 
    'zero-shot-classification', 
    'zero-shot-image-classification', 
    'zero-shot-object-detection'
