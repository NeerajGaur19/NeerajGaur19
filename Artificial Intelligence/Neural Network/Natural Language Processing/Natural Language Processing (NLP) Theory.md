
# What is NLP?

Natural Language Processing (NLP) is a branch of:

Artificial Intelligence (AI) → Machine Learning (ML) → Deep Learning (DL) → Natural Language Processing (NLP)

NLP enables computers to understand, interpret, process, and generate human language (text or speech).

## Examples of NLP Applications

* ChatGPT
* Google Translate
* Siri, Alexa
* Email Spam Detection
* Sentiment Analysis
* Text Summarization
* Speech Recognition
* Chatbots
* Language Translation

# Why NLP is Needed?

Computers understand numbers, not human language.

Human Language

    "I love this movie."

Computer Understanding

    [0.23, 0.54, 0.91, 0.12]

NLP converts human language into a numerical format that machine learning models can process.

# NLP Pipeline

    Raw Text
        ↓
    Text Cleaning
        ↓
    Tokenization
        ↓
    Stopword Removal
        ↓
    Stemming/Lemmatization
        ↓
    Feature Extraction
        ↓
    Machine Learning / Deep Learning Model
        ↓
    Prediction

 ## Example:

    Original Sentence:
    "I am learning NLP from ChatGPT"
    
    Tokenization:
    ["I", "am", "learning", "NLP", "from", "ChatGPT"]
    
    Stopword Removal:
    ["learning", "NLP", "ChatGPT"]
    
    Lemmatization:
    ["learn", "NLP", "ChatGPT"]

# Major Components of NLP

## 1. Lexical Analysis

  Studies words and characters.

  Example:

    ChatGPT is amazing

  Words:

    ChatGPT | is | amazing

  ### This process is called Tokenization.

## 2. Syntax Analysis

Checks grammatical structure. Also called Parsing.

Example:

    She is reading a book.

Grammar is correct.

Example:

    She reading book is.

Grammar is incorrect. Syntax analysis identifies grammatical relationships.

## 3. Semantic Analysis

Determines meaning.

Example:

    I saw a man with a telescope.

Question:

Who has the telescope?

* Me?
* The man?

Semantic analysis attempts to determine the intended meaning.

## 4. Discourse Integration

Understands context across sentences.

Example:

    John bought a car.
    He loves it.

NLP must understand:

    He = John
    it = car

## 5. Pragmatic Analysis

Understands real-world intent.

Example:

    Can you open the door?

Literal meaning: Question about ability.

Actual meaning: Request to open the door.

# Text Preprocessing

Before building NLP models, text must be cleaned.

## 1. Lowercasing

    text = "HELLO WORLD"

Output:

    hello world

## 2. Remove Punctuation

    Hello!!!

  ↓

    Hello

## 3. Remove Numbers

    Phone: 12345

  ↓

    Phone

## 4. Tokenization

Breaking text into words.

    "I love NLP"

  ↓

    ['I','love','NLP']

Using NLTK:

    from nltk.tokenize import word_tokenize

    word_tokenize("I love NLP")

## 5. Stop Word Removal

Common words that add little meaning.

Examples:

    is
    am
    the
    a
    an
    of
    in
    on

Sentence:

    I am learning NLP

  ↓

    learning NLP

6. Stemming

Cuts words to root form.

Examples:

    playing → play
    played → play
    plays → play

Using Porter Stemmer:

    from nltk.stem import PorterStemmer

Result may not always be a real word.

Example:

    studies → studi

## 7. Lemmatization

Converts words to dictionary form.

Examples:

    studies → study
    running → run
    better → good

Using:

    from nltk.stem import WordNetLemmatizer

Lemmatization is usually more accurate than stemming.

---

# Feature Extraction Techniques

Machines cannot understand text directly.

Convert text into numbers.

## 1. Bag of Words (BoW)

Count frequency of words.

Sentences:

    I love NLP
    I love AI

Vocabulary:

    I, love, NLP, AI

Feature Matrix:

<img width="816" height="150" alt="image" src="https://github.com/user-attachments/assets/ef1bfb80-211f-48e2-a792-11eb0c909a53" />

## 2. N-Grams

Capture neighboring words.

### Sentence:

    I love machine learning

### Unigram

    I, love, machine, learning

### Bigram

    I love
    love machine
    machine learning

### Trigram

    I love machine
    love machine learning

## 3. TF-IDF

Term Frequency – Inverse Document Frequency

Highlights important words.

Formula:

    TF-IDF = TF × IDF

High score:

* Important in one document
* Rare in other documents

Used extensively in:

* Search Engines
* Document Classification

### Word Embeddings

Traditional methods ignore meaning.

Example:
    
    King
    Queen
    Man
    Woman

BoW treats them as unrelated.

Embeddings capture semantic meaning.

### Word2Vec

Developed by Tomas Mikolov.

Example:

    King - Man + Woman = Queen

Words become vectors.

    King → [0.23, 0.56, 0.91 ...]


### GloVe

Developed by Stanford University NLP Group. Uses global word co-occurrence statistics.

### FastText

Developed by Meta AI. Handles unknown words better by learning sub-word information.

# Deep Learning in NLP

Traditional NLP:

    Text → Feature Engineering → ML Model

Deep Learning:

    Text → Neural Network → Prediction

No manual feature engineering required.

# Popular NLP Libraries

## NLTK

    import nltk

For:

* Tokenization
* Stopwords
* Stemming
* Lemmatization

## spaCy

    import spacy

Fast industrial NLP library.

Used for:

* NER
* POS Tagging
* Dependency Parsing

## Transformers

    from transformers import pipeline

Used for:

* BERT
* GPT
* T5
* RoBERTa

