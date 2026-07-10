
The attention mechanism is a technique in deep learning that allows a model to focus on the most relevant parts of the input when generating an output, rather than treating every input equally.

### Intuition

Imagine you're translating the sentence:

English: "The cat sat on the mat."

When translating the word "cat", you mainly need to pay attention to "cat" in the input, not every other word. Attention lets the model learn which words are most important at each step.

### Step-by-step

Suppose the input is:

I love machine learning

For the word "learning":

    Create its Query vector.
    Compare it with the Key vectors of all words.
    Compute similarity scores.
    Apply softmax to convert scores into probabilities.
    Take the weighted sum of the Value vectors.

Example attention weights:

<img width="935" height="241" alt="image" src="https://github.com/user-attachments/assets/78d7311b-1e75-4b5d-a5da-eaa7ff57b09e" />

The output representation is:

0.05V1​+0.10V2​+0.35V3​+0.50V4

​
So "learning" pays the most attention to itself and to "machine."

## Step 1: How Encoder-Decoder Works (Without Attention)

Suppose we want to translate:

English

I love machine learning

into French.

A Seq2Seq model has two parts:

    English Sentence
            │
            ▼
    +----------------+
    |    Encoder     |
    +----------------+
            │
            │ Final Hidden State
            ▼
    +----------------+
    |    Decoder     |
    +----------------+
            │
            ▼
    French Sentence


### What does the Encoder do?

The encoder reads one word at a time.

        I
        ↓
        love
        ↓
        machine
        ↓
        learning

After reading every word, it updates its hidden state.

Let's imagine the hidden states look like this.

        After "I"
        
        H1
        
        After "love"
        
        H2
        
        After "machine"
        
        H3
        
        After "learning"
        
        H4

The important thing is:

The decoder only receives H4 (the final hidden state).

        I → H1
        love → H2
        machine → H3
        learning → H4
        
        Decoder gets only H4

Think of H4 as a summary of the whole sentence.


<img width="1536" height="1024" alt="ChatGPT Image Jul 9, 2026, 12_00_22 AM" src="https://github.com/user-attachments/assets/8cf94b2a-f4d5-402f-b678-36f8f76fb14c" />

# Multi-Head Attention

<img width="1536" height="1024" alt="Multi-Head Attention" src="https://github.com/user-attachments/assets/b56f42a5-323f-4bb1-98ed-3ffb9ffb3de5" />

