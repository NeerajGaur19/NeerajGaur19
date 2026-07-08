
The attention mechanism is a technique in deep learning that allows a model to focus on the most relevant parts of the input when generating an output, rather than treating every input equally.

### Intuition

Imagine you're translating the sentence:

English: "The cat sat on the mat."

When translating the word "cat", you mainly need to pay attention to "cat" in the input, not every other word. Attention lets the model learn which words are most important at each step.

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
