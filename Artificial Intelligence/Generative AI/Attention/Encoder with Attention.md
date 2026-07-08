
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

