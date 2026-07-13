
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


## Self-Attention Mechanism (Step-by-Step with Real Numbers)

<img width="1536" height="1024" alt="ChatGPT Image Jul 9, 2026, 12_00_22 AM" src="https://github.com/user-attachments/assets/8cf94b2a-f4d5-402f-b678-36f8f76fb14c" />

### Problem Statement

Consider the sentence:

    "I love machine learning"

The Transformer needs to understand the relationship between every word in the sentence.

For example:

* Does "machine" relate more to "learning"?
* Does "love" relate more to "I"?
* Which words should each word pay attention to?

This is exactly what the Self-Attention Mechanism does.

## Step 1: Convert Words into Embeddings

Every word is converted into a numerical vector called an embedding.

Assume embedding size = 4

    Word	     Embedding
    I	        [1, 0, 1, 2]
    love	    [0, 2, 1, 1]
    machine	    [3, 1, 0, 2]
    learning	[2, 2, 1, 0]


The complete embedding matrix is

<img width="258" height="152" alt="image" src="https://github.com/user-attachments/assets/996e76ac-3b0c-4599-bbcc-66d3198c309c" />

Shape: 

    (4 × 4)
    4 words
    Embedding dimension = 4

Each row represents one word.

## Step 2: Create Query (Q), Key (K) and Value (V)

Self-attention does not use embeddings directly.

Instead, the embeddings are transformed into three different representations.

The Transformer learns three weight matrices:

    WQ
    WK
    WV

These matrices are initialized randomly and learned during training.

Suppose

    WQ (4×2)
    
    1 0
    0 1
    1 1
    0 2

    WK (4×2)
    
    0 1
    1 1
    2 0
    1 2
    
    WV (4×2)
    
    2 1
    1 0
    0 1
    1 2

Now compute
    
    Q = X × WQ
    
    K = X × WK
    
    V = X × WV

## Step 3: Compute Query Matrix

Multiply

    Q = X × WQ

Result
    
    Word	    Query
    I	        [2,5]
    love	    [1,5]
    machine	    [3,5]
    learning	[3,3]

Shape

    Q = (4 × 2)

## Step 4: Compute Key Matrix

    K = X × WK

Result

    Word	     Key
    I	        [4,5]
    love	    [4,4]
    machine	    [3,8]
    learning	[5,4]

Shape

    K = (4 × 2)

## Step 5: Compute Value Matrix

V = X × WV

Result
    
    Word	    Value
    I	        [4,6]
    love	    [3,3]
    machine	    [8,7]
    learning	[6,3]

Shape

    V = (4 × 2)

## What do Query, Key and Value mean?

Think of them like this:

Query : What information am I looking for?

Key : What information do I contain?

Value : What information should I send if another word attends to me?

## Step 6: Compute Attention Scores

Now compare every Query with every Key.

    Attention Scores = Q × Kᵀ

Why transpose?

    Q : (4×2)
    
    K : (4×2)
    
    Kᵀ : (2×4)
    
    Result
    
    (4×2) × (2×4)
    
    =
    
    (4×4)

### Result

<img width="905" height="282" alt="image" src="https://github.com/user-attachments/assets/151379b1-edb8-4f2d-b576-eb64372f0dc5" />

### Interpretation:

For the word machine

    35 32 49 35

means

    Machine vs I

    Machine vs love

    Machine vs machine
    
    Machine vs learning
    
The highest score is

    49

which means

    machine is most related to itself.

## Step 7: Scale the Scores

Large values can make Softmax unstable.

Therefore divide by

<img width="110" height="63" alt="image" src="https://github.com/user-attachments/assets/b95e147a-9b65-46bc-bff7-10f2a04041f1" />


Here

    dk = 2

So

<img width="135" height="43" alt="image" src="https://github.com/user-attachments/assets/589766e0-8a22-4c7d-928d-29a78079addb" />


### Scaled scores become

<img width="763" height="242" alt="image" src="https://github.com/user-attachments/assets/4e570de2-99d0-4424-a552-8d248b5b6617" />

## Step 8: Apply Softmax

Softmax converts scores into probabilities.

Each row sums to 1.

### Result

<img width="752" height="251" alt="image" src="https://github.com/user-attachments/assets/965d748c-9953-432a-8a2c-4052ccdec42f" />

Example

For the word machine

    0.06
    0.03
    0.81
    0.10

Meaning

    6% attention to I
    
    3% attention to love
    
    81% attention to machine
    
    10% attention to learning

The model has learned that machine should focus mostly on itself.




# Multi-Head Attention

<img width="1536" height="1024" alt="Multi-Head Attention" src="https://github.com/user-attachments/assets/b56f42a5-323f-4bb1-98ed-3ffb9ffb3de5" />

