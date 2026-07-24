# Transformers Self-Attention Mechanism (Step-by-Step with Real Numbers)

<img width="1536" height="1024" alt="ChatGPT Image Jul 9, 2026, 12_00_22 AM" src="https://github.com/user-attachments/assets/8cf94b2a-f4d5-402f-b678-36f8f76fb14c" />

<img width="1536" height="1024" alt="Attention Mechanism - How It Works" src="https://github.com/user-attachments/assets/77b5c151-d993-4e9e-aa5c-0a1a14c31d86" />


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

## Step 9: Compute Context Vector

Now multiply

    Attention Weights × V

    (4×4)
    ×
    (4×2)
    =
    (4×2)

### Output
    
    Word	        Context Vector
    I	            [6.32,6.71]
    love	        [6.35,6.76]
    machine	        [6.46,6.94]
    learning	    [5.95,6.55]

These are the new representations of the words.


## What is a Context Vector?

Originally

    machine
    
    ↓
    
    Embedding
    
    [3,1,0,2]

After self-attention

    machine
    
    ↓
    
    Context Vector
    
    [6.46,6.94]

Now the word machine contains information from

* I
* love
* machine
* learning

instead of only itself.

This makes the word representation context-aware.

## Complete Self-Attention Pipeline

        Sentence
           │
           ▼
        Word Embeddings (X)
           │
           ▼
        Multiply with WQ, WK, WV
           │
           ▼
        Q       K       V
           │
           ▼
        Q × Kᵀ
           │
           ▼
        Scale by √dk
           │
           ▼
        Softmax
           │
           ▼
        Attention Weights
           │
           ▼
        Multiply by V
           │
           ▼
        Context Vectors

# Key Formula

<img width="447" height="95" alt="image" src="https://github.com/user-attachments/assets/6951ffb8-b78d-41c6-9cf4-58b039ce7549" />

