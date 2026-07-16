
# Multi-Head Attention

<img width="1536" height="1024" alt="Multi-Head Attention" src="https://github.com/user-attachments/assets/b56f42a5-323f-4bb1-98ed-3ffb9ffb3de5" />

## Step 1 Input Sentence

Suppose the input sentence is

    I love machine learning

There are 4 words.

    Word1 = I
    Word2 = love
    Word3 = machine
    Word4 = learning

## Step 2 Convert Words into Embeddings

Assume embedding dimension = 4
    
    Word	    Embedding
    I	        [1 0 1 2]
    love	    [0 2 1 1]
    machine	  [3 1 0 2]
    learning	[2 2 1 0]

The embedding matrix becomes

    X (4 × 4)
    
    [
    1 0 1 2
    0 2 1 1
    3 1 0 2
    2 2 1 0
    ]

Each row represents one word.

## Step 3 Why Can't We Use One Attention?

Suppose we want to understand

    The bank is near the river.

The word

    bank

may represent

* Financial institution
* River bank

One attention head may learn location.

Another may learn grammar.

Another may learn meaning.

Another may learn subject-object relationships.

Therefore we create multiple attention heads.

Suppose

    Number of heads = 2

## Step 4 Head 1 Creates Q, K and V

Head 1 has three trainable matrices

    WQ₁
    WK₁
    WV₁

Suppose

    WQ₁
    
    [
    1 0
    0 1
    1 1
    0 2
    ]

    WK₁
    
    [
    0 1
    1 1
    2 0
    1 2
    ]

    WV₁
    
    [
    2 1
    1 0
    0 1
    1 2
    ]

Now compute

    Q₁ = XWQ₁

Similarly

    K₁ = XWK₁
    
    V₁ = XWV₁

Perform every multiplication.


## Step 5 Compute Attention Scores

Compute

    Q₁K₁ᵀ

Result

    4 × 4

Every element tells

    How much Word i attends to Word j

## Step 6 Scale

Instead of

    QKᵀ

use

    QKᵀ
    ---------
    √dk

Suppose

    dk = 2
    
    √2 = 1.414

Divide every element.

This prevents Softmax from becoming too peaked.

## Step 7 Apply Softmax

Suppose one row becomes

    [2
    4
    1
    3]

Softmax

↓

    [0.09
    0.66
    0.03
    0.22]

Meaning

    Current word gives
    
    9%
    
    66%
    
    3%
    
    22%

    attention

These are Attention Weights.

# Step 8 Compute Context Vector

Multiply

    Attention Weights
    
    ×
    
    V

Result

    Context Vector

Every row becomes a new representation of the word.

# Step 9 Head 2

Now Head 2 has different matrices

    WQ₂
    WK₂
    WV₂

These are initialized differently and learned independently.

Therefore
    
    Head 2
    
    ≠
    
    Head 1

It learns completely different relationships.

Repeat

    Q₂
    
    ↓
    
    K₂
    
    ↓
    
    V₂
    
    ↓
    
    Attention
    
    ↓
    
    Context₂

## Step 10 Multiple Heads

Suppose

Head 1 Output

    4 × 2

Head 2 Output

    4 × 2

Concatenate

    Head1
    
    [
    a b
    c d
    e f
    g h
    ]

    Head2
    
    [
    i j
    k l
    m n
    o p
    ]

Result

    [
    a b i j
    c d k l
    e f m n
    g h o p
    ]

Dimension

    4 × 4

## Step 11 Linear Projection

Multiply by

    WO

    Output
    
    =
    
    Concatenated Heads
    
    ×
    
    WO

Why?

Because every head has learned different information.

Linear projection mixes information from all heads into one meaningful representation.

## Step 12 Final Output

The encoder now outputs

    Word1'
    
    Word2'
    
    Word3'
    
    Word4'

Each word now contains information about every other word.

Example

Original

    bank

After Multi-Head Attention

    bank
    
    ↓
    
    [Rich contextual representation]

# Complete Flow

    Sentence
          │
          ▼
    Embeddings (X)
          │
          ▼
    ───────────────────────────────
    Head 1
    XWQ₁ → Q₁
    XWK₁ → K₁
    XWV₁ → V₁
          │
          ▼
    Q₁K₁ᵀ
          │
          ▼
    Scale (÷√dk)
          │
          ▼
    Softmax
          │
          ▼
    Attention₁ × V₁
          │
          ▼
    Output₁
    ───────────────────────────────
    
    ───────────────────────────────
    Head 2
    XWQ₂ → Q₂
    XWK₂ → K₂
    XWV₂ → V₂
          │
          ▼
    Q₂K₂ᵀ
          │
          ▼
    Scale
          │
          ▼
    Softmax
          │
          ▼
    Attention₂ × V₂
          │
          ▼
    Output₂
    ───────────────────────────────
    
              │
              ▼
    
    Concatenate Outputs
    
              │
              ▼
    
    Linear Projection (WO)
    
              │
              ▼
    
    Final Multi-Head Attention Output
