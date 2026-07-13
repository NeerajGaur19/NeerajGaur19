# Transformers – Positional Encoding

## Why Do We Need Positional Encoding?

Before understanding Positional Encoding, let's compare how RNNs and Transformers process a sentence.

### RNN / LSTM

Suppose the sentence is

    I love machine learning

RNN processes words one by one.

    I
    ↓
    love
    ↓
    machine
    ↓
    learning

The hidden state flows from one word to the next.

    h0 → h1 → h2 → h3 → h4

Since words are processed sequentially, the model naturally knows:

* "I" is the first word
* "love" is the second word
* "machine" is the third word
* "learning" is the fourth word

The order is built into the computation.

Transformer

Transformer processes all words simultaneously.

Instead of

    I
    ↓
    love
    ↓
    machine
    ↓
    learning

it receives

    I
    love
    machine
    learning

all at once.

Each word is converted into an embedding.

Suppose
    
    Word	    Embedding
    I	        [1 0 1 2]
    love	    [0 2 1 1]
    machine	  [3 1 0 2]
    learning	[2 2 1 0]

Embedding matrix

X

    [
    1 0 1 2
    0 2 1 1
    3 1 0 2
    2 2 1 0
    ]

Notice something important.

These embeddings contain meaning.

They do not contain position.

## The Problem

Imagine we shuffle the words.

Sentence 1

    I love machine learning

Sentence 2

    machine learning love I

Both contain the same words.

Only the order changes.

Without positional information,

Transformer only sees

    I
    love
    machine
    learning

as four vectors.

It does not know

* which word comes first
* which word comes last
* which words are adjacent

## Another Example

Consider

    Dog bites man

and

    Man bites dog

Same words.

Different meaning.

Without position,

both sentences would look almost identical.

The model cannot distinguish them correctly.

## Why Self-Attention Cannot Learn Order

Recall Self-Attention

<img width="133" height="132" alt="image" src="https://github.com/user-attachments/assets/44e45af3-5d15-4e0a-935e-eb7c70078c65" />
	​
Then

<img width="358" height="70" alt="image" src="https://github.com/user-attachments/assets/70ebb138-2635-43d9-a2ab-14cd5f3faddc" />

Notice

Everything comes from

    X

There is nothing in this equation that tells the model:

* which word is first
* second
* third

Self-Attention only learns relationships.

Not positions.

## Solution

We add a vector representing the position.

Every word receives

    Word Embedding

    +

    Position Embedding

Instead of

    Embedding

Transformer receives

    Embedding + Position


## Example

Suppose

Embedding dimension = 4

Word embeddings

    I
    
    [1 0 1 2]
    
    love
    
    [0 2 1 1]
    
    machine
    
    [3 1 0 2]
    
    learning
    
    [2 2 1 0]

Suppose positional vectors are

    Position	Positional Vector
      0	        [0.1 0.2 0.3 0.4]
      1	        [0.2 0.3 0.4 0.5]
      2	        [0.3 0.4 0.5 0.6]
      3	        [0.4 0.5 0.6 0.7]

Now add them.

First word

    Embedding
    
    [1 0 1 2]
    
    +
    
    Position
    
    [0.1 0.2 0.3 0.4]
    
    =
    
    [1.1
    0.2
    1.3
    2.4]
    
Second word

    [0 2 1 1]
    
    +
    
    [0.2 0.3 0.4 0.5]
    
    =
    
    [0.2
    2.3
    1.4
    1.5]

Continue for every word.

Now every vector contains

* meaning
* position


# Input to Transformer

Instead of

    Embeddings
    
    ↓
    
    Self Attention

we now have

    Embeddings
    
    +
    
    Positional Encoding
    
    ↓
    
    Self Attention


## Why Addition?

Many beginners ask:

    Why don't we concatenate position with the embedding?

Suppose

Embedding dimension

    512

Position vector

    512

If concatenated,

    512 + 512 = 1024

Every layer would now expect 1024 features.

The Transformer architecture would become much larger.

Instead,

we simply add them.

    512
    
    +
    
    512
    
    =
    
    512

The size stays the same.

This is computationally efficient.

## Where Does the Positional Vector Come From?

The original Transformer paper does not learn the positional vectors.

Instead, it computes them using sine and cosine functions.

<img width="620" height="276" alt="image" src="https://github.com/user-attachments/assets/397d4aa1-599f-4118-ba73-d4ea2754b247" />

For position pos and embedding dimension index i:

<img width="330" height="122" alt="image" src="https://github.com/user-attachments/assets/eee307cf-925e-47bc-8d1a-e4a6f03f144e" />

where:

* pos = word position (0, 1, 2, ...)
* d = embedding dimension
* even dimensions use sine
* odd dimensions use cosine


## Why Sine and Cosine?

The sinusoidal functions have useful properties:

* Every position gets a unique encoding.
* Nearby positions have similar encodings.
* Relative distances between words can be inferred.
* The model can generalize to sequence lengths longer than those seen during training.

For example:

    Position 0
    
    [0.00
    1.00
    0.00
    1.00]

Position 1

    [0.84
    0.54
    0.01
    0.99]

Position 2

    [0.91
    -0.42
    0.02
    0.99]

Every position gets a distinct vector.

## Complete Flow

    Sentence
    
    I love machine learning
            │
            ▼
    
    Word Embeddings
            │
            ▼
    
    Positional Encoding
            │
            ▼
    
    Add Both Vectors
            │
            ▼
    
    Final Input Embeddings
            │
            ▼
    
    Create Q, K, V
            │
            ▼
    
    Self-Attention
            │
            ▼
    
    Multi-Head Attention
            │
            ▼
    
    Feed Forward Network

