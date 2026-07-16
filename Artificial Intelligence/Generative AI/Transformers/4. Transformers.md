
<img width="513" height="691" alt="image" src="https://github.com/user-attachments/assets/a9084ba0-d549-476b-96ff-19df37f7bcf7" />

Transformers are deep learning models introduced in the paper "Attention Is All You Need" (2017). 
They rely entirely on the Attention Mechanism—no RNNs or LSTMs—which allows massive parallelization and better long-range dependency learning.

## Key Features

* No recurrence or convolution
* Entirely based on Self-Attention
* Captures long-range dependencies efficiently
* Highly parallelizable (faster training)
* Scales well with large datasets and compute

## Core Idea

Each word in a sequence looks at all other words at once to decide how much attention to pay to them, and builds a rich representation.

## Building Blocks

### 1. Input Embedding

Convert tokens to vectors.

### 2. Positional Encoding

Add position information (since there is no recurrence).

### 3. Multi-Head Self-Attention

Compute attention in parallel using multiple attention heads.

### 4. Feed Forward Network (FFN)

Two linear layers with a non-linearity.

### 5. Add & Norm

Residual connection + Layer Normalization.

### 6. Masked Multi-Head Attention (in Decoder)

Prevents looking at future tokens.

# Trasformers Encoder

