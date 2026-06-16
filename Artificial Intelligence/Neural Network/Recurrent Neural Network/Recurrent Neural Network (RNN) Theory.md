
# What is a Recurrent Neural Network (RNN)?

A Recurrent Neural Network (RNN) is a type of Artificial Neural Network specifically designed to process sequential data, where the order of data matters.

Unlike traditional feedforward neural networks, RNNs have memory that allows information from previous inputs to influence current outputs.

Examples of Sequential Data

* Text and Natural Language Processing (NLP)
* Speech Recognition
* Time Series Forecasting
* Stock Price Prediction
* Machine Translation
* Video Analysis
* Sensor Data Processing

---

# Why Traditional Neural Networks Are Not Enough?

Consider the sentence:

"The clouds are dark because it is going to ____."

To predict the missing word ("rain"), the model needs information from previous words.

A standard neural network processes inputs independently and cannot remember previous words.

RNN solves this by maintaining a hidden state (memory).

---

# Basic Architecture of RNN

At each time step:

* Input = Current element
* Hidden State = Memory from previous step
* Output = Prediction

## Mathematical Representation

For time step t:

### Hidden State

<img width="343" height="46" alt="image" src="https://github.com/user-attachments/assets/52852535-d145-47aa-957c-ac7308a9020b" />

### Output

<img width="216" height="52" alt="image" src="https://github.com/user-attachments/assets/e69c7ecc-d614-45ab-bc6f-bff41f3ad543" />

Where:

xt   = Current input
ht   = Hidden state
ht−1 = Previous hidden state
W = Weight matrices
b = Bias
f = Activation function (usually tanh)

# Unrolled RNN

Imagine the sentence:

    I → Love → Deep → Learning

Unrolled structure:

    x1 --> [RNN] --> h1
                 |
    x2 --> [RNN] --> h2
                 |
    x3 --> [RNN] --> h3
                 |
    x4 --> [RNN] --> h4

The same network parameters are reused at every time step.

# Key Components

## 1. Input Layer

Receives sequence data.

Example:

    "I love AI"

Input sequence:

    x1 = I
    x2 = love
    x3 = AI

## 2. Hidden Layer

Stores memory.

    Current Memory =
    Previous Memory + Current Input

## 3. Output Layer

Generates predictions.

Examples:

* Next word prediction
* Sentiment classification
* Speech transcription


# Types of RNN

## 1. One-to-One

Traditional Neural Network

    Input → Output

Example:

    Image → Label

Not truly an RNN use case.

## 2. One-to-Many

Single input produces sequence output.

    Input → Output1 Output2 Output3 ...

### Example

Image Captioning

    Image →
    "A dog is running"

Diagram:

    Image
       |
       V
    RNN → word1
         → word2
         → word3

## 3. Many-to-One

Sequence input, single output.

### Example

Sentiment Analysis

    "I love this movie"

Output:

    Positive

Diagram:

    x1 → x2 → x3 → x4
                |
             Output

Applications:

* Sentiment Analysis
* Spam Detection
* Document Classification

## 4. One-to-Many (Sequence Generation)

Example:

    Start token

Generate:

    Hello → How → Are → You

Applications:

* Text Generation
* Music Generation

## 5. Many-to-Many (Synchronized)

Input and output lengths are same.

Example:

Part-of-Speech Tagging

    I      love    AI
    PRON   VERB   NOUN

Diagram:

    x1 → y1
    x2 → y2
    x3 → y3

## 6. Many-to-Many (Encoder-Decoder)

Input and output lengths differ.

Example: Machine Translation

English:

    I love AI

French:

    J'aime l'IA

Architecture:

    Encoder RNN
    Input Sequence
          ↓
    Context Vector
          ↓
    Decoder RNN
    Output Sequence

Applications:

* Translation
* Chatbots
* Summarization

# Problems with Vanilla RNN

Although powerful, RNNs have major limitations.

## 1. Vanishing Gradient Problem

During backpropagation, gradients become extremely small.

Example:

    Word1 → Word2 → Word3 → ... → Word100

The network forgets early words.

Result:
    * Cannot learn long-term dependencies.

## 2. Exploding Gradient Problem

Gradients become extremely large.

Results:

* Unstable training
* Numerical overflow
