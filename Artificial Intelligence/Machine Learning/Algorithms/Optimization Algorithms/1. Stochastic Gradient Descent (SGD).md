
Stochastic Gradient Descent (SGD) is an optimization algorithm used to train machine learning and deep learning models. It minimizes the loss (error) by updating the model's parameters (weights and bias) in the direction that reduces the error.

SGD is the most widely used optimizer for training Neural Networks, Logistic Regression, Linear Regression, and many Deep Learning models.


Why Do We Need Gradient Descent?

Suppose we want to predict house prices.

Our model is

y=wx+b

Initially,

    Weight (w) = Random
    
    Bias (b) = Random

The predictions are poor.

We need a way to adjust w and b so that predictions become more accurate.

This is what Gradient Descent does.

## Think of Climbing Down a Mountain

Imagine standing on top of a mountain.

                *
               / \
              /   \
             /     \
            /       \
    _______/_________\_____
    
Your goal is to reach the lowest point (the valley).

The mountain represents the loss function.
    
    Top = High Error
    Bottom = Minimum Error

Gradient Descent finds the lowest point.

# Loss Function

For Linear Regression, the common loss is Mean Squared Error (MSE).

<img width="237" height="72" alt="image" src="https://github.com/user-attachments/assets/92279fdf-7ac2-4d5e-8c08-9882c2e08c7c" />

Lower loss means better predictions.

---

# Types of Gradient Descent

There are three main types.

## 1. Batch Gradient Descent

Uses the entire dataset to compute one update.

    Entire Dataset
          │
    Compute Gradient
          │
    Update Weights

Example:

    Dataset = 100,000 records
    
    Uses all 100,000 records
    
    One weight update

### Advantages

* Stable updates
* Accurate gradient

### Disadvantages

* Slow
* High memory usage

## 2. Stochastic Gradient Descent (SGD)

Uses only one training example at a time.

    Record 1 → Update
    
    Record 2 → Update
    
    Record 3 → Update
    
    Record 4 → Update

Every sample updates the weights immediately.

### Example

Suppose the dataset has

10,000 records

SGD does

    Record 1 → Update weights
    
    Record 2 → Update weights
    
    Record 3 → Update weights

    ...
    
    Record 10,000 → Update weights

10,000 updates occur in one epoch.

## Stochastic means random.

The training data is usually shuffled randomly, and the algorithm updates the weights after processing each randomly selected sample.


## 3. Mini-Batch Gradient Descent

Instead of using:

* Entire dataset
* One sample

It uses a small batch.

Example:

    Batch Size = 32
    
    Records 1–32
    
    ↓
    
    Update
    
    Records 33–64
    
    ↓
    
    Update

Mini-batch GD is the most commonly used approach in deep learning because it balances speed and stability.


# Comparison

<img width="637" height="187" alt="image" src="https://github.com/user-attachments/assets/71e37c3a-d8d3-4e5d-b05c-1d200a6a41a0" />

---

## How SGD Works

Suppose

    Weight = 5
    
    Learning Rate = 0.1

Sample 1

    Prediction Error
    
    ↓
    
    Gradient = +2

Update

    New Weight
    
    = Old Weight − Learning Rate × Gradient
    
    = 5 − 0.1 × 2
    
    = 4.8

Then the next sample is processed immediately.

## SGD Algorithm

Initialize weights randomly

Repeat until convergence

    Shuffle data

    For each training sample

        Predict

        Calculate loss

        Compute gradient

        Update weights

## Epoch

One complete pass through the dataset.

Example:

Dataset

1000 records

One epoch means

Record 1

↓

Record 2

↓

...

↓

Record 1000

After all records are processed,

    Epoch = 1 completed

Training usually runs for multiple epochs (e.g., 10, 50, or 100).

## Learning Rate

The learning rate determines how big a step we take while updating the weights.

If the learning rate is:

### Too Small

    Tiny steps
    
    ↓
    
    Very slow learning

### Too Large

    Huge jumps
    
    ↓
    
    May overshoot the minimum
    
    ↓
    
    Training may fail

### Good Learning Rate

    Steady movement
    
    ↓
    
    Fast convergence


## Why Does SGD Look Noisy?

Unlike Batch Gradient Descent,

SGD uses only one sample at a time.

Each sample has a different error.

Therefore,

    Loss
    
    ↓
    
    Up
    
    ↓
    
    Down
    
    ↓
    
    Up
    
    ↓
    
    Down

The path is noisy, but on average it moves toward the minimum.


# Visualization

## Batch Gradient Descent

    \
     \
      \
       \
        \
         Minimum

Smooth path.

## SGD

    \
     \/\
     /\ \
    /  \ \
         \/
          Minimum

Noisy path.

## Mini-Batch
    
    \
     \_
       \__
          \_
            Minimum

Between Batch and SGD.

---

## Advantages of SGD

* Very fast for large datasets
* Low memory usage
* Can escape shallow local minima and saddle points due to noisy updates
* Suitable for online learning (streaming data)
* Widely used in deep learning

## Disadvantages

* Noisy convergence
* May oscillate around the optimum
* Sensitive to learning rate
* Usually requires more epochs than Batch Gradient Descent

