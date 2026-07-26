
Bootstrap Aggregation, commonly called Bagging, is an ensemble learning technique used to improve the accuracy and stability of machine learning models by reducing variance and preventing overfitting.

Bagging is mainly used with models that have high variance, such as Decision Trees.

Bagging consists of two ideas:

## 1. Bootstrap

Bootstrap means random sampling with replacement.

Suppose we have a dataset with 10 records.

Original Dataset

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10

Create a bootstrap sample by randomly selecting records with replacement.

Example:

Bootstrap Sample 1

      2
      4
      4
      6
      8
      9
      9
      10
      3
      1

Notice:

* Record 4 appears twice.
* Record 9 appears twice.
* Records 5 and 7 are missing.

This is completely normal because sampling is with replacement.

Create another bootstrap sample:

Bootstrap Sample 2

    1
    1
    3
    5
    6
    7
    8
    8
    9
    10

Every bootstrap sample is different.


Step-by-Step Working of Bagging

Suppose we have this dataset:

1000 Records

### Step 1

Generate many bootstrap samples.

    Dataset
        │
        ├── Sample 1
        ├── Sample 2
        ├── Sample 3
        ├── Sample 4
        └── Sample 5

### Step 2

Train one model on each sample.
    
    Sample 1 → Tree 1
    
    Sample 2 → Tree 2
    
    Sample 3 → Tree 3
    
    Sample 4 → Tree 4
    
    Sample 5 → Tree 5

Each tree is trained independently.

Unlike boosting, one tree does not depend on another.

Step 3

Combine predictions.

             Tree1
               │
             Tree2
               │
             Tree3
               │
             Tree4
               │
             Tree5
               │
         Final Prediction

This combining process is called Aggregation.

## How Aggregation Works

Classification

Bagging uses Majority Voting.

Example:

    Tree 1 → Yes
    Tree 2 → Yes
    Tree 3 → No
    Tree 4 → Yes
    Tree 5 → No

Votes:

    Yes = 3
    
    No = 2

Final prediction:

    YES

## Regression

Bagging uses the Average.

Example:

    Tree1 = 200
    
    Tree2 = 210
    
    Tree3 = 190
    
    Tree4 = 205
    
    Tree5 = 195

Average:

    (200 + 210 + 190 + 205 + 195) / 5

    = 200

Final prediction:

    200


## Why Does Bagging Work?

Suppose one decision tree overfits.

    Tree1 Accuracy = 98%

Another tree:

    Tree2 Accuracy = 86%

Another:

    Tree3 Accuracy = 91%

Each tree makes different mistakes.

When combined,

    Majority Voting

reduces random errors and usually improves generalization.


## Random Forest is Bagging

Random Forest is simply Bagging + Random Feature Selection.

    Bagging
    
    Bootstrap Samples
          │
    Decision Trees
          │
    Majority Voting

↓

    Random Forest
    
    Bootstrap Samples
          │
    Random Feature Selection
          │
    Decision Trees
          │
    Majority Voting

The random feature selection makes the trees less correlated, which often improves performance further.


## Advantages of Bagging

    Reduces overfitting
    Reduces variance
    Improves model stability
    Works well with decision trees
    Easy to parallelize because models are independent
    Handles noisy data better than a single model
    Usually improves prediction accuracy

## Disadvantages of Bagging

    Requires more memory
    Training multiple models increases computation
    Less interpretable than a single decision tree
    Does not significantly reduce bias (it mainly reduces variance)

# Bagging vs Boosting


<img width="885" height="475" alt="image" src="https://github.com/user-attachments/assets/e1ea29e8-c570-438b-9613-f7b9cbe36820" />

