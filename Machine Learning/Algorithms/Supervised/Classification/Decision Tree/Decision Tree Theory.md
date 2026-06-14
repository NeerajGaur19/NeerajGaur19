
A Decision Tree is one of the easiest Machine Learning algorithms to understand because it works like the way humans make decisions.

Think of it as a series of Yes/No questions.

## Real-Life Example

Suppose a bank wants to predict:

Will a customer subscribe to a fixed deposit?

The model might ask:

    Age > 40?
           |
        Yes/No

If Yes:

    Balance > 50000?
           |
        Yes/No

If No:

    Campaign > 3?
           |
        Yes/No

Finally it reaches a decision:

    Subscribe = Yes

or

    Subscribe = No

This sequence of questions forms a Decision Tree.

### Example Dataset

<img width="752" height="255" alt="image" src="https://github.com/user-attachments/assets/02bf945a-b3e1-4cbd-847e-e3c3cfe8cd9f" />

The tree learns patterns automatically.

Structure of a Decision Tree

                  Root Node
                      |
                Age > 40 ?
                 /      \
               No        Yes
              /            \
         Balance>20000?   Subscribe=Yes
           /      \
          No      Yes
          |         |
    Subscribe=No Subscribe=Yes

## Terminology

### 1. Root Node

The first question.

    Age > 40?

It contains the entire dataset.

### 2. Decision Node

Intermediate questions.

    Balance > 20000?

### 3. Leaf Node

Final prediction.

    Subscribe = Yes

or

    Subscribe = No
    
### How Does the Tree Decide Which Question to Ask?

This is the most important concept.

The tree tries to find:

Which feature separates the classes best?

Example:

Features:

    Age
    Balance
    Campaign
    Duration

The algorithm checks all of them and chooses the one that creates the purest groups.

### What is Purity?

Suppose we have:

    10 customers
    
    5 Yes
    5 No

Mixed data.

Not pure.

Another group:

    10 customers
    
    10 Yes
    0 No

Perfectly pure.

The goal:

    Mixed Data
          ↓
    Split
          ↓
    Pure Groups

### Gini Impurity

The most common criterion in Decision Trees.

Measures how mixed a node is.

Pure Node
      
      10 Yes
      0 No

Gini = 0

Perfect.

### Mixed Node

    5 Yes
    5 No

Gini = 0.5

Very impure.

The algorithm chooses the split that reduces impurity the most.

### Example

Suppose:

    Age > 40?

creates:

Group 1:

    8 No
    1 Yes

Group 2:

    9 Yes
    2 No

Very good separation.

The tree may choose Age as the first split.

## Training Process

The algorithm:

    Start with all data
            ↓
    Find best feature
            ↓
    Split data
            ↓
    Repeat for each branch
            ↓
    Stop when conditions met

## How Does Prediction Work?

Suppose a new customer arrives:

<img width="586" height="91" alt="image" src="https://github.com/user-attachments/assets/b0012f86-3a46-40a1-9ee0-ef72eefea090" />

The tree asks:

    Age > 40 ?
    
    Yes
    
    ↓
    
    Balance > 50000 ?
    
    Yes
    
    ↓
    
    Subscribe = Yes

Prediction completed.

## Visual Example
          
                Age > 40?
                /       \
              No         Yes
             /             \
      Balance>20000?     YES
         /      \
       No        Yes
       |           |
      NO          YES

#### Customer:

    Age = 35
    Balance = 25000

Path:

    Age > 40 ?
    No
    
    Balance > 20000 ?
    Yes
    
    Prediction = Yes
    
## Important Hyperparameters

### 1. max_depth

Maximum depth of the tree.

    DecisionTreeClassifier(max_depth=3)

Example:

    Depth 1
      ↓
    Depth 2
      ↓
    Depth 3

Stops growing after depth 3.

### 2. min_samples_split

Minimum samples required to split.

    min_samples_split=20

If a node has fewer than 20 samples:

    No further splitting

### 3. min_samples_leaf

Minimum samples allowed in a leaf.

    min_samples_leaf=5

Prevents tiny leaf nodes.

## Overfitting Problem

Decision Trees can memorize training data.

Example:

    Tree becomes huge
    
    Age > 41?
    Balance > 52143?
    Duration > 233?
    Campaign > 2?
...

Training Accuracy:

    100%

Test Accuracy:

    70%

This is overfitting.

## Solution: Pruning

Limit tree growth using:

    max_depth
    min_samples_split
    min_samples_leaf

This produces a simpler tree.
