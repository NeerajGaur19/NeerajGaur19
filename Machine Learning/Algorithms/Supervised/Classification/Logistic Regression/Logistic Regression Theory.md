### Logistic Regression is a Classification Algorithm, not a regression algorithm.

Logistic Regression as: A Linear Regression model whose output is passed through a Sigmoid function to convert it into a probability, which is then used to classify data into categories.

It is mainly used when the target variable has two classes:

* Yes / No
* Pass / Fail
* Spam / Not Spam
* Customer Buys Product / Doesn't Buy Product


### Real-Life Example

Suppose a bank wants to predict whether a customer will subscribe to a fixed deposit.

<img width="768" height="272" alt="image" src="https://github.com/user-attachments/assets/023dd101-75df-4940-89e7-6a04768d72f6" />

### Goal: 

Given Age and Salary, predict whether the customer will subscribe.

### Why Not Use Linear Regression?

Linear Regression can predict any value:

    -2
    0.5
    3.8
    10

But for classification, we need only:

    0 = No
    1 = Yes

So Linear Regression is not suitable.

### Main Idea

Logistic Regression first calculates a score:

    z = b0 + b1x1 + b2x2 + ...

Similar to linear regression.

For example:

    z = -5 + 0.1(Age)

If Age = 40

    z = -5 + 0.1 × 40
    z = -1

But z can be any number:
    
    -100
    -5
    0
    20
    100

We need a probability between 0 and 1.

### Sigmoid Function

This is the heart of Logistic Regression.

<img width="987" height="330" alt="image" src="https://github.com/user-attachments/assets/6db14c36-2c82-463b-9966-0a64c229adba" />

The sigmoid converts any number into a probability between 0 and 1.

<img width="642" height="388" alt="image" src="https://github.com/user-attachments/assets/ce08c376-350e-42f3-9b5f-1cdb4028fbf8" />


### How Prediction Happens

Suppose sigmoid output is:

    0.85

Interpretation:

    85% chance customer will subscribe

Apply threshold:

    if probability >= 0.5:
        predict = 1
    else:
        predict = 0

### Decision Boundary

The line that separates classes.

For binary classification:

    Probability < 0.5 → Class 0
    Probability ≥ 0.5 → Class 1

Usually:

    Threshold = 0.5

### Training Process

The model starts with random weights.

Example:

* Age coefficient = 0.2
* Salary coefficient = -0.1

Predictions will be poor.

The algorithm then:

    Makes predictions
    Calculates error
    Adjusts weights
    Repeats thousands of times

until error becomes small.

### Cost Function

Linear Regression uses:

    MSE (Mean Squared Error)

Logistic Regression uses:

    Log Loss (Cross Entropy Loss)

because probabilities are being predicted.

#### Log Loss Idea

Good prediction:

    Actual = 1
    Predicted = 0.99

Small loss.

Bad prediction:

    Actual = 1
    Predicted = 0.01

Huge loss.

### Gradient Descent

To minimize Log Loss, Logistic Regression uses:

    Gradient Descent

Steps:

    Initialize weights
           ↓
    Predict
           ↓
    Calculate loss
           ↓
    Update weights
           ↓
    Repeat

### Output Interpretation

Suppose model predicts:

    model.predict_proba(X)

Output:

    [0.25, 0.75]

Meaning:

    25% chance Class 0
    75% chance Class 1

Prediction:

    Class 1

because 0.75 > 0.5

### Advantages

✅ Simple

✅ Fast

✅ Easy to interpret

✅ Works well for linearly separable data

✅ Gives probabilities

### Limitations

❌ Assumes a linear relationship between features and log-odds

❌ Struggles with complex patterns

❌ Sensitive to outliers

❌ Needs feature scaling for best performance

### When Should You Use Logistic Regression?

Use it when:

* Binary classification problems
* Small to medium datasets
* Need explainability
* Need probabilities

Examples:

* Spam detection
* Disease prediction
* Customer churn prediction
* Bank deposit subscription prediction
* Loan default prediction

### Complete Flow

    Input Features
    (Age, Salary, Balance)
    
            ↓
    
    Linear Equation
    z = b0 + b1x1 + b2x2
    
            ↓
    
    Sigmoid Function
    
            ↓
    
    Probability (0 to 1)
    
            ↓
    
    Threshold (0.5)
    
            ↓
    
    Class 0 or Class 1

