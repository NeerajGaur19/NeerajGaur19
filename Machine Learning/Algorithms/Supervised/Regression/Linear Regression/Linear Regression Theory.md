# Overview of Linear Regression 

## 1. What is Linear Regression?

Linear Regression is one of the simplest and most widely used Machine Learning algorithms used to predict a continuous numeric value based on the given input value(s).

Examples:

<img width="772" height="274" alt="image" src="https://github.com/user-attachments/assets/06d0204f-19b0-4430-973c-c750fcc0ece4" />

The algorithm tries to find the best-fit line that represents the relationship between input and output variables.
This is implemented on data where output column is numerical and not categories.

Example:

<img width="634" height="312" alt="image" src="https://github.com/user-attachments/assets/9608c335-05ac-402d-a3cf-a0cdb9379107" />

A line can be drawn:

Salary = 2 × Experience + 1

If Experience = 6:

Salary = 2(6) + 1 = 13 LPA

## 2. Why Linear Regression?

Suppose a company wants to estimate salary based on experience.
Instead of manually guessing:

* 6 years → ?
* 8 years → ?
* 10 years → ?

## 3. Intuition Behind Linear Regression

Imagine plotting data points:

<img width="469" height="420" alt="image" src="https://github.com/user-attachments/assets/124602ec-2fb9-4099-9969-29ac5c24799f" />

Linear Regression draws the best possible line through these points. This line is called: Best Fit Line. We train a Linear Regression model that learns the pattern automatically.

<img width="666" height="670" alt="image" src="https://github.com/user-attachments/assets/8373fd97-44a4-431d-8338-536888abb160" />


## 4. What Happens Internally

Linear regression repeatedly:

* Guesses a line. 
* Measures how wrong the line is.
* Calculates how the line should move.
* Adjusts the line slightly.
* Repeats until the error becomes minimal.

## 5. Linear equation

Core Idea

Linear regression assumes that the target variable can be approximated as a weighted sum of the input features:

y=β0+β1x1+β2x2+⋯+βnxn

Where:

* y = predicted output
* x1,x2,...,xn = input features
* β0 = intercept (bias term)
* β1,β2,...,βn = coefficients (weights)

For example, to predict house prices:

Price=β0+β1(Area)+β2(Bedrooms)

## Advantages

* Simple and fast.
* Easy to interpret.
* Works well when relationships are approximately linear.
* Requires relatively little training data.

## Limitations

* Assumes linear relationships.
* Sensitive to outliers.
* Can struggle with highly complex patterns.
* Multicollinearity (highly correlated features) can make coefficients unstable.

## 1. MAE (Mean Absolute Error)

Most intuitive metric.

Formula:

<img width="377" height="92" alt="image" src="https://github.com/user-attachments/assets/cbd3b644-977a-4948-a018-649a1248ce12" />

Example:

<img width="742" height="216" alt="image" src="https://github.com/user-attachments/assets/39adf6e8-e113-420a-b55b-e631f0c74d44" />

MAE:

(10 + 20 + 20) / 3 = 16.67

### Interpretation

On average, predictions are off by 16.67 units.

<img width="1536" height="1024" alt="Linear Regression Algorithm Workflow" src="https://github.com/user-attachments/assets/f0097652-1172-4ceb-a018-eebaff0108af" />

## 2. MSE (Mean Squared Error)

Instead of absolute values, errors are squared.

Example:

<img width="607" height="212" alt="image" src="https://github.com/user-attachments/assets/dce6bc9c-339f-4074-bbc4-a42ab38f8202" />

### Why square?

Large mistakes get punished heavily.

Example:

    Error = 2
    Squared = 4
    
    Error = 20
    Squared = 400

The bigger error hurts much more.

## 3. RMSE (Root Mean Squared Error)

Most commonly used metric.

Take square root of MSE.

<img width="197" height="41" alt="image" src="https://github.com/user-attachments/assets/ee54eb4b-206d-42a8-bef8-f1d2627fdaad" />

If:

    MSE = 300

Then:

    RMSE = √300 ≈ 17.32

### Interpretation

    Predictions are off by about 17.32 units on average.

### Why RMSE is Popular?

    Same unit as target variable
    Punishes large errors
    Easy to compare models

## 4. R² Score (Coefficient of Determination)

Most important interview metric.

Question:

    How much of the variation in the target variable is explained by the model?

Range:

    0 to 1

Sometimes negative.

## Interpretation

<img width="696" height="352" alt="image" src="https://github.com/user-attachments/assets/41e54862-9f09-412a-b0d0-b95d1f71db1c" />

### Example

House prices depend on:
        
        Area
        Bedrooms
        Location
        Age

If:

        R² = 0.85

Meaning:

    85% of house price variation is explained by the model.
