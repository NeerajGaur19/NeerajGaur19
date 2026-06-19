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

---

# Common Regression Algorithms

## 1. Linear Regression

Simplest regression algorithm.

It tries to fit a straight line:

<img width="1006" height="327" alt="image" src="https://github.com/user-attachments/assets/bf21717f-db67-4f2c-92bd-abde5bd1b194" />

#### 1.1 Where:

* y = predicted value
* m = slope / weightage 
* b = intercept / base value

#### 1.2 Use Case

* Predict house prices
* Predict salary based on experience

#### 1.3 Advantages

* Simple and fast
* Easy to interpret

#### 1.4 Disadvantages

* Works poorly with non-linear data
* Sensitive to outliers

## 2. Multiple Linear Regression

Extension of linear regression with multiple input variables.

y=b0​+b1 ​x1​+b2​ x2​+⋯+bn ​xn​

#### 2.1 Example

Predict house price using:

* area
* bedrooms
* location
* age

## 3. Polynomial Regression

Used when relationship is curved instead of straight.

Example equation:

<img width="1171" height="322" alt="image" src="https://github.com/user-attachments/assets/cce1cded-c3f9-4d6c-85bc-8969d46400da" />

#### 3.1 Use Case

* Growth curves
* Temperature trends

#### 3.2 Advantage
* Can model non-linear relationships.

#### 3.3 Disadvantage
* Can overfit easily.

## 4. Ridge Regression
* Linear regression with L2 regularization.
* Helps reduce overfitting and multicollinearity.

#### 4.1 Key Idea
* Adds penalty on large coefficients.

## 5. Lasso Regression
* Linear regression with L1 regularization.

#### 5.1 Special Feature
* Can reduce some coefficients to zero → feature selection.

## 6. Elastic Net Regression

#### 6.1 Combination of:
* Ridge Regression
* Lasso Regression

#### 6.2 Useful when:
* many correlated features exist

## 7. Decision Tree Regression
* Uses tree structure for prediction.

#### 7.1 Advantages
* Handles non-linear data 
* No scaling required

Disadvantages
* Can overfit

## 8. Random Forest Regression
* Collection of many decision trees.

#### 8.1 Advantages
* Better accuracy
* Reduces overfitting
* Handles complex data

#### 8.2 Disadvantages
* Slower than linear regression
* Less interpretable

## 9. Support Vector Regression (SVR)
Regression version of SVM.

Idea
Finds a boundary where prediction error stays within a margin.

Advantages

(i) Works well with high-dimensional data

(ii) Handles non-linear relationships using kernels like RBF

Disadvantages
(i) Slow for large datasets

(ii) Requires scaling

10. K-Nearest Neighbors (KNN) Regression
Prediction based on nearby data points.

Advantages
(i) Simple

(ii) No training phase

Disadvantages
(i) Slow for large datasets

(ii) Sensitive to scaling

11. Gradient Boosting Regression
Builds models sequentially to reduce errors.

Popular implementations:

(i) XGBoost

(ii) LightGBM

(iii) CatBoost

Advantages
(i) Very high accuracy

(ii) Widely used in competitions

Disadvantages
(i) More tuning required

(ii) Slower training    

---

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
