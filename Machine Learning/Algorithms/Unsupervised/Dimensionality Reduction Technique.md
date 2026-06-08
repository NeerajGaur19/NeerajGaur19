### Dimensionality Reduction is the process of reducing the number of input features (columns) in a dataset while retaining as much useful information as possible.

Simple Example

Suppose you have a dataset with 100 features:

Age	Salary	Height	Weight	...	Feature100

Dimensionality reduction may transform these 100 features into 10 or 20 new features that capture most of the important information.

## Why Do We Need It?

### 1. Reduces Complexity

Fewer features means:

* Faster training
* Less memory usage
* Simpler models

### 2. Removes Redundant Features

Some features may contain similar information.

Example:

Height (cm)	     Height (m)
  180	              1.80

Both provide essentially the same information.

### 3. Reduces Overfitting

Too many features can cause a model to memorize noise instead of learning patterns.

### 4. Helps Visualization

Humans can easily visualize:

* 1D
* 2D
* 3D

But not 100 dimensions.

Dimensionality reduction can convert:

100 features → 2 features

for plotting and understanding the data.

# Types of Dimensionality Reduction

## 1. Feature Selection

Keep the most important existing features and remove the rest.

Example:

### Original Features:

* Age
* Salary
* Gender
* Employee_ID

### Remove:

Employee_ID

### Keep:

* Age
* Salary
* Gender

### Methods:

* Correlation
* Chi-Square Test
* Mutual Information
* Recursive Feature Elimination (RFE)

## 2. Feature Extraction

Create new features from existing features.

Example:

Original:

<img width="778" height="116" alt="image" src="https://github.com/user-attachments/assets/0804be75-5897-42a2-b308-171150c6066b" />

New Feature: 

<img width="132" height="85" alt="image" src="https://github.com/user-attachments/assets/67bd6767-552e-479c-be34-3036bafc9464" />

### Popular techniques:

* Principal Component Analysis (PCA)
* Linear Discriminant Analysis (LDA)
* t-Distributed Stochastic Neighbor Embedding
* Uniform Manifold Approximation and Projection

