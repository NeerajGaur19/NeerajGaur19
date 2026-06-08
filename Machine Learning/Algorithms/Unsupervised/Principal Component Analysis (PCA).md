# Principal Component Analysis (PCA) – Detailed Explanation

PCA is one of the most important dimensionality reduction techniques in Machine Learning.

### Its goal is to: Reduce the number of features while preserving as much information (variance) as possible.

## The Problem PCA Solves

Imagine a dataset with:

<img width="622" height="268" alt="image" src="https://github.com/user-attachments/assets/4bbb928f-5d8b-4d2d-9d1a-c4b0d9bcf934" />

Notice that:

* Taller people tend to weigh more.
* Height and Weight are highly correlated.
* Both features contain similar information.

Instead of keeping both features, PCA tries to combine them into fewer variables called Principal Components.

## What is a Principal Component?

A Principal Component is a new feature created from the original features.

These new features:

* Are combinations of original features
* Capture maximum variance
* Are uncorrelated with each other

For example:

* PC1=0.7×Height+0.7×Weight
* PC2=0.7×Height−0.7×Weight

PC1 captures most of the information. <br>
PC2 captures the remaining information.

Often, PC1 alone may explain 90% + of the variance.

## Intuition Behind PCA

Suppose we plot Height vs Weight.
<br> The points might look like this:

<img width="321" height="203" alt="image" src="https://github.com/user-attachments/assets/b7274754-cbe9-4f22-9a63-4b83874836a1" />

Most variation happens along the diagonal direction. PCA finds this direction automatically.

### First Principal Component (PC1)

The direction with maximum variance. <br>
<img width="212" height="161" alt="image" src="https://github.com/user-attachments/assets/c004b2ce-9c0e-4660-8517-401798d510ca" />

### Second Principal Component (PC2)

Perpendicular to PC1.

<img width="187" height="122" alt="image" src="https://github.com/user-attachments/assets/f526b75c-66d7-4b41-bb39-6a999950d3ab" />

PC2 contains much less information. Therefore we may keep only PC1.

### Variance in PCA

PCA assumes: Features with higher variance contain more information.
<br> It searches for the direction where the data varies the most.

### Visualization

Low Variance

    * * * * *

High Variance

    *      *      *      *      *

Higher spread = Higher variance.
<br> PCA tries to preserve this spread.

## PCA Workflow

### Step 1: Standardize Data

Since PCA depends on variance, scaling is important.
<br> Example:

<img width="680" height="157" alt="image" src="https://github.com/user-attachments/assets/06b08020-40c9-410d-8519-76a6ef7bf5c9" />

Salary would dominate PCA.
<br> Therefore:

    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


