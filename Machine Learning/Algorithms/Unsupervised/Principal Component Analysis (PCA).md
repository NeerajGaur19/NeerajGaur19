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

### Step 2: Compute Covariance Matrix

Covariance tells us how features move together.

Example:

<img width="623" height="156" alt="image" src="https://github.com/user-attachments/assets/59dc799c-6955-402c-96a8-1dea1b31d6fb" />

Large covariance indicates redundancy. PCA identifies these relationships.


### Step 3: Calculate Eigenvectors and Eigenvalues

This is the mathematical heart of PCA.

Eigenvectors : Represent directions.

Eigenvalues : Represent importance of each direction.

Think of it like:

<img width="637" height="216" alt="image" src="https://github.com/user-attachments/assets/16b1f054-9ad1-4e1b-918b-2e9db066d846" />

Total variance: 90+8+2=100

Variance explained:

* PC1 = 90%
* PC2 = 8%
* PC3 = 2%

# PCA Mathematics

The principal components come from the eigenvectors of the covariance matrix.

The transformation can be represented as:

Z=XW

Where:

* X = standardized data
* W = matrix of selected eigenvectors
* Z = transformed data (principal components)
<br> You don't need to compute this manually in practice.


## Explained Variance Ratio

One of the most important outputs.

### Example:

      from sklearn.decomposition import PCA
      
      pca = PCA()
      pca.fit(X_scaled)

      print(pca.explained_variance_ratio_)

### Output:

[0.75, 0.15, 0.07, 0.03]

<img width="642" height="307" alt="image" src="https://github.com/user-attachments/assets/6340c0ef-fce4-4197-bbbf-b0fe3d645460" />

Total of first two PCs: 75+15=90%

So you could reduce 4 features to 2 while keeping 90% of the information.

## Choosing Number of Components

### Method 1: Explained Variance

Keep enough components to explain:

* 90%
* 95%
* 99%

of the variance.

      pca = PCA(n_components=0.95)

This automatically keeps enough components to retain 95% variance.

### Method 2: Scree Plot

      import matplotlib.pyplot as plt
      
      plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
               pca.explained_variance_ratio_.cumsum())

Choose the "elbow point" where additional components add little information.

## PCA in Scikit-Learn

      from sklearn.preprocessing import StandardScaler
      
      from sklearn.decomposition import PCA
      
      # Scale
      
      scaler = StandardScaler()
      
      X_scaled = scaler.fit_transform(X)
      
      # PCA
      
      pca = PCA(n_components=2)
      
      X_pca = pca.fit_transform(X_scaled)

### Now:

      print(X.shape)

### Output:

      (1000, 20)

### After PCA:

      print(X_pca.shape)

### Output:

      (1000, 2)

20 features become 2.

## Advantages of PCA

✅ Reduces dimensionality

✅ Faster model training

✅ Removes multicollinearity

✅ Helps visualization

✅ Reduces storage requirements

✅ Removes some noise


## Disadvantages of PCA

❌ Components are hard to interpret

Instead of:

* Age
* Salary

You get:

* PC1
* PC2

which have no direct business meaning.

❌ Information loss may occur.

❌ Assumes linear relationships.

❌ Sensitive to feature scaling.


## PCA vs Feature Selection

<img width="737" height="262" alt="image" src="https://github.com/user-attachments/assets/1612147a-acea-4a16-9fbb-a98b0aded3db" />

## PCA vs LDA

<img width="722" height="260" alt="image" src="https://github.com/user-attachments/assets/100bc8e4-10bd-4bb1-b2d7-c181c1049620" />


