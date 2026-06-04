
# 1. Unsupervised Learning 

a type of Machine Learning where the model learns from unlabeled data.

* Supervised Learning: Data has input (X) and output (Y).
* Unsupervised Learning: Data has only input (X), and the model tries to find hidden patterns, groups, or relationships by itself.

<img width="720" height="327" alt="image" src="https://github.com/user-attachments/assets/abf281e6-285f-49b6-ae68-55d02f10075c" />

<br>

You don't know which customers belong to which category.

An unsupervised algorithm can automatically discover:

* Young high spenders
* Older low spenders
* Medium-value customers

without being told the correct labels.

# 2. How Unsupervised Learning Works

Raw Data
    ↓
Find Patterns
    ↓
Create Groups / Relationships
    ↓
Insights

Unlike supervised learning, there is no target column.


### Example

<img width="967" height="466" alt="image" src="https://github.com/user-attachments/assets/16dac2f8-774a-48b5-b9ef-51bf6d55b34e" />

<br>


# 3. Main Types of Unsupervised Learning

## 3.1. Clustering

Groups similar data points together.

Customers
   ↓
Cluster 1 → Students
Cluster 2 → Working Professionals
Cluster 3 → Senior Citizens

### 3.1.1 Popular Clustering Algorithms

* K-Means
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Model
  
### 3.1.2 Use Cases

* Customer segmentation
* Market analysis
* Recommendation systems
* Image segmentation

## 3.2 Dimensionality Reduction

Reduces the number of features while preserving important information.

Example:

100 Features
     ↓
10 Features

### 3.2.1 Benefits:

* Faster training
* Less memory usage
* Better visualization
* Reduced noise

### 3.2.2 Popular Algorithms

* Principal Component Analysis (PCA)
* t-SNE
* UMAP

### 3.2.3 Use Cases
* Data visualization
* Feature extraction
* Noise reduction

## 3.3 Association Rule Learning

Finds relationships between items.

Example:

People who buy Bread
       ↓
Often buy Butter

### 3.3.1 Popular Algorithms
* Apriori
* FP-Growth

### 3.3.2 Use Cases
* Market basket analysis
* Product recommendations
* Cross-selling

# 4. Most Common Algorithms

<img width="662" height="352" alt="image" src="https://github.com/user-attachments/assets/85e0b7c4-2abc-4a81-a53c-cb71f69e8eab" />

<br/>

# 5. Advantages

✅ No labeled data required

✅ Finds hidden patterns

✅ Useful for exploratory data analysis (EDA)

✅ Can discover unknown groups in data

# 6. Limitations

❌ No ground truth labels to evaluate easily

❌ Results can be difficult to interpret

❌ Sensitive to parameter choices (e.g., number of clusters in K-Means)

❌ Different algorithms may produce different groupings
