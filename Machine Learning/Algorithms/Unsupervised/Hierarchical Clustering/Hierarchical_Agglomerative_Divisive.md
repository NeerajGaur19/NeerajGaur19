# Hierarchical Clustering

Hierarchical Clustering is an unsupervised machine learning algorithm that groups similar data points into clusters by building a hierarchy (tree) of clusters.

Instead of creating all clusters at once (like K-Means), it merges or splits clusters step by step.

<img width="944" height="492" alt="image" src="https://github.com/user-attachments/assets/4ab2291b-e75a-48ce-8eac-c3db44a50031" />

# Types

## 1. Agglomerative clustering

There will be no initial k value like we had in k-means clustering.
    
Step 1 : For example here we have 6 data points and each point is its own cluster.
    
Step 2 : Prepare Proximity marix - How far away the points (Initial clusters from Step 1) are from each other
    
Step 3 : Points closer to each other (distance between points is less compared to other point's distance) are put in one cluster.

<img width="820" height="435" alt="image" src="https://github.com/user-attachments/assets/6f77b885-b74b-4619-92ee-808be72a92d4" />



## 2. Divisive clustering
