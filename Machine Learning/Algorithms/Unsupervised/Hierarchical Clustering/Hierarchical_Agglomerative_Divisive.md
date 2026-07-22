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


<img width="800" height="451" alt="image" src="https://github.com/user-attachments/assets/e7067586-2b91-48ed-8081-cd27b46ac602" />

0 and 4 are closer so placed in one cluster. Next time , it is observed that 1 and 5 are closer so another cluster. Then 2 and 3 are closer so another cluster.

<img width="710" height="449" alt="image" src="https://github.com/user-attachments/assets/f234fd78-f624-483f-bc61-22c38a446ae0" />

To visualize the clusters, we use dendogram. Dendogram is nothing but another visual representation of clustering process.

## Dendrogram

<img width="703" height="456" alt="image" src="https://github.com/user-attachments/assets/10df4465-df56-4920-b17f-b7e8a35df8f8" />


<img width="792" height="454" alt="image" src="https://github.com/user-attachments/assets/82895e38-03bd-40af-b0db-5478e91707d5" />

To calculate distance in hierarchial clustering, we use Euclidean Distance.

<img width="820" height="239" alt="image" src="https://github.com/user-attachments/assets/722e59d1-f6d9-4106-9249-9e51c7bfd7ad" />


## 2. Divisive clustering
