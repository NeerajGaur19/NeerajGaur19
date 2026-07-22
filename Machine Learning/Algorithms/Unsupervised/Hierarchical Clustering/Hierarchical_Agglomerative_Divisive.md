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


## Mathematical working of Agglomerative Clustering

Lets say there are 5 data points. A, B , C , D, E are the data points.

## Proximity matrix

<img width="894" height="374" alt="image" src="https://github.com/user-attachments/assets/18541daa-d251-4d39-99a6-1aaa6354af59" />

Now A and E are closest, so A,E is now one cluster. New proximity matrix will be

<img width="939" height="453" alt="image" src="https://github.com/user-attachments/assets/c64bb430-f056-4f93-a406-af1cb0f5abb3" />

how we calculate distance between (A,E) to B ?

For this we use

## linkage methods

Note: Each linkage method may indicate different clusters are closest to each other. We apply all linkage and whichever is giving best result that cluster points are kept together next iteration.

How do we decide best result is that where intercluster disctance is high and each cluster is well defined.

    Single Linkage
    
    Complete Linkage
    
    Average Linkage
    
    Word Linkage

<img width="372" height="259" alt="image" src="https://github.com/user-attachments/assets/1d341960-2db8-450c-9977-511e17c062d9" />

## 1. Single Linkage

Distance between their two closest point in clusters is termed as Single linkage.

<img width="321" height="263" alt="image" src="https://github.com/user-attachments/assets/d78ec58f-3cf9-4874-bffb-eafe629ed119" />

## 2. Complete Linkage

Distance between their two farthest point in clusters is termed as Complete linkage.

<img width="310" height="257" alt="image" src="https://github.com/user-attachments/assets/6ca48865-0fa8-45a3-a6e4-3926bee8bb64" />

## 3. Average Linkage

Distance between centroids of clusters is termed as average linkage.

<img width="342" height="264" alt="image" src="https://github.com/user-attachments/assets/0d1aa873-c77a-4e5d-88f9-46d09ad6324a" />

### How we get centroid to centroid distance ?

Average of distances : Distance of each data point in one cluster to other points of 2nd cluster.

For example heredistance between 4 data points to each of 5 points in 2nd cluster and then we take average of those distances.

<img width="374" height="266" alt="image" src="https://github.com/user-attachments/assets/38f6a0ea-4006-43f0-902a-54c1bad64d52" />

## 4. Ward Linkage

Sum of sqaure of all the points of one cluster to distance with all the points of another cluster.

<img width="471" height="268" alt="image" src="https://github.com/user-attachments/assets/9eee377b-4b92-4f08-b6f9-7641c98149b9" />

<img width="771" height="345" alt="image" src="https://github.com/user-attachments/assets/86d4b32b-f69c-4a31-93b0-f9b9ec924790" />

For example lets use, complete Linkage method

<img width="449" height="312" alt="image" src="https://github.com/user-attachments/assets/2986c7b5-8c8b-46cb-bf89-df68c5f68071" />

To combine next points into cluster we see that C and D are closest.

<img width="530" height="314" alt="image" src="https://github.com/user-attachments/assets/54c5ea3b-202e-4441-9866-3afb6038e0c7" />

Using Complete linkage (Farthest data point)

<img width="503" height="334" alt="image" src="https://github.com/user-attachments/assets/4c95ff8a-20cd-40b3-932c-d5db19900bf3" />

Smallest distance indicate next cluster to be formed. Here we see that (A,E) is close to B cluster.

New proximity matrix is formed. Here we see that (A,E) to (C,D) is 9

And B to (C,D) is 5. So, we take maximum distance as per complete linkage definition (farthest distance).

<img width="835" height="464" alt="image" src="https://github.com/user-attachments/assets/6e1fa290-9270-44cc-805b-08ba573a1ec7" />


<img width="817" height="503" alt="image" src="https://github.com/user-attachments/assets/cb23d5e5-f51a-4395-a30d-f0b22c8813cc" />

## Final cluster

<img width="505" height="409" alt="image" src="https://github.com/user-attachments/assets/a7336f12-f4fd-4004-a7fc-b7891d97e7e9" />



## 2. Divisive clustering
