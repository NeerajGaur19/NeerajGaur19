
# What is K-Means?

K-Means is an algorithm that divides data into K groups (clusters).

Here,

K = Number of clusters you want.

Example:

If

K = 2

the algorithm creates 2 clusters.

If

K = 5

it creates 5 clusters.

---

# Why is it called K-Means?

* K = Number of clusters
* Means = Mean (average) position of all points in a cluster

The average point is called the centroid.

Think of the centroid as the center of a cluster

---

# Example

Suppose we have students based on:

    Student	  Height
        A	      150
        B	      152
        C	      155
        D	      178
        E	      180
        F	      182

You can clearly see two groups:

Group 1: 150, 152, 155

Group 2: 178, 180, 182

If we choose

K = 2

K-Means will automatically create these two clusters.

---

# Step-by-Step Working of K-Means

Imagine points on a graph:

    A •      B •

          C •


                     D •
                 E •

                        F •

Suppose

K = 2

## Step 1: Choose K

Choose the number of clusters.

K = 2

## Step 2: Select Initial Centroids

Randomly pick two points as centroids.

For example:

Centroid 1 = A

Centroid 2 = F

These are just starting guesses.

## Step 3: Calculate Distance

Every point checks:

Which centroid is closer?

Usually, K-Means uses Euclidean distance (the straight-line distance).

### Example:

Point C

Distance to Centroid A = 5

Distance to Centroid F = 20

Since 5 is smaller,

C joins Cluster 1.

Every point does the same.

# Step 4: Form Clusters

Now suppose we get

Cluster 1
    
    A
    B
    C

Cluster 2

    D
    E
    F

## Step 5: Find New Centroids

Now calculate the average position of each cluster.

Example:

Cluster 1

    A
    B
    C

New centroid = Average of A, B, and C

Similarly,

Cluster 2

    D
    E
    F

New centroid = Average of D, E, and F

## Step 6: Repeat

Again calculate distances using the new centroids.

Some points may switch clusters.

Again calculate new centroids.

Repeat until nothing changes.

This is called convergence.

---

# Final Output

Cluster 1

    A
    B
    C

Centroid ●

Cluster 2

    D
    E
    F

Centroid ●

The algorithm stops.

---

# Visual Representation

Before

    A      B

     C

          D

       E

             F

## After clustering

Cluster 1

    A  B
     C

      |------------|

Cluster 2

      D
    E
        F

---

# Flow of the Algorithm

      Start

        ↓

    Choose K

        ↓

    Select Random Centroids

        ↓

    Calculate Distance

        ↓

    Assign Points to Nearest Centroid

        ↓

    Calculate New Centroids

        ↓

    Clusters Changed?

      Yes
      
       ↓
    
    Repeat

      No
    
       ↓

    Stop

# Small Numerical Example

Suppose we have numbers

    2
    3
    4
    10
    11
    12

Choose

    K = 2

Random centroids

    3 and 11

Assign numbers

Cluster 1

    2
    3
    4

Cluster 2

    10
    11
    12

New centroids

Cluster 1

    (2+3+4)/3 = 3

Cluster 2

    (10+11+12)/3 = 11

No changes.

Algorithm stops.

# Advantages of K-Means

✅ Simple to understand

✅ Easy to implement

✅ Fast on large datasets

✅ Works well when clusters are clearly separated

# Disadvantages of K-Means

❌ You must choose K before running the algorithm.

❌ Sensitive to the initial random centroids; different starting points can produce different results.

❌ Doesn't work well for irregularly shaped clusters.

❌ Sensitive to outliers (extreme values), which can pull centroids away from the true center.
