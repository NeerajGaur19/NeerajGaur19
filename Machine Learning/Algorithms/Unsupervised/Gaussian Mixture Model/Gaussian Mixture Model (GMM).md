
Gaussian Mixture Model (GMM) is an unsupervised machine learning algorithm used for clustering and density estimation.

Unlike K-Means, which assigns each data point to exactly one cluster, GMM assigns probabilities that a data point belongs to each cluster.

It is one of the most important probabilistic clustering algorithms.

# What is a Gaussian Distribution?

A Gaussian Distribution is another name for the Normal Distribution (Bell Curve).

Example:

Suppose we measure the heights of 10,000 people.

Most people have average height.

Very few are extremely short or extremely tall.

                     *
                  *     *
                *         *
              *             *
            *                 *
    ------*----------------------*---------

The curve is called a Gaussian Curve.


# Mean and Standard Deviation

A Gaussian distribution is defined by only two parameters.

## Mean (μ)

The center of the bell curve.

Example
    
    Marks
    
    40 50 60 70 80

         ↑

       Mean = 60

## Standard Deviation (σ)

Measures how spread out the data is.

Small σ

        /\
       /  \
      /    \

Large σ

         /\
        /  \
       /    \
      /      \
     /        \
    
## Real Life

Suppose

Customer Age
    
    18
    20
    22
    21
    19
    20
    21
    22

These ages approximately follow one Gaussian.

But another customer segment:

    55
    56
    57
    58
    60

forms another Gaussian.

The overall dataset actually contains two Gaussian distributions.     


## Why K-Means Fails

Imagine this data.

    Young Customers
    
    ●●●●●●●

Older Customers

    □□□□□□□

Suppose there are a few customers in between.

K-Means makes a hard assignment.

Example

Age = 40

K-Means says

    100% Young

or

    100% Old

No uncertainty.


## GMM Thinks Differently

Instead of saying

    Customer = Young

It says

Probability

    Young = 60%
    
    Old = 40%

Much more realistic.

This is called Soft Clustering.

---


Hard Clustering vs Soft Clustering
K-Means
Person

↓

Cluster 1

Only one cluster.

GMM
Person

↓

Cluster 1 = 70%

Cluster 2 = 30%

Belongs partly to multiple clusters.

