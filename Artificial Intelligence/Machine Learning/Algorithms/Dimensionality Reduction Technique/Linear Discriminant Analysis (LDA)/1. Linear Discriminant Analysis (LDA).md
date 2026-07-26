
# LDA (Linear Discriminant Analysis) 

is a supervised dimensionality reduction technique.

It reduces the number of features while keeping different classes as far apart as possible. To solve the problem of inseparable classification data, we have to use LDA. LDA will project the data onto a new axis to make the data easily separable and while doing so it will automatically reduce the doimension of data.

Unlike PCA, LDA uses the target variable (y). Unlike PCA, LDA main focus is not on dimensionality reduction but on separating the data by maximizing the separation between the classes.

LDA will also make a new axis and project the data points on that new axis. The new axis that LDA will create should follow these two criterias

* Maximize the distance between the means of two classes / categories.
* Minimize the variation (spread) within each class.

## Fischer Discriminant Ratio
<img width="713" height="168" alt="image" src="https://github.com/user-attachments/assets/a3086d64-e491-4bcf-9f9b-7f77c5e848ca" />

<img width="656" height="290" alt="image" src="https://github.com/user-attachments/assets/828aa913-674c-4041-b95b-ee38fe94be92" />


# Imagine a dataset

Suppose you have:

<img width="787" height="245" alt="image" src="https://github.com/user-attachments/assets/4438ed68-832b-414e-8b19-bb3ff6d7d455" />

Features:
    
    Height
    Weight
    Age

Target:

    Gender

LDA knows which samples are Male and which are Female.

### Its goal is:  Find a new axis where males and females are separated as much as possible.

# Why do we need LDA?

Suppose you have 100 features.

Training becomes:

* slower
* more memory
* more chance of overfitting

So we reduce dimensions.

Instead of:

    100 Features
          ↓
    LDA
          ↓
    2 Features

we get

    Feature1
    Feature2

that still separate the classes well.

# How is LDA different from PCA?

<img width="672" height="280" alt="image" src="https://github.com/user-attachments/assets/2d1302af-60ba-48a9-8df4-af6681e21391" />

# Main idea

LDA wants

## Same class → close together

Example:

    Male
    
    M
    M
    M
    M

Small spread.

## Different classes → far apart

    Male
    
    M
    M
    M
    

    F
    F
    F
    Female

Large distance.

So LDA tries to

    Within-class distance
    ↓↓↓
    
    Between-class distance
    ↑↑↑

# Number of components in LDA

Suppose

Classes = 2

Maximum LDA components =

    Classes − 1
    
    2 − 1 = 1

Only one component.

---

Suppose

Classes = 4

Maximum components

    4 − 1 = 3

---

Suppose

Features = 100

Classes = 3

Maximum components

    min(features, classes−1)
    
    min(100,2)
    
    =2

Only 2 components.

# Advantages

✅ Reduces dimensions

✅ Improves classification

✅ Uses class labels

✅ Fast

✅ Easy to interpret

# Disadvantages

❌ Works mainly for classification

❌ Assumes classes are approximately linearly separable

❌ Assumes similar covariance across classes (an assumption of classical LDA)

❌ Maximum components are limited to number of classes − 1
