
# LDA (Linear Discriminant Analysis) 

is a supervised dimensionality reduction technique.

It reduces the number of features while keeping different classes as far apart as possible.

Unlike PCA, LDA uses the target variable (y).


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

