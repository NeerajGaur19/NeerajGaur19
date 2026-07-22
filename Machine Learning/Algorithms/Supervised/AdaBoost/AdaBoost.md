# AdaBoost

As we know Bagging is technique where we select the final result based on the Majority Voting. And bagging is happening in parallel as sub sets are created parallely and all DTs are working parallely.

<img width="923" height="429" alt="image" src="https://github.com/user-attachments/assets/a29f9bc6-7689-4f70-87e7-ad8bf972d9a3" />

Boosting is a machine learning technique that combines multiple weak learners to form a strong learner.

AdaBoost is one of the first and most famous Boosting Algorithms in Machine Learning.
Its full form is: Adaptive Boosting

### AdaBoost combines many weak learners to create one strong learner.

Think:

Weak models + Weak models + Weak models -----------------> Strong Model

## What is a Weak Learner?

A weak learner is a model that performs only slightly better than random guessing.

In AdaBoost, the weak learner is usually:

    Decision Stump

A decision stump is a very small Decision Tree with:

    depth = 1
Meaning:

    only one split
    very simple model

Real-Life Analogy

Imagine a teacher taking help from many students.

    First student answers some questions
    Teacher notices mistakes
    Next student focuses more on difficult questions
    Again mistakes are checked
    Next student focuses even more on wrongly answered questions

Finally: All students together produce a very accurate result.
That is AdaBoost.

## Why “Adaptive”?

Because it adapts by giving more attention to wrongly classified data points.

## Core Idea

AdaBoost works in rounds.

Each new model:

* focuses more on previous mistakes
* tries to correct them

# Step-by-Step Working of AdaBoost

Suppose we have this dataset:

<img width="789" height="265" alt="image" src="https://github.com/user-attachments/assets/b9c72e0f-5ca5-4214-bc05-e05427ae611e" />


## STEP 1 — Assign Equal Weights

Initially every row gets equal importance.

A → 0.25

B → 0.25

C → 0.25

D → 0.25

## STEP 2 — Train First Weak Learner

AdaBoost creates first decision stump.

Example rule:

If Hours > 2.5 → Pass

Else → Fail

Suppose it predicts:

<img width="836" height="253" alt="image" src="https://github.com/user-attachments/assets/9dc152e9-dac2-433e-b6d8-99583fd1708d" />







