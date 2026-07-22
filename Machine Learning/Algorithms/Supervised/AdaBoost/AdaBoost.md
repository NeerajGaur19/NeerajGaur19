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

Perfect prediction.

But usually some mistakes happen.

Let’s assume:

<img width="825" height="90" alt="image" src="https://github.com/user-attachments/assets/4bf04ccb-6415-4b22-bf88-1e7c1a8ce6a0" />

## STEP 3 — Calculate Error

Error = sum of weights of wrong predictions.

Suppose:

* Only B wrong
* Weight of B = 0.25

Then:

* Error=0.25

### Why here error = sum of weights of wrong predictions??

In normal machine learning: Every row has equal importance.

Error = Number of wrong predictions / Total predictions

Example:

Total : 10

Wrong : 2

Error:

* Error = 2/10 = 0.2

Every row has equal importance.

## But AdaBoost Thinks Differently

### AdaBoost says:

“Some data points are more important than others.”

So instead of counting mistakes equally, AdaBoost uses weights.

## STEP 4 — Calculate Model Performance

AdaBoost calculates importance of this stump.

Formula:

α = 1/2 ln ((1-Error)/Error)

Note: Here ln is natural logarithm and base of logarithm is e ≈ 2.718

Meaning: ln(1.5) asks: “e raised to what power gives 1.5?” 
Answer: <img width="126" height="46" alt="image" src="https://github.com/user-attachments/assets/699b79d9-c403-45e8-92eb-0df1a299fd41" />

That is why: ln(1.5)≈0.405

Where:

    * smaller error → higher alpha
    * higher alpha → model more important

If error is small:

    * alpha becomes large

Meaning:

    * This model is trustworthy

## STEP 5 — Increase Weights of Wrong Predictions

Wrongly classified rows get higher weight.

Example:

Before:

    B → 0.25

After:

    B → 0.50

Correct rows get lower weights.

Why?

    Because next model should focus more on difficult cases.

