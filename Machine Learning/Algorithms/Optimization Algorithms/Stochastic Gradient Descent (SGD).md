
Stochastic Gradient Descent (SGD) is an optimization algorithm used to train machine learning and deep learning models. It minimizes the loss (error) by updating the model's parameters (weights and bias) in the direction that reduces the error.

SGD is the most widely used optimizer for training Neural Networks, Logistic Regression, Linear Regression, and many Deep Learning models.


Why Do We Need Gradient Descent?

Suppose we want to predict house prices.

Our model is

y=wx+b

Initially,

    Weight (w) = Random
    
    Bias (b) = Random

The predictions are poor.

We need a way to adjust w and b so that predictions become more accurate.

This is what Gradient Descent does.

## Think of Climbing Down a Mountain

Imagine standing on top of a mountain.

                *
               / \
              /   \
             /     \
            /       \
    _______/_________\_____
    
Your goal is to reach the lowest point (the valley).

The mountain represents the loss function.
    
    Top = High Error
    Bottom = Minimum Error

Gradient Descent finds the lowest point.


