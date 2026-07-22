Gradient Boosting is an ensemble technique where: Many small weak models work together to gradually reduce errors.

It builds models:

one after another

each new model tries to fix previous mistakes

Very similar idea to AdaBoost, but smarter and more mathematical.

## Simple Intuition

Imagine a student learning mathematics.

Teacher 1

    Predicts score:70
    
    Actual score was: 90
    
    Error: +20

Teacher 2

    Now focuses only on correcting that error.
    
    Predicts correction: +15
    
    Updated prediction: 70 + 15 = 85

    Still error: +5

Teacher 3

Corrects remaining error:+4

    Final prediction: 89

Very close to actual value: 90

That is Gradient Boosting.

## Core Idea

Instead of training next model on original target:

Gradient Boosting trains next model on: Residual Errors

Meaning:

Residual=Actual−Predicted

Residual=y−y^​

