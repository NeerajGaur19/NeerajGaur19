### K-Nearest Neighbors (KNN) is one of the simplest and most intuitive Machine Learning algorithms. It is a supervised learning algorithm used for:

* Classification (predicting categories)
* Regression (predicting numerical values)

The idea is: "A new data point is likely to belong to the same class as its nearest neighbors."

### Real-Life Example

Suppose you have fruits with two features:

<img width="800" height="267" alt="image" src="https://github.com/user-attachments/assets/fb1bb1c2-1842-4834-8db1-3b87cd95e12b" />

A new fruit appears:

<img width="623" height="92" alt="image" src="https://github.com/user-attachments/assets/2d36cf7c-f546-4a84-a639-a14de9f87d7e" />

KNN looks at the nearest fruits and decides whether it is an Apple or Orange.

### How KNN Works

#### Step 1: Choose K

K = Number of neighbors to consider.

Example:

* K = 3 → Look at 3 nearest points
* K = 5 → Look at 5 nearest points

#### Step 2: Calculate Distance

Most commonly KNN uses Euclidean Distance.

For two points:

<img width="398" height="67" alt="image" src="https://github.com/user-attachments/assets/bb8ddb40-4b06-49c0-96d8-7e2f2d13b2be" />

Example:

* Point A = (2,3)
* Point B = (5,7)

Distance:

<img width="286" height="168" alt="image" src="https://github.com/user-attachments/assets/907b2475-603d-4c7b-8889-e49c0eabfb18" />

#### Step 3: Find Nearest Neighbors

Calculate distance from new point to every training point.

Sort distances.

Pick K smallest distances.

Example:

<img width="582" height="212" alt="image" src="https://github.com/user-attachments/assets/398a0280-3da7-4594-ae7d-f52f5662e4f0" />

For K=3:

* Apple = 2 votes
* Orange = 1 vote

Prediction = Apple

#### Step 4: Majority Voting

For classification:

Choose the class with maximum votes.

Example:

<img width="137" height="305" alt="image" src="https://github.com/user-attachments/assets/51b00828-9105-4db9-a668-9fd5cb9323cd" />

K = 5

Cat = 3 votes

Dog = 2 votes

Prediction = Cat

### KNN for Regression

Instead of voting, take the average.

Example:

Nearest house prices:

<img width="102" height="152" alt="image" src="https://github.com/user-attachments/assets/bb6c3dba-4b19-45d4-9952-d28ecfbd45c3" />

Prediction:

<img width="218" height="83" alt="image" src="https://github.com/user-attachments/assets/34d7638f-fa0d-4e41-93fb-d7aa31b16626" />

Predicted price = ₹55 lakh

### Why Scaling Is Important

KNN depends on distances.

Consider:

<img width="570" height="152" alt="image" src="https://github.com/user-attachments/assets/f09a5428-6683-45ab-8576-18a658990a8f" />

Salary dominates distance calculation.

Therefore always scale data using:

    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

 This is extremely important for KNN.


### Choosing Best K

#### Very Small K

K = 1

* Sensitive to noise
* Overfitting

Example:

One unusual point can change prediction.

#### Very Large K

K = 50

* Underfitting
* Ignores local patterns

#### Common Values

    K = 3
    K = 5
    K = 7
    K = 9

Usually select K using cross-validation.




