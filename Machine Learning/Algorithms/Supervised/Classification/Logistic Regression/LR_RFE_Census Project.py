#dataset drive link: https://drive.google.com/file/d/1CnE8T3t7CsZi9HXe5hn_tTT_mrKvQJnp/view?usp=sharing

#Problem:
- The dataset is a demographic census dataset.
- It contains information about individuals, including their age, workclass,education, etc
- **The goal is to predict whether the person earns more than $50K or not per year.**
- **The dataset has "?" values present as well, deal with them and apply Logistic Regression and VIF technique.**

#You have to perform a task today, i've shared the colab file with the dataset
#You have to apply logistic regression on it. Everything is mentioned on the screen.
#DEADLINE: 8:15 AM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/census-income .csv')

df

df.info()

df['annual_income'].value_counts()

df['workclass'].value_counts()

df.isnull().sum()

df=df.replace("?",np.nan)

df.isnull().sum().sum()

df.dropna(inplace=True)

df.isnull().sum().sum()

#duplicates

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

#outliers

for col in df.columns:
  if df[col].dtype != "object":
    plt.boxplot(df[col])
    plt.xlabel(col)
    plt.show()

#Columns not to drop:
#1. columns without a box, since it means no range of data is there(so most values are extreme)
#dropping outliers from such a column might drop the whole column itself
#2. target column (since its dependent on input), no outliers in input means nothing in the output

out_list=['age','fnlwgt','education-num','hours-per-week']

for col in out_list:
  Q1=df[col].quantile(0.25)
  Q3=df[col].quantile(0.75)

  IQR= Q3-Q1

  LB= Q1-1.5*(IQR)
  UB= Q3+1.5*(IQR)

  df=df[(df[col]>=LB) & (df[col]<=UB)]

#Label Encoding

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in df.columns:
  if df[col].dtype == "object":
    df[col]=le.fit_transform(df[col])

df

#After label encoding if we are applying VIF, its done before sending the data to the model

#MODEL BUILDING
#1. splitting the data into x and y axis
#2. splitting the data into train and test sets
#3. training the model
#4. prediction
#5. evaluation

x=df.iloc[:,:-1]
y=df['annual_income']

x

y

#RFE: Recursive Feature Elimination
#feature selection technique
#used to identify the most relevant features according the pattern

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Scaling helps Logistic Regression perform better since it uses distance based optimization
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train) #fit on training data and transform it(mean=0, std=1)
x_test=scaler.transform(x_test) #transform test data using same scaling parameters

log_reg=LogisticRegression()

#How RFE works:
#1. trains a model using all features in the dataset
#2. rank the features on the basis of importance(calculated using model coefficients(for linear models) or feature importance scores(for tree based models))
#3. eliminates least important features
#4. repeats until desired number of features selected is reached
#5. final model

#estimator=logistic regression is the base model
#n_features_to_select= select top 10 best features(on the basis of best pattern)
rfe=RFE(estimator=log_reg, n_features_to_select=10)
rfe.fit(x_train,y_train)

#rfe.support_ gives a boolean mask of selected (True=selected)
selected_features=x.columns[rfe.support_]
print("Top 10 selected features using RFE: ")
print(selected_features)

log_reg.fit(x_train[:, rfe.support_], y_train) #x_train: all rows, 10 cols

score=log_reg.score(x_test[:,rfe.support_],y_test) #x_test: all rows, 10 cols
print("Model Accuracy after RFE: ", score)

#PCA and LDA dimensionality reduction

