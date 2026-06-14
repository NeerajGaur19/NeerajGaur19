###Dataset drive link : https://drive.google.com/file/d/16cg5Ksa8pG6EhYfOrbKz9j_01SEjzpYT/view

####TASKS TO BE PERFORMED:
#### 1. Exploration
#### 2. EDA
#### 3. explore the fraud and geniune transactions
#### 4. Scale the data
#### 5. Split the data till training and testing
####DEADLINE: 08:00 AM

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading dataset
df = pd.read_csv('/content/creditcard.csv')

#Rows and columns
df.shape

df.info()

df.head()

df['Class'].value_counts()

fraud = df[df['Class'] == 1]
NotFraud = df[df['Class'] == 0]

print("Fraud",len(fraud))
print("Not Fraud",len(NotFraud))

df.describe()

df['Class'].unique()

sns.countplot(x='Class', data=df)
plt.title("Distribution")
plt.show()

###Fraud are very less compared to people who are not Fraud

sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Class VS Amount")
plt.show()

plt.figure(figsize=(20,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm',linewidths=0.2)
plt.title("Correlation Matrix")
#plt.tight_layout()
plt.show()



df.isna().sum().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.shape

#Checking outliers.
for col in df.columns:
  if df[col].dtype != 'object':
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score

scaler = StandardScaler()

df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

df = df.drop(['Amount', 'Time'], axis=1)

df.head()



X = df.iloc[:,:-1]
y = df['Class']

X.head()

y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)


final_model = DecisionTreeClassifier(max_depth=8, class_weight='balanced' )

final_model.fit(X_train,y_train)

y_predict = final_model.predict(X_test)

print("accuracy score with decision tree",accuracy_score(y_predict,y_test)*100)

print(classification_report(y_test,y_predict))

print(precision_score(y_predict,y_test)*100)

from sklearn.ensemble import RandomForestClassifier

class1 = RandomForestClassifier(n_estimators=100)

class1.fit(X_train,y_train)

y_pred1 = class1.predict(X_test)

print("accuracy score with random forest",accuracy_score(y_pred1,y_test)*100)





