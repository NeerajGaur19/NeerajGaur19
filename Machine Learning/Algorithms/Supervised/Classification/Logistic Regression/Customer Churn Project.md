https://colab.research.google.com/drive/1IjoHoNczJLGoAaCdj9zyOpNXWbf1UBOg?usp=sharing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/customer_churn_.csv')

df.head()

df.shape

df.info()

df.isnull().sum().sum()

df.duplicated().sum()

df['Churn'].value_counts()

#Check Outliers
for col in df.columns:
  if df[col].dtype=='int64' or df[col].dtype=='float64':
    sns.boxplot(df[col])
    plt.xlabel(col)
    plt.title("Outliers")
    plt.show()

Only 2 columns were of Int64 and one column of Float64 data type, outliers are very less




sns.countplot(x='Churn', data=df)
plt.title("Distribution")
plt.show()

sns.heatmap(df.select_dtypes(include=['int64','float64']).corr(), annot=True)
plt.show()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.info()

# Dropping CustomerID column as used for identification only, not useful for modeling.

df.drop(['customerID'], axis=1, inplace=True)

df.head()

df.isnull().sum().sum()

df.dropna(inplace=True)

df.isnull().sum().sum()

#Label encoding is required as many columns are of object type.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.columns:
  if df[col].dtype=='object':
    df[col]=le.fit_transform(df[col])


df.info()

df.head()

df.shape

X = df.drop('Churn', axis=1)
y = df['Churn']

X

y

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=50)

X_train.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

final_model = LogisticRegression()

final_model.fit(X_train,y_train)

y_pred = final_model.predict(X_test)

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score
print("accuracy score is: ", accuracy_score(y_pred,y_test)*100)

###So, accuracy score is 81%, means out of 100 predictions 81 are correct.

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

### So TRUE negative is 941, it means correctly predicted non-churn customers  

as 941.  whereas 206 is True positive, it means correctly predicted customers who will churn as 206

print(classification_report(y_test, y_pred))

### Recall value 0.59 - Out of all customers churned, model was able to identify 59% as per above classification report

###Precision value 0.64 - 64% churn prediction are correct.


