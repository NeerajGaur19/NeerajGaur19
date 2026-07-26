#dataset drive link: https://drive.google.com/file/d/1CnE8T3t7CsZi9HXe5hn_tTT_mrKvQJnp/view?usp=sharing

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading dataset
df = pd.read_csv('/content/census-income .csv')

#Rows and columns
df.shape

df.info()

#Checking first 5 records to understand data and columns
df.head()

df['annual_income'].value_counts()

###Checking unique values in columns

df['annual_income'].unique()

df['workclass'].unique()

df['education'].unique()

df['marital-status'].unique()

df['occupation'].unique()

df['relationship'].unique()

df['race'].unique()

df['sex'].unique()

sns.countplot(x='annual_income', data=df)
plt.title("Distribution")
plt.show()

sns.boxplot(x='annual_income', y='age', data=df)
plt.title("AGE VS INCOME")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation")
plt.show()

#As there are some characters such as ? in multiple column so replacing them with NaN and then dropping NaN
df.replace('?',pd.NA,inplace=True)

df.isna().sum().sum()

df = df.dropna()

df.isna().sum().sum()

df.duplicated().sum()

df.shape

###Compared to 30,162 rows, only 23 are duplicates so not dropping.



#Checking outliers.
for col in df.columns:
  if df[col].dtype != 'object':
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()

sns.heatmap(df.select_dtypes(include=['int64','float64']).corr(), annot=True)
plt.show()

Not dropping outliers as these provide important information on data.

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


#Encoding categorical columns
le = LabelEncoder()

for col in df.columns:
  if df[col].dtype=='object':
    df[col]=le.fit_transform(df[col])


df.info()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor

df.shape

X = df.iloc[:,:-1]
y = df['annual_income']

X.head()

y.head()

#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

vif_df = pd.DataFrame()
vif_df['Feature'] = X.columns

vif_df['VIF']=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]

print(vif_df)

X.drop(['native-country'], axis=1,inplace=True)


X.head()

vif_df = pd.DataFrame()
vif_df['Feature'] = X.columns

vif_df['VIF']=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]

print(vif_df)

X.drop(['education-num'], axis=1,inplace=True)




X.head()

vif_df = pd.DataFrame()
vif_df['Feature'] = X.columns
vif_df['VIF']=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]

print(vif_df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=100)


final_model = LogisticRegression()

final_model.fit(X_train,y_train)

y_pred = final_model.predict(X_test)

print("accuracy score is: ", accuracy_score(y_pred,y_test)*100)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred))

