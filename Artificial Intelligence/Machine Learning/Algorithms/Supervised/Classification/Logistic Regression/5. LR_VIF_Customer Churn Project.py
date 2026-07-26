#Problem:
**predicting whether a customer will churn (leave the service) or not based on their characteristics and service usage patterns.**


**customerID:**	Unique ID assigned to each customer (used for identification only, not useful for modeling).

**gender:** Gender of the customer – Male or Female. May or may not influence churn.

**SeniorCitizen:**	Whether the customer is a senior citizen (0 = No, 1 = Yes). Binary numerical feature.

**Partner:**	Whether the customer has a partner (Yes/No).

**Dependents:**	Whether the customer has dependents (children, family) (Yes/No).

**tenure:**	Number of months the customer has stayed with the company. Longer tenure may mean higher loyalty.

**PhoneService:**	Whether the customer has a phone service (Yes/No).

**MultipleLines:**	Whether the customer has multiple phone lines – values like Yes / No / No phone service.

**InternetService:**	Type of internet service – options: DSL, Fiber optic, No.

**OnlineSecurity:**	Whether the customer has online security add-on – Yes / No / No internet service.

**OnlineBackup:**	Whether the customer has online backup service – Yes / No / No internet service.

**DeviceProtection:**	Whether the customer has device protection – Yes / No / No internet service.

**TechSupport:**	Whether the customer has technical support service – Yes / No / No internet service.

**StreamingTV:**	Whether the customer streams TV – Yes / No / No internet service.

**StreamingMovies:**	Whether the customer streams movies – Yes / No / No internet service.

**Contract:**	Type of contract: Month-to-month, One year, or Two year. This is an important feature—monthly contracts often churn more.

**PaperlessBilling:**	Whether the customer uses paperless billing – Yes or No.

**PaymentMethod:**	How the customer pays: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic).

**MonthlyCharges:**	The amount charged to the customer monthly. Higher charges may influence churn.

**TotalCharges:**	The total amount charged to the customer over their tenure.

**Churn (Target):**	Whether the customer churned (Yes = left, No = stayed). This is what you're predicting using logistic regression.


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/customer_churn_.csv')
df

df.info()

#EDA
#1. null

df.isnull().sum()

#2. duplicates

df.duplicated().sum()

#3. outliers

df.columns

for col in df.columns:
  if df[col].dtype != 'object':
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()

#if no box is present in the boxplot = no range of values available(majority values are extremes/outliers)
#so dropping outliers from these columns might drop the whole column itself

df.info()

#convert total charges from object to numerical
#errors=coerce: whenever you face an unconvertable value, we replace it with null

df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')

df.info()

df.isnull().sum().sum()

df.dropna(inplace=True)

df.isnull().sum().sum()

#dropping CustomerID since its not relevant according to the problem statement

df.drop(['customerID'],axis=1,inplace=True)

df

#4. Label Encoding
#Label Encoding converts the categories(of object column) into numeric values(where the numerical values are just a representation of tha category)

from sklearn.preprocessing import LabelEncoder

#create an instance
LE=LabelEncoder()

for col in df.columns:
  if df[col].dtype == 'object':
    df[col]=LE.fit_transform(df[col])

#fit: converts the category into number alphabetically
#transform: transform the whole column data according to the number given

df

#Model Building
#1. splitting the data into x and y

x=df.iloc[:,:-1] #feature/independent: all rows, all columns except last
y=df['Churn'] #target/dependent

#VIF : Variance Inflation Factor is a feature selection technique
# calculates the multicollinearity(extent of relationship) for a column wrt the whole data
# Acceptable VIF for a column is under 5.

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_df=pd.DataFrame()

#storing all the feature column names in the "Features" column
vif_df['Features']=x.columns
vif_df

x.columns

#for i in range(len(x.columns)) : iterating through each column in x
#variance_inflation_factor(x.values,i) : calculates VIF value for the current column wrt to the rest of the columns

vif_df['Multicollinearity']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]

vif_df

x.drop('MonthlyCharges', axis=1, inplace=True)

#DRAWBACK
#this column is important so to overcome dropping, we use techniques like PCA and LDA

vif_df=pd.DataFrame()
vif_df['Features']=x.columns
vif_df['Multicollinearity']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]

vif_df

x.drop('tenure', axis=1, inplace=True)

vif_df=pd.DataFrame()
vif_df['Features']=x.columns
vif_df['Multicollinearity']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]

vif_df

x.drop('PhoneService', axis=1, inplace=True)

vif_df=pd.DataFrame()
vif_df['Features']=x.columns
vif_df['Multicollinearity']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]

vif_df

#2. splitting the data into training and testing sets

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)

x_train #Images of dogs and humans

x_test #Images of dogs and humans(FOR TESTING)

y_train #ANSWER/label for dog and human images

y_test

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(x_train,y_train) #Images of dogs and humans along with the names

y_pred=model.predict(x_test)

from sklearn.metrics import *

accuracy_score(y_test,y_pred)*100

# confusion matrix is designed for classification problems where it compares the actuand and predicted labels
#True positive: Actual also positive, Pred also positive
#False position: Actual negative, Pred falsely as positive
#True Negative: Actual Negative, Pred Negative
#False negative: Actual Positive, Pred falsely as negative

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix\n", confusion_matrix(y_test,y_pred))

#fmt: format parameter that helps considering decimal values for data
sns.heatmap(cm,annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
