<In Progress>

https://awibisono.github.io/2016/06/20/accelerated-gradient-descent.html

https://medium.com/@piyushkashyap045/understanding-nesterov-accelerated-gradient-nag-340c53d64597


accelerated gradient descent, nesterov gradient descent

#all libraries are imported here
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

url ="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/refs/heads/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df.head(2)

pd.set_option("display.max_columns",None) #load all columns as is

df.head(2)

df.info()

#never do this before you analuze the data
df.isnull().sum() # this is bad

df['gender'].value_counts()

df['Churn'] = df['Churn'].map({"No":0, "Yes":1})
df['gender'] = df['gender'].map({"Female":0, "Male":1})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')

df.isnull().sum() # this is bad

df.dropna(inplace=True)

df.isnull().sum()

df.head(2)

#can a technique feature selection RFE
df.columns

#rfe
numerical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

for col in numerical_features:
  if df[col].dtype == "object":
    df[col] = pd.factorize(df[col])[0]

numerical_features = [f for f in numerical_features if f in df.columns]

X = df[numerical_features]
y = df['Churn']

#data leakage -> split the data first
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, stratify=y)

rf = RandomForestClassifier(n_estimators=200)
rfe = RFE(estimator=rf, n_features_to_select=5) #6,7,8
rfe.fit(X_train,y_train) #use only train data

selected_features = [f for f,sel in zip(numerical_features,rfe.support_) if sel]
selected_features

# fname = ['sandra','natasha','shivam']
# lname = ['r','mishra','dubey']

# list(zip(fname,lname))

#another model - rfe output
rf.fit(X_train,y_train)
importances = pd.DataFrame({'feature':numerical_features,
                            'importance':rf.feature_importances_})
importances = importances.sort_values('importance',ascending=False)
importances

import seaborn as sns

sns.barplot(x='importance',y='feature',data=importances)

selected_features

from sklearn.preprocessing import StandardScaler
#model building
X = df[selected_features]
y = df['Churn']

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import pickle

#save it , scaler

with open('scaler.pkl', 'wb') as f:
  pickle.dump(scaler,f)

import tensorflow as tf
from tensorflow.keras import layers, models

#based on my data what is feature? 5 input layer 5 nodes?

#you create your first ever neural network
def create_model():
  model = models.Sequential([
      layers.Dense(64, activation='relu',input_shape = (5,)),
      layers.Dense(32, activation='relu'),
      layers.Dense(16,activation='relu'),
      layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam',loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

ann_model = create_model()
ann_model.summary()

history = ann_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.grid(alpha=0.4)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.grid(alpha=0.4)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

#bayesian optimization -> rfe, rf, tain, nwu - production nn - toy program

!pip install keras-tuner

import keras_tuner as kt

#number of layers?
#number of nodes?
# activation function?
#regularization? - nn do not overfit

# ip - 1,2,3,4 - op
16,32,48, 64, 80, 96, ------ 256

def build_model(hp):
  model = models.Sequential()

  n_layers = hp.Int('n_layers',min_value=1,max_value=4, step=1)

  for i in range(n_layers):
    model.add(layers.Dense(
        units = hp.Int(f"units_{i}", min_value=16, max_value=256, step=16),
        activation = hp.Choice(f"activation_{i}", values=['relu','tanh','elu']),
        kernel_regularizer = tf.keras.regularizers.l2(
            hp.Float(f"lr_{i}", min_value=1e-5, max_value=1e-2, sampling='log')
        )
    ))
    model.add(layers.Dropout(
        rate = hp.Float(f"dropout_{i}",min_value=0.0, max_value=0.5, step=0.1)
    ))

  model.add(layers.Dense(1, activation='sigmoid'))

  #laerning rate and oiptimser
  learning_rate = hp.Float('learning_rate',min_value=1e-4, max_value=1e-2,sampling='log')
  optimizer = hp.Choice('optimizer',values=['adam','rmsprop','sgd'])

  if optimizer == 'adam':
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer == "rmsprop":
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9)

  model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
  return model

# grid search, random search



tuner = kt.BayesianOptimization(
    hypermodel = build_model,
    objective='val_accuracy',
    max_trials = 20,
    num_initial_points=5, #random
    seed = 10,
    directory = 'keras_tuner_logs',
    project_name = "misson_ai_sekho"
)

tuner.search_space_summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

tuner.search(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split = 0.20,
    callbacks=[stop_early]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

for i in range(best_hps.get('n_layers')):
  print(f"  Layer {i+1} units   : {best_hps.get(f'units_{i}')}")
  print(f"  Layer {i+1} activation   : {best_hps.get(f'activation_{i}')}")
  print(f"  Layer {i+1} dropout  : {best_hps.get(f'dropout_{i}')}")
  print(f"  Layer {i+1} L2  : {best_hps.get(f'lr_{i}')}")
print(f"  Optimizer  : {best_hps.get(f'optimizer')}")
print(f"  Learning Rate  : {best_hps.get(f'learning_rate')}")

best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    X_test_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split = 0.20,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],
)

loss, accuracy = best_model.evaluate(X_test_scaled, y_test)
print(f"loss : {loss}")
print(f"accuracy : {accuracy}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.grid(alpha=0.4)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (bayesian)")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.grid(alpha=0.4)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (bayesian)")

#save my model?
best_model.save("best_churn_model.h5") #weights and bias

with open('selected_feature.pkl','wb') as f:
  pickle.dump(selected_features,f)

#prediction?
def load_model_comp():
  model = tf.keras.models.load_model('/content/best_churn_model.h5')

  #load scaler
  with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

  with open('selected_feature.pkl','rb') as f:
    selected_features = pickle.load(f)

  return model, scaler, selected_features

X[selected_features].head(3)

def pred():
  model, scaler, selected_feature = load_model_comp()

  sample_customer = pd.DataFrame([[34, 1, 0, 56.78, 2300]],columns=selected_features)
  sample_scaled = scaler.transform(sample_customer)
  prediction = model.predict(sample_scaled)[0][0]
  print(f"Churn prob:{prediction}")
  print(f"Will churn :{'yes' if prediction> 0.5 else 'no'}")

pred()

