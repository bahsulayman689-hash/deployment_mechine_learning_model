import random
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math
import pickle 
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings as wr
wr.filterwarnings("ignore")


liver_disease_data = pd.read_csv("Liver_disease_data.csv")
print(liver_disease_data.shape)
print(liver_disease_data.head())
print(liver_disease_data.info())
print(liver_disease_data.isnull().sum())
print(liver_disease_data.duplicated().sum())

print(liver_disease_data["Diagnosis"].value_counts())

#1--> the person has a liver disease
#0--> the person do not have liver disease

X = liver_disease_data.drop(columns=['Diagnosis'], axis=0)
print(X, X.shape)
y  = liver_disease_data["Diagnosis"]
print(y)
print(y.dtype)


X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
print(X.shape)
print(X_train.shape)
print(X_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=100,
    random_state=42,
    n_jobs=-1,
    verbose=0,
    solver='liblinear',
    class_weight='balanced'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
Y_test_accuracy = accuracy_score(y_test, y_pred)
score = cross_val_score(
    model,
    X,
    y, 
    cv=3,
    n_jobs=-1
)
predict = cross_val_predict(model,
                            X,
                            y,
                            cv=3,
                            method='predict')
print(f"Accuracy of the model is {Y_test_accuracy}%")
print(f"classification_report  of the model {classification_report(y_test, y_pred)}")
print(f"the confusion_metrices of the model {confusion_matrix(y_test, y_pred)}")
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.set_style("whitegrid")
sns.heatmap(cm, annot=True, annot_kws={"size": 8}, cbar=True, cbar_kws={"label": "prediction"}, fmt='d', cmap='viridis')
plt.show()
print(score)
print(np.argmax(score))
print(predict)
print(np.argmax(predict))
print('='*70, '\n')
input_data = (48,0,19.971406944382398,18.50094350686029,0,0,9.92830825444697,0,0,63.738955838709444)
input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1, -1)
input_data_reshape_scaled = scaler.transform(input_data_reshape)
prediction = model.predict(input_data_reshape_scaled)
print(prediction)
if prediction[0] == 1:
    print("the person has liver disease")
else:
    print("the person do not have a liver disease")
print('\n', '='*70)
joblib.dump(model, "model_logistic.pkl")
model = joblib.load("model_logistic.pkl")
joblib.dump(scaler, "scaler_liver.pkl")
scaler = joblib.load("scaler_liver.pkl")
print(scaler.feature_names_in_)

print(liver_disease_data.columns)
