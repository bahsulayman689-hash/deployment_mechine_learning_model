import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
breast_cancer_data = pd.read_csv("breast-cancer.csv")
print(breast_cancer_data.shape)
breast_cancer_data.info()
print(breast_cancer_data.head())
print(breast_cancer_data.tail())
print(breast_cancer_data.isnull().sum())
print(breast_cancer_data.duplicated().sum())

breast_cancer_data["diagnosis"] = breast_cancer_data["diagnosis"].map({"B":1, "M":0})
print(breast_cancer_data["diagnosis"].value_counts())

X = breast_cancer_data.drop(['id', 'diagnosis'], axis=1)
print(X)
y = breast_cancer_data["diagnosis"]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
print(X_train_scaler)
print(X_test_scaler)

model = LogisticRegression(max_iter=100,
                           random_state=42,
                           class_weight='balanced')
model.fit(X_train_scaler, y_train)
Y_Pred = model.predict(X_test_scaler)
Y_testing_Accuracy = accuracy_score(y_test, Y_Pred)
class_report = classification_report(y_test, Y_Pred)
confusion_me = confusion_matrix(y_test, Y_Pred)
print(f"Accuracy of the model is {Y_testing_Accuracy}")
print(class_report)
print(confusion_matrix)
plt.figure(figsize=(10, 9))
sns.set_style('whitegrid')
sns.heatmap(confusion_me, fmt='d', annot=True, annot_kws={"size":8}, cbar=True, cbar_kws={"label":"prediction"}, cmap='viridis')
plt.show()

###0---->M
###1---->B
input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)

input_data_Array = np.asarray(input_data)

input_data_reshape = input_data_Array.reshape(1, -1)

input_data_reshape_scaler = scaler.transform(input_data_reshape)

prediction = model.predict(input_data_reshape_scaler)
print(prediction)

if prediction[0] == 1:
    print("The person has  a breast cancer")
else:
    print("The person has no breast cancer")

joblib.dump(model, 'breast_cancer.pkl')
model = joblib.load('breast_cancer.pkl')
joblib.dump(scaler, 'scaler_breast.pkl')
scaler = joblib.load("scaler_breast.pkl")
