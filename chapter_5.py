import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import warnings as wr
wr.filterwarnings("ignore")

raw_mail_data = pd.read_csv("mail_data.csv")
print(raw_mail_data.head())
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
print(mail_data.head())
print(mail_data.shape)
#TODO:label encoder of the categorical columns
#o--> spam
#1--> ham
mail_data["Category"] = mail_data['Category'].map({"spam":0, "ham": 1})
print(mail_data.head())

#seperating the text and label

X = mail_data["Message"]

Y = mail_data["Category"]

print(X)

print(Y)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
#Convert the numerical value for the computer to understand concept
#feature extraction
feature_extraction = TfidfVectorizer(min_df=1, 
                                     stop_words='english',
                                     lowercase=True,
                                     ngram_range=(1, 2))
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert the y_train and y_Test

y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(X_train_features)

model = LogisticRegression(random_state=42,
                           class_weight='balanced')

model.fit(X_train_features, y_train)

y_pred = model.predict(X_train_features)
accuracy_of_training = accuracy_score(y_train, y_pred)
print(f"the accuracy of the model is {accuracy_of_training}")

y_pred_test = model.predict(X_test_features)
accuracy_of_test = accuracy_score(y_test, y_pred_test)
print(f"the accuracy of the model is {accuracy_of_test}")

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(9, 4))
sns.heatmap(
    cm,
    fmt='d',
    annot=True,
    cbar=False,
    cmap='Set1',
    xticklabels=["ham", "spam"],
    yticklabels=["ham", "spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Detection Confusion Matrix")
plt.show()

cm = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(9, 4))
sns.heatmap(
    cm,
    fmt='d',
    annot=True,
    cbar=False,
    cmap='Set2',
    xticklabels=["spam", "ham"],
    yticklabels=["spam", "ham"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Detection Confusion Matrix")
plt.show()

input_mail =  ("URGENT: Your Account Has Been Locked. Dear Customer, we noticed unusual activity on your account. Click the secure link below to verify your login credentials immediately.")

input_mail_features = feature_extraction.transform([input_mail])

prediction = model.predict(input_mail_features)

print(prediction)

if (prediction[0] == 1):
    print("the message you recieve is ham mail")
else:
    print("the mesaage you recieve is spam mail")

joblib.dump(model, "model_mail.pkl")
joblib.dump(feature_extraction, "feature_mail.pkl")

model = joblib.load("model_mail.pkl")
feature_extraction = joblib.load("feature_mail.pkl")