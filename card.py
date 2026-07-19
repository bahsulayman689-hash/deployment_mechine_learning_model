import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

credits_card_data = pd.read_csv("creditcard.csv")
print(credits_card_data.head())
print(credits_card_data.tail())
print(credits_card_data.shape)
credits_card_data.info()
print(credits_card_data.isnull().sum())

print(credits_card_data["Class"].value_counts())
#0--->normal
#1-->fradulent
#TODO:seper the data for analysis
legit = credits_card_data[credits_card_data.Class == 0]
fraud = credits_card_data[credits_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)

#statistic measurement
print(legit.Amount.describe())

print(fraud.Amount.describe())
#compare the values for both transactions
print(credits_card_data.groupby("Class").mean())
#deal with unbalance data
#undersampling
legit_sample = legit.sample(n=492)
new_data = pd.concat([legit_sample, fraud], axis=0)
print(new_data.head())
print(new_data["Class"].value_counts())
print(new_data.groupby("Class").mean())

scaler = StandardScaler()
# Fit and transform Time and Amount so they match the scale of V1-V28
credits_card_data['Time'] = scaler.fit_transform(credits_card_data['Time'].values.reshape(-1, 1))
credits_card_data['Amount'] = scaler.fit_transform(credits_card_data['Amount'].values.reshape(-1, 1))

# Re-extract separate datasets from the scaled master frame
legit = credits_card_data[credits_card_data.Class == 0]
fraud = credits_card_data[credits_card_data.Class == 1]

# Deal with unbalanced data via Under-sampling
legit_sample = legit.sample(n=492, random_state=42) # Added random_state for consistency
new_data = pd.concat([legit_sample, fraud], axis=0)
print(new_data.head())
print(new_data["Class"].value_counts())
print(new_data.groupby("Class").mean())
#TODO:split the data in feature and label
X = new_data.drop('Class', axis=1)
y = new_data["Class"]
print(X)
print(y)
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X.shape)
print(X_train.shape)
print(x_test.shape)

model = OneVsOneClassifier(LogisticRegression(max_iter=1000,
                           class_weight='balanced',
                           random_state=42,
                           solver='lbfgs'))
model.fit(X_train, y_train)
y_pred = model.predict(x_test)
Testing_Accuracy_of_model = accuracy_score(y_test, y_pred)
print(f"the accuracy of the model is {Testing_Accuracy_of_model*100:.4f}%")
print(f"the confusion metrix is {confusion_matrix(y_test, y_pred)}")
print(f"the classreport is {classification_report(y_test, y_pred)}")


y_pred = model.predict(X_train)
Train_Accuracy_of_model = accuracy_score(y_train, y_pred)
print(f"the accuracy of the model is {Train_Accuracy_of_model*100:.4f}%")
print(f"the confusion metrix is {confusion_matrix(y_train, y_pred)}")
print(f"the classreport is {classification_report(y_train, y_pred)}")

credit_card_input = ("0,1.19185711131486,0.26615071205963,0.16648011335321,0.448154078460911,0.0600176492822243,-0.0823608088155687,-0.0788029833323113,0.0851016549148104,-0.255425128109186,-0.166974414004614,1.61272666105479,1.06523531137287,0.48909501589608,-0.143772296441519,0.635558093258208,0.463917041022171,-0.114804663102346,-0.183361270123994,-0.145783041325259,-0.0690831352230203,-0.225775248033138,-0.638671952771851,0.101288021253234,-0.339846475529127,0.167170404418143,0.125894532368176,-0.00898309914322813,0.0147241691924927,2.69")

parsed_number = [float(x) for x in credit_card_input.split(",")]
raw_time = parsed_number[0]
pca_features = parsed_number[1:29] # Grabs all 28 anonymized V columns
raw_amount = parsed_number[29]
# =====================================================================
# 🛠️ THE CORE FIX: SCALE ONLY WHAT THE SCALER EXPECTS
# =====================================================================
# If your training script used a single scaler for Amount:
scaled_amount = scaler.transform(np.array([[raw_amount]]))[0][0]

# Note: If you didn't scale Time during training, leave raw_time as is.
# If you scaled Time with a different scaler, use that scaler here.
# For now, let's keep Time as raw if it wasn't scaled:
scaled_time = raw_time 

# 3. Rebuild the 30-feature layout to match your Logistic Regression model shape
# Shape: [Time, V1, V2, ..., V28, Amount]
final_features = [scaled_time] + pca_features + [scaled_amount]

# Convert to 2D numpy matrix layout with 1 row and 30 columns
model_ready_input = np.array(final_features).reshape(1, -1)



prediction = model.predict(model_ready_input)
print(prediction)
if prediction[0] == 0:
    print("🚨 The credit card is normal")
else:
    print("✅The credit is a fradulent is scammp")
#print(f"🚨 Fraud Confidence: {prediction_proba[1]:.2%}")
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "credit_scaler.pkl")
#print("\n✅ Deployment assets 'credit_model.pkl' and 'credit_scaler.pkl' exported successfully!")
model = joblib.load("credit_model.pkl")
scaler = joblib.load("credit_scaler.pkl")
print("\n✅ Deployment assets 'credit_model.pkl' and 'credit_scaler.pkl' exported successfully!")