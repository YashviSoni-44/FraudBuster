import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Load the dataset
file_path = "new_transaction_csv.csv"
df = pd.read_csv(file_path)
df2=df.copy()

df.rename(columns={"customer_id": "sender_id", "merchant_id": "receiver_id"}, inplace=True)

# Drop unnecessary columns
df_cleaned = df.drop(columns=['transaction_id', 'account_id', 'ip_address', 'timestamp', 'previous_transactions_timestamp'])

# List of categorical columns to encode
categorical_columns = ['merchant_category', 'transaction_type', 'location', 'channel', 'sender_id', 'receiver_id', 'customer_occupation', 'account_type']

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# Save the transformed dataset
df_cleaned.to_csv("transformed_transactions.csv", index=False)

data = pd.read_csv("transformed_transactions.csv")

# Define features and target
X = data.drop(columns=["fraud"])
y = data["fraud"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# Train model
rf_model = XGBClassifier(random_state=42)
rf_model.fit(x_train_resampled, y_train_resampled)

y_pred = rf_model.predict(x_test_scaled)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Take user input and test
print("Enter transaction details:")
user_input = {}
for col in X.columns:
    value = input(f"{col}: ")
    user_input[col] = value  # Convert to float

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    user_df[col] = le.fit_transform(user_df[col])
    label_encoders[col] = le

user_df_scaled = scaler.transform(user_df)

# Predict fraud
user_prediction = rf_model.predict(user_df_scaled)
print("Fraud Prediction:", "Fraudulent" if user_prediction[0] == 1 else "Not Fraudulent")