import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('sample_data_2.csv')
relevant_channels = [
    'Airflow', 'Nasal Pressure', 'SpO2', 'ECG1', 'ECG2',
    'Thor', 'Abdo', 'Snore', 'Pulse', 'Respiratory Rate',
    'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2',
    'Chin1', 'Chin2', 'Chin3'
]

# Separate features and labels
X = data[relevant_channels]
y = data['Label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract feature importances
feature_importances = rf.feature_importances_
features = np.array(relevant_channels)

# Sort features by importance
indices = np.argsort(feature_importances)[::-1]
sorted_features = features[indices]
sorted_importances = feature_importances[indices]

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(len(sorted_importances)), sorted_importances, align="center")
plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
plt.show()

# Select top features (e.g., top 10)
top_features = sorted_features[:10]

# Train a new model using only the top features
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

rf_top = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top.fit(X_train_top, y_train)

# Evaluate the model
y_pred = rf_top.predict(X_test_top)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)
