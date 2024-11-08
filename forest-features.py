import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import ast

# Define relevant channels
relevant_channels = [
    'Airflow', 'Nasal Pressure', 'SpO2', 'ECG1', 'ECG2',
    'Thor', 'Abdo', 'Snore', 'Pulse', 'Respiratory Rate',
    'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2',
    'Chin1', 'Chin2', 'Chin3'
]

# Load and preprocess data with padding/truncation
def load_data(csv_file, sequence_length):
    df = pd.read_csv(csv_file)

    # Separate features and labels
    labels = df['Label'].values
    df = df.drop('Label', axis=1)
    
    # Convert string representations of lists into arrays and apply padding/truncation
    features = []
    for _, row in df.iterrows():
        event_data = []
        for col in df.columns:
            try:
                # Handle 'nan' values and malformed strings
                if pd.isna(row[col]):
                    channel_data = np.array([])
                else:
                    channel_data = np.array(ast.literal_eval(row[col]))
            except (ValueError, SyntaxError):
                channel_data = np.array([])

            # Pad or truncate to the fixed sequence length
            if len(channel_data) < sequence_length:
                # Pad with zeros if the sequence is shorter than desired length
                padded_data = np.pad(channel_data, (0, sequence_length - len(channel_data)), mode='constant')
            else:
                # Truncate if the sequence is longer than desired length
                padded_data = channel_data[:sequence_length]

            event_data.append(padded_data)

        # Stack channels to form (sequence_length, channels)
        features.append(np.stack(event_data, axis=-1))

    features = np.array(features)
    return features, labels

# Load the data
csv_file = 'sample_data_2.csv'
sequence_length = 60  # Adjust this based on your data's typical length
X, y = load_data(csv_file, sequence_length)

# Flatten the 3D array to 2D array for RandomForestClassifier
X_flattened = X.reshape(X.shape[0], -1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_encoded, test_size=0.30, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract feature importances
feature_importances = rf.feature_importances_

# Since we flattened the features, we need to create a list of feature names that match the flattened structure
flattened_feature_names = [f"{channel}_{i}" for channel in relevant_channels for i in range(sequence_length)]

# Aggregate importances by channel
channel_importances = {channel: 0 for channel in relevant_channels}
for feature, importance in zip(flattened_feature_names, feature_importances):
    channel = feature.split('_')[0]
    channel_importances[channel] += importance

# Sort channels by importance
sorted_channels = sorted(channel_importances.items(), key=lambda item: item[1], reverse=True)



# Plot channel importances
channels, importances = zip(*sorted_channels)
plt.figure(figsize=(12, 6))
plt.title("Channel Importances")
plt.bar(range(len(importances)), importances, align="center")
plt.xticks(range(len(importances)), channels, rotation=90)
plt.show()

# Select all channels
top_channels = [channel for channel, _ in sorted_channels]
top_features_indices = [i for i, feature in enumerate(flattened_feature_names) if feature.split('_')[0] in top_channels]
X_train_top = X_train[:, top_features_indices]
X_test_top = X_test[:, top_features_indices]

# Train a new model using only the top channels
rf_top = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top.fit(X_train_top, y_train)

# Evaluate the model
y_pred = rf_top.predict(X_test_top)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)
print(f'Test Accuracy: {rf_top.score(X_test_top, y_test):.4f}')
