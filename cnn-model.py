import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras import backend as K
import sys
import ast
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

# Clear any previous session
K.clear_session()

# Define a fixed sequence length for padding/truncating
sequence_length = 60  # Adjust this based on your data's typical length

# Load and preprocess data with padding/truncation
def load_data(csv_file, sequence_length, selected_features):
    df = pd.read_csv(csv_file)

    # Filter the dataframe to include only the selected features
    df = df[selected_features + ['Label']]

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

# Define the 1D CNN model with reduced complexity
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Function to train and evaluate the model with given features
def train_and_evaluate(csv_file, sequence_length, selected_features):
    X, y = load_data(csv_file, sequence_length, selected_features)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.33, random_state=42)

    # Reshape the data for the 1D CNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Create and compile the model
    model = create_model((X_train.shape[1], X_train.shape[2]), y_categorical.shape[1])

    # Add early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

    # Train the model with callbacks
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    return accuracy

# User selects the features they want to analyze from the CSV file
selected_features = ['HRate', 'HR Sentec', 'Mic', 'Derived HR', 'NASAL KANUL', 'Sum', 'Pressure', 'C4', 'M2', 'Chin3', 'SENTEC CO2', 'Ox Status', 'E1', 'E2', 'LLeg2', 'A1', 'SENTEC HR', 'Chin1', 'OxStatus', 'Thermistor', 'LEG/R', 'O1', 'Oksijen Sentec', 'SpO2', 'SENTEC O2', 'Channel 51', 'Pleth', 'ManPos', 'A2', 'ECG1', 'C3', 'Nasal Pressure', 'RLeg1', 'Manual Pos', 'EMG1', 'HR', 'PTT', 'EMG2', 'F3', 'Respiratory Rate', 'Karbondioksit', 'Termistor', 'Pulse', 'LLeg1', 'O2', 'Abdo', 'ROC', 'RLeg2', 'CPAP Press', 'Snore', 'ECG2', 'Chin2', 'LOC', 'Position', 'Airflow', 'CPAP Flow', 'M1', 'F4']

# Load data with selected features and evaluate for each subset of features
csv_file = "all_channels.csv"
accuracies = []

for i in range(1, len(selected_features) + 1):
    accuracy = train_and_evaluate(csv_file, sequence_length, selected_features[:i])
    accuracies.append((i, accuracy))
    print(f"Accuracy with {i} features: {accuracy:.4f}")

# Print all accuracies and corresponding number of features used
for num_features, accuracy in accuracies:
    print(f"Accuracy: {accuracy:.4f} with {num_features} features")
