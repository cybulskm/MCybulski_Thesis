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

relevant_channels = [
    'F3', 'Nasal Pressure', 'Chin3', 'SpO2', 'E2', 'Pulse', 'Chin2', 'ECG1', 'Chin1' 
]

# Define a fixed sequence length for padding/truncating
sequence_length = 60  # Adjust this based on your data's typical length

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

# Load data
csv_file = '9_channels.csv'
X, y = load_data(csv_file, sequence_length)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.33, random_state=42)

# Reshape the data for the 1D CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Define the 1D CNN model with reduced complexity
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

# Train the model with callbacks
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate classification report
unique_labels = np.unique(y_true_classes)
target_names = label_encoder.inverse_transform(unique_labels)
report = classification_report(y_true_classes, y_pred_classes, target_names=target_names)
print(report)
print(f'Test Accuracy: {accuracy:.4f}')

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.show()

plot_history(history)

# Save the model
# model.save('cnn_model.h5')
