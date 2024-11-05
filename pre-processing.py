import pyedflib
import pandas as pd
import numpy as np
import os

# Define the relevant channels for sleep apnea diagnosis
relevant_channels = [
    'Airflow', 'Nasal Pressure', 'SpO2', 'ECG1', 'ECG2',
    'Thor', 'Abdo', 'Snore', 'Pulse', 'Respiratory Rate',
    'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2',
    'Chin1', 'Chin2', 'Chin3'
]

def extract_features_from_edf(file_path):
    edf_file = pyedflib.EdfReader(file_path)
    print(file_path)
    features = {channel: [] for channel in relevant_channels}
    min_length = float('inf')
    
    # Find the minimum length of all signals to ensure all arrays are of the same length
    for channel in relevant_channels:
        try:
            signal_index = edf_file.getSignalLabels().index(channel)
            signal_data = edf_file.readSignal(signal_index)
            if len(signal_data) < min_length:
                min_length = len(signal_data)
        except ValueError:
            print(file_path)
            print(f"Channel {channel} not found in the EDF file.")
    
    # Extract data for each relevant channel and truncate to the minimum length
    for channel in relevant_channels:
        try:
            signal_index = edf_file.getSignalLabels().index(channel)
            signal_data = edf_file.readSignal(signal_index)[:min_length]
            features[channel] = signal_data
        except ValueError:
            print(file_path)
            print(f"Channel {channel} not found in the EDF file.")
    
    edf_file.close()
    features_df = pd.DataFrame(features)
    return features_df

def classify_apnea(event_segment):
    airflow = event_segment['Airflow']
    thoracic_effort = event_segment['Thor']
    abdominal_effort = event_segment['Abdo']
    
    threshold = 0.2  # Define a threshold for detecting significant reductions
    if np.mean(airflow) < threshold and np.mean(thoracic_effort) < threshold and np.mean(abdominal_effort) < threshold:
        return 'CSA'
    elif np.mean(airflow) < threshold and (np.mean(thoracic_effort) > threshold or np.mean(abdominal_effort) > threshold):
        return 'OSA'
    else:
        return 'Normal'

def label_and_combine_data(csa_dir, osa_dir, segment_length=30):
    data = []
    
    # Process CSA files
    for file_name in os.listdir(csa_dir):
        file_path = os.path.join(csa_dir, file_name)
        features_df = extract_features_from_edf(file_path)
        features_df['Label'] = 'CSA'
        data.append(features_df)
    
    # Process OSA files
    for file_name in os.listdir(osa_dir):
        file_path = os.path.join(osa_dir, file_name)
        features_df = extract_features_from_edf(file_path)
        features_df['Label'] = 'OSA'
        data.append(features_df)
    
    # Combine all data into a single DataFrame
    combined_df = pd.concat(data, ignore_index=True)
    
    # Normalize the data
    for channel in relevant_channels:
        combined_df[channel] = (combined_df[channel] - combined_df[channel].mean()) / combined_df[channel].std()
    
    # Segment the data and classify each segment
    segments = []
    labels = []
    for i in range(0, len(combined_df), segment_length):
        segment = combined_df.iloc[i:i+segment_length]
        if len(segment) == segment_length:
            segments.append(segment[relevant_channels].values)
            label = classify_apnea(segment)
            labels.append(label)
    
    segments = np.array(segments)
    labels = np.array(labels)
    
    return segments, labels

def save_to_csv(segments, labels, output_file):
    # Flatten the segments array and create a DataFrame
    flattened_segments = segments.reshape(segments.shape[0], -1)
    df_segments = pd.DataFrame(flattened_segments, columns=[f'{channel}_{i}' for i in range(segments.shape[1]) for channel in relevant_channels])
    
    # Add the labels to the DataFrame
    df_segments['Label'] = labels
    
    # Save the DataFrame to a CSV file
    df_segments.to_csv(output_file, index=False)

# Example usage
csa_dir = 'CSA-data'  # Directory containing CSA EDF files
osa_dir = 'OSA-data'  # Directory containing OSA EDF files
segments, labels = label_and_combine_data(csa_dir, osa_dir)

# Save the preprocessed data to a CSV file
output_file = 'preprocessed_psg_data.csv'
save_to_csv(segments, labels, output_file)
print("Data saved to csv: ", output_file)

# Display the shape of the preprocessed data
print(f'Segments shape: {segments.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Data saved to {output_file}')
