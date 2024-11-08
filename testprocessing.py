import pandas as pd
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler

# Define the relevant channels for sleep apnea diagnosis
relevant_channels = [
    'Airflow', 'Nasal Pressure', 'SpO2', 'ECG1', 'ECG2',
    'Thor', 'Abdo', 'Snore', 'Pulse', 'Respiratory Rate',
    'F3', 'F4', 'C3', 'C4', 'O1', 'O2'
]
relevant_events = [  
  "Hypopnea", "Central Apnea", "Obstructive Apnea", "Mixed Apnea"]

def extract_features_from_edf(file_path):
    edf_file = pyedflib.EdfReader(file_path)
    features = {channel: [] for channel in relevant_channels}
    sampling_rates = {}
    
    # Extract data for each relevant channel and its sampling rate
    for channel in relevant_channels:
        try:
            signal_index = edf_file.getSignalLabels().index(channel)
            signal_data = edf_file.readSignal(signal_index)
            sampling_rate = edf_file.getSampleFrequency(signal_index)
            features[channel] = signal_data
            sampling_rates[channel] = sampling_rate
        except ValueError:
            print(f"Channel {channel} not found in the EDF file.")
            features[channel] = None
            sampling_rates[channel] = None
    
    edf_file.close()
    
    # Find the minimum length of all signals to ensure consistency
    min_length = min(len(signal) for signal in features.values() if signal is not None)
    for channel in relevant_channels:
        if features[channel] is not None:
            features[channel] = features[channel][:min_length]
        else:
            features[channel] = [np.nan] * min_length  # Fill missing channels with NaN
    
    features_df = pd.DataFrame(features)
    return features_df, sampling_rates

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    for event in root.findall('.//ScoredEvent'):
        if (event.find('Name').text) not in relevant_events:
            continue
        else:
            name = event.find('Name').text
            start = float(event.find('Start').text)
            duration = float(event.find('Duration').text)
            annotations.append((name, start, duration))
        
    return annotations

def preprocess_and_label(edf_file_path, xml_file_path):
    features_df, sampling_rates = extract_features_from_edf(edf_file_path)
    
    # Normalize the data for each channel
    scaler = StandardScaler()
    for channel in relevant_channels:
        if features_df[channel].isnull().all():  # Skip channels with all NaN values
            continue
        features_df[channel] = scaler.fit_transform(features_df[channel].values.reshape(-1, 1)).flatten()
    
    # Parse annotations
    annotations = parse_annotations(xml_file_path)
    
    segments = []
    
    # Create 60-second windows for each annotated event
    min_sampling_rate = int(min(rate for rate in sampling_rates.values() if rate))
    window_size = 60 * min_sampling_rate
    
    for event in annotations:
        label, start_time, _ = event
        start_idx = int(start_time * min_sampling_rate)
        end_idx = start_idx + window_size
        
        # Create a data segment for this event from each channel
        segment_data = {}
        for channel in relevant_channels:
            if sampling_rates[channel] is not None:
                channel_data = features_df[channel].iloc[start_idx:end_idx].tolist()
                
                # Pad with NaNs if there's insufficient data for the window
                if len(channel_data) < window_size:
                    channel_data += [np.nan] * (window_size - len(channel_data))
                
                segment_data[channel] = channel_data
            else:
                segment_data[channel] = [np.nan] * window_size
        
        segment_data['Label'] = label
        segments.append(segment_data)
    
    return segments

def save_to_csv(segments, file_name):
    df_segments = pd.DataFrame(segments)
    df_segments.to_csv(file_name, index=False)

def process_files(file_pairs, output_csv):
    all_segments = []
    
    # Process each EDF/XML file pair in the provided list
    for edf_file, xml_file in file_pairs:
        print(f"Processing {edf_file} and {xml_file}")
        segments = preprocess_and_label(edf_file, xml_file)
        all_segments.extend(segments)
    
    # Save all segments to a single CSV file
    save_to_csv(all_segments, output_csv)
    print(f"Data has been saved to {output_csv}")

# Example usage
file_pairs = [
    ('CSA-data/CSA AHI 5-15/1/1', 'CSA-data/CSA AHI 5-15/1/1.XML'),
    ('CSA-data/CSA AHI 5-15/2/2', 'CSA-data/CSA AHI 5-15/2/2.XML'),
    ('CSA-data/CSA AHI 5-15/3/3', 'CSA-data/CSA AHI 5-15/3/3.XML'),
    ('CSA-data/CSA AHI 5-15/4/4', 'CSA-data/CSA AHI 5-15/4/4.XML'),
    ('CSA-data/CSA AHI 15-30/1/1', 'CSA-data/CSA AHI 15-30/1/1.XML'),
    ('CSA-data/CSA AHI 15-30/2/2', 'CSA-data/CSA AHI 15-30/2/2.XML'),
    ('CSA-data/CSA AHI 15-30/3/3', 'CSA-data/CSA AHI 15-30/3/3.XML'),
    ('CSA-data/CSA AHI 15-30/4/4', 'CSA-data/CSA AHI 15-30/4/4.XML'),
    ('CSA-data/CSA AHI lower 5/1/1', 'CSA-data/CSA AHI lower 5/1/1.XML'),
    ('CSA-data/CSA AHI lower 5/2/2', 'CSA-data/CSA AHI lower 5/2/2.XML'),
    ('CSA-data/CSA AHI lower 5/3/3', 'CSA-data/CSA AHI lower 5/3/3.XML'),
    ('CSA-data/CSA AHI lower 5/4/4', 'CSA-data/CSA AHI lower 5/4/4.XML'),
    ('CSA-data/CSA AHI upper 30/1/1', 'CSA-data/CSA AHI upper 30/1/1.XML'),
    ('CSA-data/CSA AHI upper 30/2/2', 'CSA-data/CSA AHI upper 30/2/2.XML'),
    ('CSA-data/CSA AHI upper 30/4/4', 'CSA-data/CSA AHI upper 30/4/4.XML'),
    ('OSA-data/OSA AHI 5-15/1/1', 'OSA-data/OSA AHI 5-15/1/1.XML'),
    ('OSA-data/OSA AHI 5-15/2/2', 'OSA-data/OSA AHI 5-15/2/2.XML'),
    ('OSA-data/OSA AHI 5-15/3/3', 'OSA-data/OSA AHI 5-15/3/3.XML'),
    ('OSA-data/OSA AHI 5-15/4/4', 'OSA-data/OSA AHI 5-15/4/4.XML'),
    ('OSA-data/OSA AHI 15-30/1/1', 'OSA-data/OSA AHI 15-30/1/1.XML'),
    ('OSA-data/OSA AHI 15-30/2/2', 'OSA-data/OSA AHI 15-30/2/2.XML'),
    ('OSA-data/OSA AHI 15-30/3/3', 'OSA-data/OSA AHI 15-30/3/3.XML'),
    ('OSA-data/OSA AHI 15-30/4/4', 'OSA-data/OSA AHI 15-30/4/4.XML'),
    ('OSA-data/OSA AHI lower 5/1/1', 'OSA-data/OSA AHI lower 5/1/1.XML'),
    ('OSA-data/OSA AHI lower 5/2/2', 'OSA-data/OSA AHI lower 5/2/2.XML'),
    ('OSA-data/OSA AHI lower 5/3/3', 'OSA-data/OSA AHI lower 5/3/3.XML'),
    ('OSA-data/OSA AHI lower 5/4/4', 'OSA-data/OSA AHI lower 5/4/4.XML'),
    ('OSA-data/OSA AHI upper 30/1/1', 'OSA-data/OSA AHI upper 30/1/1.XML'),
    ('OSA-data/OSA AHI upper 30/2/2', 'OSA-data/OSA AHI upper 30/2/2.XML'),
    ('OSA-data/OSA AHI upper 30/4/4', 'OSA-data/OSA AHI upper 30/4/4.XML')
    ]

output_csv_file = '16_channels.csv'

# Process all EDF/XML pairs from the list and save to a CSV file
process_files(file_pairs, output_csv_file)
