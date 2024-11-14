import os
import pandas as pd
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
import csv

def parse_annotations(rml_file_path):
    tree = ET.parse(rml_file_path)
    root = tree.getroot()
    namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
    events = []
    for event in root.findall('.//ns:Event', namespace):
        event_type = event.get('Type')
        if "apnea" in event_type.lower():
            start_time = float(event.get('Start'))
            duration = float(event.get('Duration'))
            events.append((event_type, start_time, duration))
    return events

def extract_features_from_edf(edf_file_path):
    edf_file = pyedflib.EdfReader(edf_file_path)
    signal_labels = edf_file.getSignalLabels()
    features = {channel: [] for channel in signal_labels}
    sampling_rates = {}
    
    for channel in signal_labels:
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
    
    min_length = min(len(signal) for signal in features.values() if signal is not None)
    for channel in signal_labels:
        if features[channel] is not None:
            features[channel] = features[channel][:min_length]
        else:
            features[channel] = [np.nan] * min_length
    
    features_df = pd.DataFrame(features)
    return features_df, sampling_rates

def preprocess_and_label(edf_file_path, annotations, remaining_annotations):
    features_df, sampling_rates = extract_features_from_edf(edf_file_path)
    
    scaler = StandardScaler()
    for channel in features_df.columns:
        if features_df[channel].isnull().all():
            continue
        features_df[channel] = scaler.fit_transform(features_df[channel].values.reshape(-1, 1)).flatten()
    
    segments = []
    min_sampling_rate = int(min(rate for rate in sampling_rates.values() if rate))
    window_size = 60 * min_sampling_rate
    edf_duration = len(features_df) / min_sampling_rate
    
    for event in annotations:
        label, start_time, _ = event
        if start_time >= edf_duration:
            remaining_annotations.append((label, start_time - edf_duration, _))
            continue
        
        start_idx = int(start_time * min_sampling_rate)
        end_idx = start_idx + window_size
        segment_data = {}
        for channel in features_df.columns:
            if sampling_rates[channel] is not None:
                channel_data = features_df[channel].iloc[start_idx:end_idx].tolist()
                if len(channel_data) < window_size:
                    channel_data += [np.nan] * (window_size - len(channel_data))
                segment_data[channel] = channel_data
            else:
                segment_data[channel] = [np.nan] * window_size
        
        segment_data['Label'] = label
        segments.append(segment_data)
    
    return segments, remaining_annotations

def save_to_csv(data, output_csv):
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

def process_events_for_patient(rml_path, edf_group, output_file):
    file_pairs = [(edf, rml_path) for edf in edf_group]
    process_files(file_pairs, output_file)

def process_files(file_pairs, output_csv):
    all_segments = []
    remaining_annotations = []

    for edf_file, rml_file in file_pairs:
        annotations = parse_annotations(rml_file)
        annotations = remaining_annotations + annotations
        remaining_annotations = []
        segments, remaining_annotations = preprocess_and_label(edf_file, annotations, remaining_annotations)
        all_segments.extend(segments)
    
    save_to_csv(all_segments, output_csv)
    print(f"Data has been saved to {output_csv}")

# Directory paths and file processing
input_dir = 'dataset'
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(input_dir):
    rml_files = [f for f in files if f.endswith('.rml')]
    edf_files = sorted([os.path.join(root, f) for f in files if f.endswith('.edf')])

    for rml_file in rml_files:
        rml_path = os.path.join(root, rml_file)
        edf_group = [edf for edf in edf_files if rml_file.split('.rml')[0] in edf]

        output_file = os.path.join(output_dir, f"{os.path.splitext(rml_file)[0]}.csv")
        process_events_for_patient(rml_path, edf_group, output_file)