#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phosphorylation Site Prediction: Data Preparation Pipeline

This script processes protein sequence data and prepares it for phosphorylation site prediction.
It extracts features from protein sequences and creates a dataset ready for machine learning.

Usage:
    python phosphorylation_data_prep.py [window_size]

    window_size: half-size of the window around phosphorylation sites (default: 5)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_sequences(sequence_file):
    """
    Extract headers and sequences from a FASTA file.
    
    Parameters:
    sequence_file (str): Path to the FASTA file
    
    Returns:
    pandas.DataFrame: DataFrame with Header and Sequence columns
    """
    print("Extracting sequences from", sequence_file)
    
    # Lists to store the processed headers and sequences
    headers = []
    sequences = []
    current_seq = ""
    
    # Open and read the file
    with open(sequence_file, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # If there is an existing sequence, append it before starting a new one
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
                # Remove the ">" and extract the middle part from the header
                full_header = line[1:]
                parts = full_header.split("|")
                # Use the middle part if available; otherwise, use the full header
                middle = parts[1] if len(parts) > 1 else full_header
                headers.append(middle)
            else:
                # Concatenate sequence lines
                current_seq += line
        # Append the last collected sequence
        if current_seq:
            sequences.append(current_seq)
    
    # Create a DataFrame with the extracted header parts and sequences
    df = pd.DataFrame({
        "Header": headers,
        "Sequence": sequences
    })
    
    print(f"Extracted {len(df)} sequences")
    return df

def load_labels(labels_file):
    """
    Load the phosphorylation site labels from an Excel file.
    
    Parameters:
    labels_file (str): Path to the Excel file with labels
    
    Returns:
    pandas.DataFrame: DataFrame with labels
    """
    print("Loading labels from", labels_file)
    df_labels = pd.read_excel(labels_file)
    print(f"Loaded {len(df_labels)} labeled sites")
    return df_labels

def merge_and_clean_data(df_sequences, df_labels):
    """
    Merge sequence data with labels and clean the dataset.
    
    Parameters:
    df_sequences (pandas.DataFrame): DataFrame with sequences
    df_labels (pandas.DataFrame): DataFrame with labels
    
    Returns:
    pandas.DataFrame: Merged and cleaned DataFrame
    """
    print("Merging sequence and label data...")
    
    # Merge the sequence data with the labels
    df_merged = pd.merge(
        df_sequences,
        df_labels,
        left_on="Header",
        right_on="UniProt ID",
        how="left"
    )
    
    # Create a target column (1 for positive sites, 0 for negative)
    df_merged["target"] = df_merged["UniProt ID"].notnull().astype(int)
    
    # Drop rows with missing UniProt ID (these are sequences without labels)
    df_merged.dropna(subset=["UniProt ID"], inplace=True)
    
    # Drop sequences that are too long (for memory efficiency)
    df_merged = df_merged[df_merged["Sequence"].str.len() <= 5000]
    
    print(f"After merging and cleaning: {len(df_merged)} rows")
    return df_merged

def generate_negative_samples(df_merged):
    """
    Generate negative samples for each protein sequence.
    
    Parameters:
    df_merged (pandas.DataFrame): Merged DataFrame with positive sites
    
    Returns:
    pandas.DataFrame: DataFrame with both positive and negative samples
    """
    print("Generating balanced negative samples...")
    
    # Set Position column to integer type
    df_merged["Position"] = df_merged["Position"].astype(int)
    
    # Set a seed for reproducible random sampling
    random.seed(42)
    
    # We'll store the final data in a list of DataFrames, then concatenate at the end
    df_list = []
    
    # Group by the sequence ID (assuming 'Header' is unique per sequence)
    for header_value, group in df_merged.groupby("Header"):
        # Extract the amino-acid sequence (assuming one sequence per Header)
        seq = group["Sequence"].iloc[0]
        
        # (A) Positive positions from the DataFrame
        positive_positions = group["Position"].unique().tolist()
        
        # (B) Find all S/T/Y positions in the sequence
        st_y_positions = [i+1 for i, aa in enumerate(seq) if aa in ["S", "T", "Y"]]
        
        # (C) Exclude the positives → negative candidates
        negative_candidates = [pos for pos in st_y_positions if pos not in positive_positions]
        
        # (D) Number of positives for this sequence
        n_pos = len(positive_positions)
        
        # Sample negative sites (same number as positives if possible)
        if len(negative_candidates) >= n_pos:
            sampled_negatives = random.sample(negative_candidates, n_pos)
        else:
            # Just use whatever is available (partial negative set)
            sampled_negatives = negative_candidates
        
        # Create new rows for negative sites
        new_rows = []
        for neg_pos in sampled_negatives:
            new_rows.append({
                "Header": header_value,
                "Sequence": seq,
                "UniProt ID": group["UniProt ID"].iloc[0],
                "AA": seq[neg_pos - 1],   # -1 because 'neg_pos' is 1-based, Python string index is 0-based
                "Position": neg_pos,
                "target": 0
            })
        
        # Mark positives in the group with target=1
        group = group.copy()
        group["target"] = 1
        
        # Combine positives & negatives, store in a list
        neg_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([group, neg_df], ignore_index=True)
        df_list.append(combined_df)
    
    # Combine all groups into final DataFrame
    df_final = pd.concat(df_list, ignore_index=True)
    
    print(f"Final dataset with negative samples: {len(df_final)} rows")
    return df_final

def extract_window(sequence, position, window_size=5):
    """
    Extract a window of amino acids around a position.
    
    Parameters:
    sequence (str): Full protein sequence
    position (int): Position of interest (1-based)
    window_size (int): Half-size of the window on each side
    
    Returns:
    str: Window of amino acids
    """
    pos_idx = position - 1  # Convert to 0-based index
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    window = sequence[start:end]
    return window

def extract_aac(sequence):
    """
    Extract Amino Acid Composition (AAC) from a protein sequence.
    
    Parameters:
    sequence (str): Protein sequence
    
    Returns:
    dict: Dictionary with amino acids as keys and their frequencies as values
    """
    # List of 20 standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with zeros
    aac = {aa: 0 for aa in amino_acids}
    
    # Count amino acids
    seq_length = len(sequence)
    for aa in sequence:
        if aa in aac:
            aac[aa] += 1
    
    # Convert counts to frequencies
    for aa in aac:
        aac[aa] = aac[aa] / seq_length if seq_length > 0 else 0
        
    return aac

def extract_dpc(sequence):
    """
    Extract Dipeptide Composition (DPC) from a protein sequence.
    
    Parameters:
    sequence (str): Protein sequence
    
    Returns:
    dict: Dictionary with dipeptides as keys and their frequencies as values
    """
    # List of 20 standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with all possible dipeptides
    dpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            dpc[aa1 + aa2] = 0
    
    # Count dipeptides
    if len(sequence) < 2:
        return dpc
    
    total_dipeptides = len(sequence) - 1
    for i in range(total_dipeptides):
        dipeptide = sequence[i:i+2]
        if dipeptide in dpc:
            dpc[dipeptide] += 1
    
    # Convert counts to frequencies
    for dipeptide in dpc:
        dpc[dipeptide] = dpc[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0
        
    return dpc

def extract_tpc(sequence):
    """
    Extract Tripeptide Composition (TPC) from a protein sequence.
    
    Parameters:
    sequence (str): Protein sequence
    
    Returns:
    dict: Dictionary with tripeptides as keys and their frequencies as values
    """
    # List of 20 standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with all possible tripeptides
    tpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            for aa3 in amino_acids:
                tpc[aa1 + aa2 + aa3] = 0
    
    # Count tripeptides
    if len(sequence) < 3:
        return tpc
    
    total_tripeptides = len(sequence) - 2
    for i in range(total_tripeptides):
        tripeptide = sequence[i:i+3]
        if tripeptide in tpc:
            tpc[tripeptide] += 1
    
    # Convert counts to frequencies
    for tripeptide in tpc:
        tpc[tripeptide] = tpc[tripeptide] / total_tripeptides if total_tripeptides > 0 else 0
        
    return tpc

def process_tpc_in_batches(df, batch_size=500, window_size=5, output_dir="tpc_batches"):
    """
    Process TPC features in batches to avoid memory errors.
    
    Parameters:
    df (DataFrame): DataFrame containing sequences and positions
    batch_size (int): Number of samples to process in each batch
    window_size (int): Size of window around phosphorylation site
    output_dir (str): Directory to save batch files
    
    Returns:
    list: Paths to all batch files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate number of batches
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batch_files = []
    
    print(f"Processing {n_samples} samples in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        print(f"Processing batch {batch_idx+1}/{n_batches} (samples {start_idx}-{end_idx})")
        
        # Get batch data
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # Extract windows if not already done
        if 'Window' not in batch_df.columns:
            tqdm.pandas(desc="Extracting windows")
            batch_df['Window'] = batch_df.progress_apply(
                lambda row: extract_window(row['Sequence'], row['Position'], window_size=window_size), 
                axis=1
            )
        
        # Process TPC features for this batch
        tpc_batch = []
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Extracting TPC"):
            window = row['Window']
            tpc_dict = extract_tpc(window)
            # Add identifier columns and target
            tpc_dict['Header'] = row['Header']
            tpc_dict['Position'] = row['Position']
            tpc_dict['target'] = row['target']
            tpc_batch.append(tpc_dict)
        
        # Convert to DataFrame and save this batch
        batch_output_file = os.path.join(output_dir, f"tpc_features_batch_{batch_idx+1}.csv")
        tpc_batch_df = pd.DataFrame(tpc_batch)
        tpc_batch_df.to_csv(batch_output_file, index=False)
        
        # Release memory
        del tpc_batch, tpc_batch_df
        
        # Add file to list of batch files
        batch_files.append(batch_output_file)
        
        print(f"Batch {batch_idx+1} saved to {batch_output_file}")
    
    return batch_files

def combine_tpc_batches(batch_files, output_file="phosphorylation_tpc_features_window5.csv"):
    """
    Combine all TPC batch files into a single file.
    
    Parameters:
    batch_files (list): List of batch file paths
    output_file (str): Output file path
    
    Returns:
    str: Path to the combined file
    """
    print(f"Combining {len(batch_files)} batch files...")
    
    # Use pandas to combine batch files
    combined_df = pd.concat([pd.read_csv(file) for file in tqdm(batch_files, desc="Reading batches")])
    
    # Save combined file
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined TPC features saved to {output_file}")
    return output_file

def binary_encode_amino_acid(aa):
    """
    Binary encode a single amino acid into a 20-dimensional vector.
    
    Parameters:
    aa (str): Single letter amino acid code
    
    Returns:
    list: 20-dimensional binary vector
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize vector with zeros
    encoding = [0] * 20
    
    # Set the corresponding position to 1
    if aa in amino_acids:
        idx = amino_acids.index(aa)
        encoding[idx] = 1
    
    return encoding

def extract_binary_encoding(sequence, position, window_size=5):
    """
    Extract binary encoding features for a window around the phosphorylation site.
    
    Parameters:
    sequence (str): Full protein sequence
    position (int): Position of the phosphorylation site (1-based)
    window_size (int): Half size of the window on each side
    
    Returns:
    list: Binary encoding features for the window
    """
    # Convert position to 0-based index
    pos_idx = position - 1
    
    # Define window boundaries
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    # Extract window sequence
    window = sequence[start:end]
    
    # Pad with 'X' if necessary to reach the desired window length
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    # Binary encode each amino acid in the window
    binary_features = []
    for aa in padded_window:
        binary_features.extend(binary_encode_amino_acid(aa))
    
    return binary_features

def process_binary_encoding(df, window_size=5, batch_size=1000, output_file=None):
    """
    Process binary encoding features for all samples in batches.
    
    Parameters:
    df (DataFrame): DataFrame with sequences and positions
    window_size (int): Half size of the window on each side
    batch_size (int): Number of samples to process in each batch
    output_file (str): Output file for the features
    
    Returns:
    str: Path to the output file
    """
    if output_file is None:
        output_file = f"phosphorylation_binary_encoding_window{window_size}.csv"
        
    print(f"Extracting binary encoding features (window size {window_size*2+1})...")
    
    # Calculate number of batches
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_data = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        print(f"Processing batch {batch_idx+1}/{n_batches} (samples {start_idx}-{end_idx})")
        
        # Get batch data
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        batch_data = []
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Binary encoding"):
            # Extract binary features
            binary_features = extract_binary_encoding(row['Sequence'], row['Position'], window_size)
            
            # Create a dictionary with features
            feature_dict = {f"BE_{i+1}": binary_features[i] for i in range(len(binary_features))}
            
            # Add identifier columns and target
            feature_dict['Header'] = row['Header']
            feature_dict['Position'] = row['Position']
            feature_dict['target'] = row['target']
            
            batch_data.append(feature_dict)
        
        # Add batch data to all data
        all_data.extend(batch_data)
        
        print(f"Batch {batch_idx+1} processed")
    
    # Convert to DataFrame and save
    be_df = pd.DataFrame(all_data)
    be_df.to_csv(output_file, index=False)
    print(f"Binary encoding features saved to {output_file}")
    
    return output_file

def load_physicochemical_properties(file_path="physiochemical_property.csv"):
    """
    Load physicochemical properties from CSV file.
    
    Parameters:
    file_path (str): Path to CSV file with physicochemical properties
    
    Returns:
    dict: Dictionary mapping amino acids to their physicochemical properties
    """
    print(f"Loading physicochemical properties from {file_path}")
    prop_df = pd.read_csv(file_path)
    
    # Assuming first column is amino acid and others are properties
    properties = {}
    for _, row in prop_df.iterrows():
        aa = row.iloc[0]  # First column is amino acid
        properties[aa] = row.iloc[1:].values.tolist()
    
    return properties

def extract_physicochemical_features(sequence, position, window_size=5, properties=None):
    """
    Extract physicochemical features for a window around the phosphorylation site.
    
    Parameters:
    sequence (str): Full protein sequence
    position (int): Position of the phosphorylation site (1-based)
    window_size (int): Half size of the window on each side
    properties (dict): Dictionary mapping amino acids to their physicochemical properties
    
    Returns:
    list: Physicochemical features for the window
    """
    if properties is None:
        properties = load_physicochemical_properties()
    
    # Convert position to 0-based index
    pos_idx = position - 1
    
    # Define window boundaries
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    # Extract window sequence
    window = sequence[start:end]
    
    # Pad with 'X' if necessary to reach the desired window length
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    # Get properties for each amino acid in the window
    physico_features = []
    for aa in padded_window:
        if aa in properties:
            physico_features.extend(properties[aa])
        else:
            # Use zeros for unknown amino acids
            num_props = len(next(iter(properties.values())))
            physico_features.extend([0] * num_props)
    
    return physico_features

def process_physicochemical_features(df, window_size=5, batch_size=1000, output_file=None):
    """
    Process physicochemical features for all samples in batches.
    
    Parameters:
    df (DataFrame): DataFrame with sequences and positions
    window_size (int): Half size of the window on each side
    batch_size (int): Number of samples to process in each batch
    output_file (str): Output file for the features
    
    Returns:
    str: Path to the output file
    """
    if output_file is None:
        output_file = f"phosphorylation_physicochemical_window{window_size}.csv"
        
    print(f"Extracting physicochemical features (window size {window_size*2+1})...")
    
    # Load properties once
    properties = load_physicochemical_properties()
    
    # Calculate number of batches
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_data = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        print(f"Processing batch {batch_idx+1}/{n_batches} (samples {start_idx}-{end_idx})")
        
        # Get batch data
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        batch_data = []
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Physicochemical"):
            # Extract physicochemical features
            physico_features = extract_physicochemical_features(
                row['Sequence'], row['Position'], window_size, properties
            )
            
            # Create a dictionary with features
            feature_dict = {f"PC_{i+1}": physico_features[i] for i in range(len(physico_features))}
            
            # Add identifier columns and target
            feature_dict['Header'] = row['Header']
            feature_dict['Position'] = row['Position']
            feature_dict['target'] = row['target']
            
            batch_data.append(feature_dict)
        
        # Add batch data to all data
        all_data.extend(batch_data)
        
        print(f"Batch {batch_idx+1} processed")
    
    # Convert to DataFrame and save
    pc_df = pd.DataFrame(all_data)
    pc_df.to_csv(output_file, index=False)
    print(f"Physicochemical features saved to {output_file}")
    
    return output_file

def extract_aac_features(df, window_size=5, output_file=None):
    """
    Extract AAC features for all samples.
    
    Parameters:
    df (DataFrame): DataFrame with sequences and positions
    window_size (int): Half size of the window on each side
    output_file (str): Output file for the features
    
    Returns:
    str: Path to the output file
    """
    if output_file is None:
        output_file = f"phosphorylation_aac_features_window{window_size}.csv"
    
    print("Creating sequence windows...")
    tqdm.pandas(desc="Extracting windows")
    df['Window'] = df.progress_apply(
        lambda row: extract_window(row['Sequence'], row['Position'], window_size), 
        axis=1
    )
    
    print("Extracting AAC features...")
    aac_data = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="AAC features"):
        window = row['Window']
        aac_dict = extract_aac(window)
        # Add identifier columns and class label
        aac_dict['Header'] = row['Header']
        aac_dict['Position'] = row['Position']
        aac_dict['target'] = row['target']
        aac_data.append(aac_dict)
    
    aac_df = pd.DataFrame(aac_data)
    aac_df.to_csv(output_file, index=False)
    print(f"AAC features saved to {output_file}")
    
    return output_file

def extract_dpc_features(df, window_size=5, output_file=None):
    """
    Extract DPC features for all samples.
    
    Parameters:
    df (DataFrame): DataFrame with sequences and positions
    window_size (int): Half size of the window on each side
    output_file (str): Output file for the features
    
    Returns:
    str: Path to the output file
    """
    if output_file is None:
        output_file = f"phosphorylation_dpc_features_window{window_size}.csv"
    
    # Check if 'Window' column already exists
    if 'Window' not in df.columns:
        print("Creating sequence windows...")
        tqdm.pandas(desc="Extracting windows")
        df['Window'] = df.progress_apply(
            lambda row: extract_window(row['Sequence'], row['Position'], window_size), 
            axis=1
        )
    
    print("Extracting DPC features...")
    dpc_data = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="DPC features"):
        window = row['Window']
        dpc_dict = extract_dpc(window)
        # Add identifier columns and class label
        dpc_dict['Header'] = row['Header']
        dpc_dict['Position'] = row['Position']
        dpc_dict['target'] = row['target']
        dpc_data.append(dpc_dict)
    
    dpc_df = pd.DataFrame(dpc_data)
    dpc_df.to_csv(output_file, index=False)
    print(f"DPC features saved to {output_file}")
    
    return output_file

def merge_all_features(window_size=5, output_file=None):
    """
    Merge all extracted features for machine learning.
    
    Parameters:
    window_size (int): Window size used for feature extraction
    output_file (str): Output file for merged features
    
    Returns:
    str: Path to the output file
    """
    if output_file is None:
        output_file = f"phosphorylation_all_features_window{window_size}.csv"
        
    print("Merging all features...")
    
    # Load all feature files
    aac_df = pd.read_csv(f"phosphorylation_aac_features_window{window_size}.csv")
    dpc_df = pd.read_csv(f"phosphorylation_dpc_features_window{window_size}.csv")
    tpc_df = pd.read_csv(f"phosphorylation_tpc_features_window{window_size}.csv")
    be_df = pd.read_csv(f"phosphorylation_binary_encoding_window{window_size}.csv")
    pc_df = pd.read_csv(f"phosphorylation_physicochemical_window{window_size}.csv")
    
    # Merge all features on Header and Position
    merged_df = aac_df.merge(dpc_df, on=['Header', 'Position', 'target'])
    merged_df = merged_df.merge(tpc_df, on=['Header', 'Position', 'target'])
    merged_df = merged_df.merge(be_df, on=['Header', 'Position', 'target'])
    merged_df = merged_df.merge(pc_df, on=['Header', 'Position', 'target'])
    
    # Save merged features
    merged_df.to_csv(output_file, index=False)
    print(f"All features merged and saved to {output_file}")
    
    return output_file

def split_data(features_file, test_size=0.2, val_size=0.25, random_state=42, output_dir="split_data"):
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
    features_file (str): Path to the file with all features
    test_size (float): Proportion of data for testing
    val_size (float): Proportion of training data for validation
    random_state (int): Random seed for reproducibility
    output_dir (str): Directory to save the split data
    
    Returns:
    dict: Paths to the output files
    """
    # Create a directory to store the split data
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {features_file}...")
    final_ready_features = pd.read_csv(features_file)
    
    # Extract features and target
    print("Splitting data...")
    X = final_ready_features.drop(['Header', 'Position', 'target'], axis=1)
    y = final_ready_features['target']
    id_cols = final_ready_features[['Header', 'Position']]
    
    # First split into train+val and test (80/20)
    X_train_val, X_test, y_train_val, y_test, id_train_val, id_test = train_test_split(
        X, y, id_cols, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_train_val, y_train_val, id_train_val, 
        test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    # Save each split to CSV
    print("Saving train set...")
    train_df = pd.concat([id_train, X_train, y_train], axis=1)
    train_output = os.path.join(output_dir, "train_data.csv")
    train_df.to_csv(train_output, index=False)
    
    print("Saving validation set...")
    val_df = pd.concat([id_val, X_val, y_val], axis=1)
    val_output = os.path.join(output_dir, "val_data.csv")
    val_df.to_csv(val_output, index=False)
    
    print("Saving test set...")
    test_df = pd.concat([id_test, X_test, y_test], axis=1)
    test_output = os.path.join(output_dir, "test_data.csv")
    test_df.to_csv(test_output, index=False)
    
    print("Data splitting and saving complete!")
    
    return {
        'train': train_output,
        'val': val_output,
        'test': test_output
    }

def main(window_size=5):
    """
    Main function that runs the complete data preparation pipeline.
    
    Parameters:
    window_size (int): Half-size of the window around phosphorylation sites
    """
    print("=" * 80)
    print(f"Phosphorylation Site Prediction: Data Preparation Pipeline (window_size={window_size})")
    print("=" * 80)
    
    # Step 1: Extract sequences from FASTA file
    sequence_file = "Sequence_data.txt"
    df_sequences = extract_sequences(sequence_file)
    
    # Step 2: Load phosphorylation site labels
    labels_file = "labels.xlsx"
    df_labels = load_labels(labels_file)
    
    # Step 3: Merge and clean data
    df_merged = merge_and_clean_data(df_sequences, df_labels)
    
    # Step 4: Generate balanced negative samples
    df_final = generate_negative_samples(df_merged)
    
    # Step 5: Save raw merged data with negative samples (optional)
    raw_output = "raw_merged_with_negative_samples.csv"
    df_final.to_csv(raw_output, index=False)
    print(f"Raw merged data with negative samples saved to {raw_output}")
    
    # Step 6: Extract AAC features
    aac_file = extract_aac_features(df_final, window_size)
    
    # Step 7: Extract DPC features
    dpc_file = extract_dpc_features(df_final, window_size)
    
    # Step 8: Extract TPC features
    tpc_batch_dir = f"tpc_batches_window{window_size}"
    batch_files = process_tpc_in_batches(df_final, window_size=window_size, output_dir=tpc_batch_dir)
    tpc_file = combine_tpc_batches(batch_files, output_file=f"phosphorylation_tpc_features_window{window_size}.csv")
    
    # Step 9: Extract binary encoding features
    be_file = process_binary_encoding(df_final, window_size)
    
    # Step 10: Extract physicochemical features
    pc_file = process_physicochemical_features(df_final, window_size)
    
    # Step 11: Merge all features
    all_features_file = merge_all_features(window_size)
    
    # Step 12: Split data into train, validation, and test sets
    split_data(all_features_file)
    
    print("=" * 80)
    print("Data preparation complete! The data is now ready for model building.")
    print("=" * 80)

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            window_size = int(sys.argv[1])
            print(f"Using window size: {window_size}")
        except ValueError:
            print("Invalid window size. Using default window size of 5.")
            window_size = 5
    else:
        window_size = 5  # Default window size
    
    # Run the main pipeline
    main(window_size)