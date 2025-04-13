import pandas as pd
import numpy as np
import os
import datatable as dt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(data_dir: str = 'split_data', use_datatable: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load train, validation, and test data using either pandas or datatable.
    
    Args:
        data_dir: Directory containing the split data files
        use_datatable: Whether to use datatable for faster loading
        
    Returns:
        Tuple of (features_dict, metadata_dict)
        - features_dict contains 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
        - metadata_dict contains 'feature_names', 'header_train', 'position_train', etc.
    """
    # Data files
    train_file = os.path.join(data_dir, 'train_data.csv')
    val_file = os.path.join(data_dir, 'val_data.csv')
    test_file = os.path.join(data_dir, 'test_data.csv')
    
    if use_datatable:
        print("Loading data using datatable...")
        train_data = dt.fread(train_file).to_pandas()
        val_data = dt.fread(val_file).to_pandas()
        test_data = dt.fread(test_file).to_pandas()
    else:
        print("Loading data using pandas...")
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        test_data = pd.read_csv(test_file)
    
    # Extract features, labels, and metadata
    X_train = train_data.drop(['Header', 'Position', 'target'], axis=1).values
    y_train = train_data['target'].values
    header_train = train_data['Header'].values
    position_train = train_data['Position'].values
    
    X_val = val_data.drop(['Header', 'Position', 'target'], axis=1).values
    y_val = val_data['target'].values
    header_val = val_data['Header'].values
    position_val = val_data['Position'].values
    
    X_test = test_data.drop(['Header', 'Position', 'target'], axis=1).values
    y_test = test_data['target'].values
    header_test = test_data['Header'].values
    position_test = test_data['Position'].values
    
    # Get feature names
    feature_names = train_data.drop(['Header', 'Position', 'target'], axis=1).columns.tolist()
    
    # Pack into dictionaries
    features_dict = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    metadata_dict = {
        'feature_names': feature_names,
        'header_train': header_train,
        'position_train': position_train,
        'header_val': header_val,
        'position_val': position_val,
        'header_test': header_test,
        'position_test': position_test
    }
    
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples, {X_test.shape[0]} test samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return features_dict, metadata_dict

def normalize_features(features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize features using StandardScaler.
    
    Args:
        features_dict: Dictionary containing 'X_train', 'X_val', 'X_test'
        
    Returns:
        Dictionary with normalized features
    """
    # Create a new dictionary to avoid modifying the original
    normalized_dict = {}
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    normalized_dict['X_train'] = scaler.fit_transform(features_dict['X_train'])
    
    # Transform validation and test data using the same scaler
    normalized_dict['X_val'] = scaler.transform(features_dict['X_val'])
    normalized_dict['X_test'] = scaler.transform(features_dict['X_test'])
    
    # Copy the labels
    normalized_dict['y_train'] = features_dict['y_train']
    normalized_dict['y_val'] = features_dict['y_val']
    normalized_dict['y_test'] = features_dict['y_test']
    
    print("Features normalized")
    
    return normalized_dict

def get_feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """
    Group features by their type (AAC, DPC, TPC, BE, PC)
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature group to list of feature indices
    """
    feature_groups = {
        'AAC': [],
        'DPC': [],
        'TPC': [],
        'BE': [],
        'PC': []
    }
    
    # Identify feature groups based on naming patterns
    for i, feature in enumerate(feature_names):
        if feature in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
            feature_groups['AAC'].append(i)
        elif len(feature) == 2 and all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in feature):
            feature_groups['DPC'].append(i)
        elif len(feature) == 3 and all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in feature):
            feature_groups['TPC'].append(i)
        elif feature.startswith('BE_'):
            feature_groups['BE'].append(i)
        elif feature.startswith('PC_'):
            feature_groups['PC'].append(i)
    
    # Print summary of feature groups
    for group, indices in feature_groups.items():
        print(f"{group}: {len(indices)} features")
    
    return feature_groups

class PhosphorylationDataset(Dataset):
    """PyTorch Dataset for phosphorylation prediction data"""
    
    def __init__(self, features, labels):
        """Initialize dataset with features and labels"""
        self.features = features
        self.labels = labels
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        x = torch.FloatTensor(self.features[idx])
        y = torch.FloatTensor([self.labels[idx]])
        return x, y

def create_data_loaders(arranged_features, batch_size=32):
    """Create PyTorch DataLoaders from arranged features"""
    # Create datasets
    train_dataset = PhosphorylationDataset(arranged_features['X_train'], arranged_features['y_train'])
    val_dataset = PhosphorylationDataset(arranged_features['X_val'], arranged_features['y_val'])
    test_dataset = PhosphorylationDataset(arranged_features['X_test'], arranged_features['y_test'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}