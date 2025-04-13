# feature_arrangers/multichannel_arranger.py
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

class MultichannelArranger:
    """
    Arranges features into multiple channels based on feature type
    (AAC, DPC, TPC, BE, PC) for multi-channel CNN processing.
    """
    def __init__(self, feature_columns: List[str]):
        """
        Initialize the multi-channel arranger
        
        Args:
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns
        
        # Define the column groups based on naming pattern
        self.aac_cols = [col for col in feature_columns if col in list('ACDEFGHIKLMNPQRSTVWY')]
        self.dpc_cols = [col for col in feature_columns if len(col) == 2 and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in col)]
        self.tpc_cols = [col for col in feature_columns if len(col) == 3 and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in col)]
        self.be_cols = [col for col in feature_columns if col.startswith('BE_')]
        self.pc_cols = [col for col in feature_columns if col.startswith('PC_')]
        
        # Report feature group sizes
        print(f"Feature groups found:")
        print(f"  AAC features: {len(self.aac_cols)}")
        print(f"  DPC features: {len(self.dpc_cols)}")
        print(f"  TPC features: {len(self.tpc_cols)}")
        print(f"  BE features: {len(self.be_cols)}")
        print(f"  PC features: {len(self.pc_cols)}")
        
        # Store feature groups and their sizes
        self.feature_groups = [
            ('AAC', self.aac_cols),
            ('DPC', self.dpc_cols),
            ('TPC', self.tpc_cols),
            ('BE', self.be_cols),
            ('PC', self.pc_cols)
        ]
        
        # Filter out empty groups
        self.active_groups = [(name, cols) for name, cols in self.feature_groups if cols]
        self.group_names = [name for name, cols in self.active_groups]
        
        print(f"Using {len(self.active_groups)} channels: {self.group_names}")
        
        # Calculate shapes for each group
        self.group_shapes = self._calculate_group_shapes()
        
        # Create feature mappings for each group
        self.feature_indices = self._create_feature_indices()
        
    def _calculate_group_shapes(self) -> Dict[str, Tuple[int, int]]:
        """
        Calculate appropriate shapes for each feature group
        """
        shapes = {}
        
        for group_name, cols in self.active_groups:
            n_features = len(cols)
            
            # Calculate a square-ish shape
            side = int(np.ceil(np.sqrt(n_features)))
            shapes[group_name] = (side, side)
            
            print(f"  {group_name} shape: {shapes[group_name]}")
            
        return shapes
    
    def _create_feature_indices(self) -> Dict[str, Dict[str, int]]:
        """
        Create mapping from feature names to indices
        """
        feature_indices = {}
        
        # Create feature to index mapping for the entire dataset
        all_feature_to_idx = {col: idx for idx, col in enumerate(self.feature_columns)}
        
        # Create mapping for each group
        for group_name, cols in self.active_groups:
            feature_indices[group_name] = {col: all_feature_to_idx[col] for col in cols if col in all_feature_to_idx}
            
        return feature_indices
    
    def arrange_features(self, X: np.ndarray) -> torch.Tensor:
        """
        Arrange features into separate channels based on feature type
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            X_arranged: Arranged features (n_samples, n_channels, height, width)
                        where each channel corresponds to a feature type
        """
        n_samples = X.shape[0]
        n_channels = len(self.active_groups)
        
        # Find the maximum height and width for padding
        max_height = max(height for height, _ in self.group_shapes.values())
        max_width = max(width for _, width in self.group_shapes.values())
        
        # List to store tensors for each channel (all padded to the same size)
        channel_tensors = []
        
        # Process each feature group as a separate channel
        for group_name, cols in self.active_groups:
            height, width = self.group_shapes[group_name]
            
            # Create a zero-filled tensor for this channel
            channel = np.zeros((n_samples, max_height, max_width))
            
            # Get indices mapping for this group
            indices = self.feature_indices[group_name]
            
            # Place features in the channel in row-major order
            for i, col in enumerate(cols):
                if col in indices:
                    feature_idx = indices[col]
                    
                    # Calculate position in 2D grid
                    row = i // width
                    col_idx = i % width
                    
                    # Make sure we don't exceed the dimensions
                    if row < height and col_idx < width and row < max_height and col_idx < max_width:
                        channel[:, row, col_idx] = X[:, feature_idx]
            
            # Add this channel to our list
            channel_tensors.append(channel)
        
        # Stack all channels along the channel dimension
        X_arranged = np.stack(channel_tensors, axis=1)
        
        # Convert to PyTorch tensor
        return torch.from_numpy(X_arranged).float()