# feature_arrangers/natural_grouping_arranger.py
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

class NaturalGroupingArranger:
    """
    Arranges features by their natural grouping (AAC, DPC, TPC, BE, PC)
    into a single 2D feature map for CNN processing.
    """
    def __init__(self, feature_columns: List[str], height: Optional[int] = None, width: Optional[int] = None):
        """
        Initialize the natural grouping arranger
        
        Args:
            feature_columns: List of feature column names
            height: Optional height of output matrix (will calculate if not provided)
            width: Optional width of output matrix (will calculate if not provided)
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
        
        # Calculate dimensions for the output 2D grid
        self._calculate_dimensions(height, width)
        
        # Create feature arrangement map
        self.arrangement = self._create_arrangement()
    
    def _calculate_dimensions(self, height: Optional[int], width: Optional[int]):
        """
        Calculate appropriate dimensions for the feature grid
        """
        # Get total number of features
        total_features = (len(self.aac_cols) + len(self.dpc_cols) + 
                         len(self.tpc_cols) + len(self.be_cols) + 
                         len(self.pc_cols))
        
        # If specific dimensions are provided, use them
        if height is not None and width is not None:
            # Check if provided dimensions are sufficient
            if height * width < total_features:
                print(f"Warning: Specified dimensions ({height}x{width}) are too small for {total_features} features")
                # Calculate new dimensions
                side = int(np.ceil(np.sqrt(total_features)))
                self.height = side
                self.width = side
            else:
                self.height = height
                self.width = width
        else:
            # Calculate square-ish shape to fit all features
            side = int(np.ceil(np.sqrt(total_features)))
            self.height = side
            self.width = side
        
        print(f"Using grid dimensions: {self.height}x{self.width} for {total_features} features")
    
    def _create_arrangement(self):
        """
        Create a 2D arrangement of features based on their natural grouping
        """
        # Initialize the arrangement grid with -1 (indicating no feature)
        arrangement = np.ones((self.height, self.width), dtype=int) * -1
        
        # Create a mapping from feature name to column index
        feature_to_idx = {col: idx for idx, col in enumerate(self.feature_columns)}
        
        # Get all feature groups in a meaningful order
        all_groups = [
            ('AAC', self.aac_cols),
            ('DPC', self.dpc_cols),
            ('TPC', self.tpc_cols),
            ('BE', self.be_cols),
            ('PC', self.pc_cols)
        ]
        
        # Position tracking
        current_row = 0
        current_col = 0
        
        # Place features from each group in a rectangle-like area
        for group_name, cols in all_groups:
            if not cols:
                continue
            
            # Calculate a good rectangular shape for this group
            n_features = len(cols)
            width = min(n_features, self.width)
            height = (n_features + width - 1) // width  # Ceiling division
            
            # Check if we need to start a new row
            if current_col + width > self.width:
                current_row += 1
                current_col = 0
            
            # Check if we have enough space
            if current_row + height > self.height:
                print(f"Warning: Not enough space for all features. Some will be omitted.")
                break
            
            # Place features in the grid
            for i, col in enumerate(cols):
                if col in feature_to_idx:
                    row = current_row + (i // width)
                    col_idx = current_col + (i % width)
                    
                    # Check boundaries
                    if row < self.height and col_idx < self.width:
                        arrangement[row, col_idx] = feature_to_idx[col]
            
            # Update position for next group
            current_col += width
            if current_col >= self.width:
                current_row += height
                current_col = 0
        
        return arrangement
    
    def arrange_features(self, X):
        """
        Arrange features into a 2D grid based on natural grouping
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            X_arranged: Arranged features (n_samples, 1, height, width)
        """
        n_samples = X.shape[0]
        
        # Create a zero-filled array for the target shape
        X_arranged = np.zeros((n_samples, 1, self.height, self.width))
        
        # Fill the arranged array with feature values
        for i in range(self.height):
            for j in range(self.width):
                idx = self.arrangement[i, j]
                if idx >= 0 and idx < X.shape[1]:
                    X_arranged[:, 0, i, j] = X[:, idx]
        
        # Convert to PyTorch tensor
        return torch.from_numpy(X_arranged).float()