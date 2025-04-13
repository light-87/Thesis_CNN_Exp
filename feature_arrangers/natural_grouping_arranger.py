# feature_arrangers/natural_grouping_arranger.py
import numpy as np
import torch

class NaturalGroupingArranger:
    """
    Arranges features by their natural grouping (AAC, DPC, TPC, BE, PC)
    """
    def __init__(self, feature_columns, height=None, width=None):
        """
        Initializes the natural grouping arranger
        
        Args:
            feature_columns: List of feature column names
            height: Optional height of output 2D matrix
            width: Optional width of output 2D matrix
        """
        self.feature_columns = feature_columns
        
        # Define the column groups based on naming pattern
        self.aac_cols = [col for col in feature_columns if col in list('ACDEFGHIKLMNPQRSTVWY')]
        self.dpc_cols = [col for col in feature_columns if len(col) == 2 and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in col)]
        self.tpc_cols = [col for col in feature_columns if len(col) == 3 and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in col)]
        self.be_cols = [col for col in feature_columns if col.startswith('BE_')]
        self.pc_cols = [col for col in feature_columns if col.startswith('PC_')]
        
        # Calculate default dimensions if not provided
        total_features = len(feature_columns)
        if height is None or width is None:
            # Try to make it as square as possible
            size = int(np.ceil(np.sqrt(total_features)))
            self.height = size
            self.width = size
        else:
            self.height = height
            self.width = width
            
        # Verify dimensions are sufficient
        if self.height * self.width < total_features:
            size = int(np.ceil(np.sqrt(total_features)))
            self.height = size
            self.width = size
            print(f"Warning: Specified dimensions too small, adjusted to {self.height}x{self.width}")
            
        # Create feature arrangement map
        self.arrangement = self._create_arrangement()
        
    def _create_arrangement(self):
        """
        Creates the feature arrangement by placing similar feature types together
        """
        # Arrange features by group
        all_columns_ordered = self.aac_cols + self.dpc_cols + self.tpc_cols + self.be_cols + self.pc_cols
        
        # Create a map for faster index lookup
        column_to_idx = {col: idx for idx, col in enumerate(self.feature_columns)}
        
        # Create the 2D arrangement
        arrangement = np.zeros((self.height, self.width), dtype=int)
        
        # Fill in the arrangement
        for i, col in enumerate(all_columns_ordered):
            if i >= self.height * self.width:
                break
            row = i // self.width
            col_idx = i % self.width
            arrangement[row, col_idx] = column_to_idx.get(col, 0)
            
        # Fill any remaining spots with zeros
        return arrangement
    
    def arrange_features(self, X):
        """
        Arranges features into a 2D grid based on natural grouping
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            X_arranged: Arranged features (n_samples, 1, height, width)
        """
        n_samples = X.shape[0]
        X_arranged = np.zeros((n_samples, 1, self.height, self.width))
        
        # Use advanced indexing to map features
        for i in range(self.height):
            for j in range(self.width):
                idx = self.arrangement[i, j]
                if idx < X.shape[1]:
                    X_arranged[:, 0, i, j] = X[:, idx]
        
        return torch.from_numpy(X_arranged).float()