import numpy as np
from typing import Dict, Tuple

def arrange_features_randomly(features_dict: Dict[str, np.ndarray], target_shape: Tuple[int, int] = None) -> Dict[str, np.ndarray]:
    """
    Arrange features into a random 2D matrix shape for CNN input.
    
    Args:
        features_dict: Dictionary containing 'X_train', 'X_val', 'X_test'
        target_shape: Desired shape (height, width) of the 2D matrix. If None, 
                     will use the nearest square to feature dimension
    
    Returns:
        Dictionary with reshaped features for CNN
    """
    # Get the number of features
    n_features = features_dict['X_train'].shape[1]
    
    # Determine target shape if not provided
    if target_shape is None:
        # Find the closest square or rectangular shape to total features
        side_length = int(np.ceil(np.sqrt(n_features)))
        target_shape = (side_length, side_length)
    
    # Calculate total size of the target shape
    total_size = target_shape[0] * target_shape[1]
    
    # Check if we need padding
    need_padding = total_size > n_features
    
    # Create a mapping of original feature indices to positions in the 2D grid
    feature_indices = np.arange(n_features)
    np.random.shuffle(feature_indices)  # Shuffle to randomize position in the grid
    
    # Create reshaped arrays for each set
    reshaped_dict = {}
    for key in ['X_train', 'X_val', 'X_test']:
        # Get the original data
        X = features_dict[key]
        n_samples = X.shape[0]
        
        # Create a padded array if needed
        if need_padding:
            padded_X = np.zeros((n_samples, total_size))
            padded_X[:, :n_features] = X
            X = padded_X
        
        # Reorder features based on the random mapping
        reordered_X = np.zeros((n_samples, total_size))
        for i, idx in enumerate(feature_indices):
            if i < total_size:  # Ensure we don't exceed the target size
                reordered_X[:, i] = X[:, idx]
        
        # Reshape to target shape with samples as the first dimension (for PyTorch: [N, C, H, W])
        # Note: PyTorch expects channel-first format: [batch_size, channels, height, width]
        reshaped_dict[key] = reordered_X.reshape(n_samples, 1, target_shape[0], target_shape[1])
    
    # Copy the labels
    reshaped_dict['y_train'] = features_dict['y_train']
    reshaped_dict['y_val'] = features_dict['y_val']
    reshaped_dict['y_test'] = features_dict['y_test']
    
    print(f"Features arranged randomly into shape {target_shape}")
    print(f"New data shapes: {reshaped_dict['X_train'].shape}")
    
    return reshaped_dict