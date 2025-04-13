import numpy as np
import json
import os
from typing import Dict, Tuple, List, Optional

def arrange_features_by_importance(features_dict: Dict[str, np.ndarray], 
                                   target_shape: Tuple[int, int] = None,
                                   importance_file: str = 'feature_importance.json',
                                   feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Arrange features based on their importance (most important in the center)
    
    Args:
        features_dict: Dictionary containing 'X_train', 'X_val', 'X_test'
        target_shape: Desired shape (height, width) of the 2D matrix. If None, 
                     will use the nearest square to feature dimension
        importance_file: Path to JSON file with feature importance scores
        feature_names: List of feature names (optional). If not provided, will use index as names
        
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
    
    # Load feature importance
    print(f"Loading feature importance from {importance_file}")
    try:
        with open(importance_file, 'r') as f:
            importance_dict = json.load(f)
        
        # Create ordered list of feature names by importance (most important first)
        ordered_features = list(importance_dict.keys())
        print(f"Loaded {len(ordered_features)} features with importance scores")
    except Exception as e:
        print(f"Error loading feature importance: {e}")
        print("Using random feature ordering instead")
        if feature_names is None:
            feature_names = [str(i) for i in range(n_features)]
        ordered_features = feature_names.copy()
        np.random.shuffle(ordered_features)
    
    # If feature_names is not provided, we'll try to match by index
    if feature_names is None:
        # Assume feature names match the keys in importance_dict (this might not always work!)
        feature_ordering = ordered_features[:n_features]
        feature_to_index = {feature: i for i, feature in enumerate(feature_ordering)}
    else:
        # Map each feature name to its index
        feature_to_index = {feature: i for i, feature in enumerate(feature_names)}
        
        # Filter ordered_features to keep only those that exist in feature_names
        valid_ordered_features = [f for f in ordered_features if f in feature_to_index]
        
        # Add any missing features at the end
        missing_features = [f for f in feature_names if f not in set(valid_ordered_features)]
        feature_ordering = valid_ordered_features + missing_features
    
    # Create a spiraling order for the 2D grid, starting from the center
    height, width = target_shape
    center_y, center_x = height // 2, width // 2
    
    # Create grid coordinates
    grid_coords = []
    for i in range(max(height, width)):
        for y in range(center_y - i, center_y + i + 1):
            for x in range(center_x - i, center_x + i + 1):
                # Check if the coordinate is on the boundary of the current square
                if (y == center_y - i or y == center_y + i or 
                    x == center_x - i or x == center_x + i):
                    # Check if the coordinate is within the grid
                    if 0 <= y < height and 0 <= x < width:
                        # Check if we haven't added this coordinate already
                        if (y, x) not in grid_coords:
                            grid_coords.append((y, x))
    
    # If we haven't filled the grid completely, add any missing coordinates
    for y in range(height):
        for x in range(width):
            if (y, x) not in grid_coords:
                grid_coords.append((y, x))
    
    # Create a mapping of feature indices to positions in the 2D grid
    # Most important features go to the center coordinates
    feature_to_position = {}
    for i, feature in enumerate(feature_ordering):
        if i < len(grid_coords):
            # If the feature is in our dataset
            if feature in feature_to_index:
                feature_to_position[feature_to_index[feature]] = grid_coords[i]
    
    # Create a list for any features that weren't in the importance file
    missing_indices = [i for i in range(n_features) if i not in [feature_to_index.get(f) for f in feature_ordering if f in feature_to_index]]
    
    # Assign any missing features to remaining grid positions
    remaining_positions = grid_coords[len(feature_ordering):]
    for i, idx in enumerate(missing_indices):
        if i < len(remaining_positions):
            feature_to_position[idx] = remaining_positions[i]
    
    # Create reshaped arrays for each set
    reshaped_dict = {}
    for key in ['X_train', 'X_val', 'X_test']:
        # Get the original data
        X = features_dict[key]
        n_samples = X.shape[0]
        
        # Create a zero-filled array for the target shape
        reshaped_X = np.zeros((n_samples, height, width))
        
        # Place each feature at its assigned position
        for feature_idx, (y, x) in feature_to_position.items():
            if feature_idx < X.shape[1]:
                reshaped_X[:, y, x] = X[:, feature_idx]
        
        # Reshape to target shape with samples as the first dimension (for PyTorch: [N, C, H, W])
        reshaped_dict[key] = reshaped_X.reshape(n_samples, 1, height, width)
    
    # Copy the labels
    reshaped_dict['y_train'] = features_dict['y_train']
    reshaped_dict['y_val'] = features_dict['y_val']
    reshaped_dict['y_test'] = features_dict['y_test']
    
    print(f"Features arranged by importance into shape {target_shape}")
    print(f"New data shapes: {reshaped_dict['X_train'].shape}")
    
    return reshaped_dict