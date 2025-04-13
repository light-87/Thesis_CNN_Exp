import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class CNN(nn.Module):
    """
    CNN model for phosphorylation site prediction with improved regularization
    """
    def __init__(self, 
                 input_channels: int = 1, 
                 input_height: int = 96, 
                 input_width: int = 96,
                 filters: List[int] = [32, 64, 128],
                 kernel_size: Tuple[int, int] = (3, 3),
                 pool_size: Tuple[int, int] = (2, 2),
                 dense_units: List[int] = [128, 64],
                 dropout_rate: float = 0.5,
                 l2_reg: float = 0.001):
        """
        Initialize CNN model
        
        Args:
            input_channels: Number of input channels (1 for grayscale)
            input_height: Height of input images
            input_width: Width of input images
            filters: List of filter counts for convolutional layers
            kernel_size: Size of convolutional kernels
            pool_size: Size of pooling windows
            dense_units: List of units in dense layers
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
        """
        super(CNN, self).__init__()
        
        # Store model parameters
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.filters = filters
        self.l2_reg = l2_reg
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, filters[0], kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.pool1 = nn.MaxPool2d(pool_size)
        self.dropout1 = nn.Dropout2d(dropout_rate / 2)  # Spatial dropout for conv layers
        
        # Additional convolutional blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(1, len(filters)):
            block = nn.Sequential(
                nn.Conv2d(filters[i-1], filters[i], kernel_size, padding='same'),
                nn.BatchNorm2d(filters[i]),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                nn.Dropout2d(dropout_rate / 2)  # Add dropout to each conv block
            )
            self.conv_blocks.append(block)
        
        # Calculate flattened size after convolutions and pooling
        # Start with input dimensions
        h, w = input_height, input_width
        # Apply pooling operations to calculate final feature map size
        for _ in range(len(filters)):
            h = h // pool_size[0]
            w = w // pool_size[1]
        
        # Ensure dimensions don't become zero
        h = max(1, h)
        w = max(1, w)
        
        # Calculate flattened size
        flattened_size = filters[-1] * h * w
        print(f"Feature map size after convolutions: {h}x{w} with {filters[-1]} channels")
        print(f"Flattened size: {flattened_size}")
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        prev_units = flattened_size
        for units in dense_units:
            layer = nn.Sequential(
                nn.Linear(prev_units, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # Higher dropout rate for dense layers
            )
            self.dense_layers.append(layer)
            prev_units = units
        
        # Output layer
        self.output = nn.Linear(prev_units, 1)
        
    def forward(self, x):
        """Forward pass through the network"""
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Additional convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        # Output layer with sigmoid activation for binary classification
        x = torch.sigmoid(self.output(x))
        
        return x
    
    def get_l2_regularization_loss(self):
        """Calculate L2 regularization loss for all weights"""
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:  # Only apply to weights, not biases
                l2_loss += torch.norm(param, 2)
        return self.l2_reg * l2_loss