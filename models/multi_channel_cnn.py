# models/multichannel_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class MultichannelCNN(nn.Module):
    """
    CNN model that processes multiple input channels separately and then
    combines them for phosphorylation site prediction.
    """
    def __init__(self, 
                 input_channels: int = 5,  # Number of feature type channels
                 input_height: int = 32, 
                 input_width: int = 32,
                 filters_per_channel: List[int] = [16, 32],
                 shared_filters: List[int] = [64, 128],
                 kernel_size: Tuple[int, int] = (3, 3),
                 pool_size: Tuple[int, int] = (2, 2),
                 dense_units: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.0001):
        """
        Initialize MultichannelCNN model
        
        Args:
            input_channels: Number of input channels (feature types)
            input_height: Maximum height of input feature matrices
            input_width: Maximum width of input feature matrices
            filters_per_channel: List of filter counts for per-channel convolutions
            shared_filters: List of filter counts for shared convolutions after channel fusion
            kernel_size: Size of convolutional kernels
            pool_size: Size of pooling windows
            dense_units: List of units in dense layers
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
        """
        super(MultichannelCNN, self).__init__()
        
        # Store model parameters
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.l2_reg = l2_reg
        
        # Create separate convolutional paths for each input channel
        self.channel_paths = nn.ModuleList()
        
        for _ in range(input_channels):
            # Create a sequence of convolutional layers for this channel
            layers = []
            
            # Input shape: [batch_size, 1, height, width]
            in_channels = 1
            
            for out_channels in filters_per_channel:
                # Add convolutional block (conv + batch norm + relu + pool + dropout)
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(pool_size),
                    nn.Dropout2d(dropout_rate / 2)  # Lower dropout in conv layers
                )
                layers.append(conv_block)
                in_channels = out_channels
            
            # Add this channel's convolutional path to the module list
            self.channel_paths.append(nn.Sequential(*layers))
        
        # Calculate the output size after per-channel convolutions
        # After each pooling operation, dimensions are halved
        h_out = input_height
        w_out = input_width
        for _ in filters_per_channel:
            h_out = h_out // pool_size[0]
            w_out = w_out // pool_size[1]
        
        # Ensure dimensions don't go below 1
        h_out = max(1, h_out)
        w_out = max(1, w_out)
        
        # Calculate output channels from each path
        channel_output_filters = filters_per_channel[-1] if filters_per_channel else 1
        
        # Create shared convolutional layers after concatenation
        self.shared_conv_layers = nn.ModuleList()
        in_channels = channel_output_filters * input_channels
        
        for out_channels in shared_filters:
            # Add convolutional block
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                nn.Dropout2d(dropout_rate / 2)
            )
            self.shared_conv_layers.append(conv_block)
            in_channels = out_channels
            
            # Update output dimensions
            h_out = max(1, h_out // pool_size[0])
            w_out = max(1, w_out // pool_size[1])
        
        # Calculate flattened size
        self.flattened_size = shared_filters[-1] * h_out * w_out if shared_filters else channel_output_filters * input_channels * h_out * w_out
        
        # Create dense layers
        self.dense_layers = nn.ModuleList()
        prev_units = self.flattened_size
        
        for units in dense_units:
            layer = nn.Sequential(
                nn.Linear(prev_units, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.dense_layers.append(layer)
            prev_units = units
        
        # Create output layer
        self.output_layer = nn.Linear(prev_units, 1)
        
        # Print model dimensions for debugging
        print(f"MultichannelCNN model created:")
        print(f"  Input channels: {input_channels}")
        print(f"  Input dimensions: {input_height}x{input_width}")
        print(f"  Output dimensions after channel paths: {h_out}x{w_out}")
        print(f"  Flattened size: {self.flattened_size}")
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Process each channel separately
        channel_outputs = []
        
        for i in range(self.input_channels):
            # Extract this channel's data
            # Input shape: [batch_size, channels, height, width]
            # We want: [batch_size, 1, height, width]
            channel_input = x[:, i:i+1, :, :]
            
            # Pass through this channel's convolutional path
            channel_output = self.channel_paths[i](channel_input)
            
            # Add to outputs list
            channel_outputs.append(channel_output)
        
        # Concatenate channel outputs along the channel dimension
        # From list of [batch_size, out_channels, h, w] to [batch_size, out_channels*num_paths, h, w]
        x = torch.cat(channel_outputs, dim=1)
        
        # Process through shared convolutional layers
        for layer in self.shared_conv_layers:
            x = layer(x)
        
        # Flatten the output
        x = x.view(batch_size, -1)
        
        # Process through dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.output_layer(x))
        
        return x
    
    def get_l2_regularization_loss(self):
        """
        Calculate L2 regularization loss for all weights
        """
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:  # Only apply to weights, not biases
                l2_loss += torch.norm(param, 2) ** 2
        return self.l2_reg * l2_loss / 2  # Standard L2 regularization formula