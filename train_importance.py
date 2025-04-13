import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Import our modules
from data_loader import load_data, normalize_features, create_data_loaders, PhosphorylationDataset
from feature_arrangers.importance_arranger import arrange_features_by_importance
from models.cnn_model import CNN
from logger import ExperimentLogger

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, optimizer, criterion, device, l2_reg=0):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Add L2 regularization if applicable
        if l2_reg > 0 and hasattr(model, 'get_l2_regularization_loss'):
            l2_loss = model.get_l2_regularization_loss()
            loss += l2_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Move predictions to CPU for metric calculation
        probs = outputs.detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        y_true.extend(targets.cpu().numpy().flatten())
        y_pred.extend(preds.flatten())
        y_pred_proba.extend(probs.flatten())
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_auc = roc_auc_score(y_true, y_pred_proba)
    epoch_prec = precision_score(y_true, y_pred)
    epoch_rec = recall_score(y_true, y_pred)
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'auc': epoch_auc,
        'prec': epoch_prec,
        'rec': epoch_rec
    }

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Move predictions to CPU for metric calculation
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(preds.flatten())
            y_pred_proba.extend(probs.flatten())
    
    # Calculate validation metrics
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(y_true, y_pred)
    val_auc = roc_auc_score(y_true, y_pred_proba)
    val_prec = precision_score(y_true, y_pred)
    val_rec = recall_score(y_true, y_pred)
    
    return {
        'loss': val_loss,
        'acc': val_acc,
        'auc': val_auc,
        'prec': val_prec,
        'rec': val_rec
    }

def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance on the given dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for the dataset to evaluate
        device: Device to run evaluation on ('cuda' or 'cpu')
        
    Returns:
        Dictionary of metrics and numpy arrays of predictions
    """
    model.eval()  # Set model to evaluation mode
    
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Move predictions back to CPU for sklearn metrics
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            y_true.extend(targets.numpy().flatten())
            y_pred.extend(preds.flatten())
            y_pred_proba.extend(probs.flatten())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    return metrics, y_true, y_pred, y_pred_proba

def train_importance_arrangement_model(data_dir='split_data', 
                                      importance_file='feature_importance.json',
                                      target_shape=None, 
                                      batch_size=32, 
                                      epochs=50, 
                                      patience=10,
                                      learning_rate=0.001,
                                      weight_decay=0.0001,
                                      dropout_rate=0.5,
                                      experiment_name="importance_cnn"):
    """
    Train a CNN model using importance-based arranged features.
    
    Args:
        data_dir: Directory containing the split data files
        importance_file: Path to JSON file with feature importance scores
        target_shape: Shape for arranging features (height, width)
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 regularization)
        dropout_rate: Dropout rate for regularization
        experiment_name: Name of the experiment for logging
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create experiment logger
    logger = ExperimentLogger(experiment_name)
    
    # Log configuration
    config = {
        'data_dir': data_dir,
        'importance_file': importance_file,
        'target_shape': target_shape,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout_rate': dropout_rate,
        'experiment_name': experiment_name
    }
    logger.log_config(config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_message(f"Using device: {device}")
    
    # Load and normalize data
    logger.log_message("Loading and normalizing data...")
    features_dict, metadata_dict = load_data(data_dir, use_datatable=True)
    norm_features_dict = normalize_features(features_dict)
    
    # Arrange features by importance for CNN
    logger.log_message("Arranging features by importance...")
    arranged_features = arrange_features_by_importance(
        norm_features_dict, 
        target_shape, 
        importance_file,
        metadata_dict['feature_names']
    )
    
    # Create data loaders
    logger.log_message("Creating data loaders...")
    data_loaders = create_data_loaders(arranged_features, batch_size)
    
    # Get input dimensions for the model
    input_shape = arranged_features['X_train'].shape[1:]  # (C, H, W)
    logger.log_message(f"Input shape for CNN: {input_shape}")
    
    # Create the model
    logger.log_message("Creating model...")
    model = CNN(
        input_channels=input_shape[0],
        input_height=input_shape[1],
        input_width=input_shape[2],
        dropout_rate=dropout_rate,
        l2_reg=weight_decay
    )
    model = model.to(device)
    
    # Log model summary
    logger.log_model_summary(model)
    
    # Create directories for saving models
    os.makedirs('models/saved', exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler for adaptive learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize variables for early stopping
    best_val_auc = 0
    patience_counter = 0
    
    # Training loop
    logger.log_message("Starting training...")
    for epoch in range(epochs):
        # Train and validate
        train_metrics = train_epoch(model, data_loaders['train'], optimizer, criterion, device)
        val_metrics = validate(model, data_loaders['val'], criterion, device)
        
        # Log epoch metrics
        logger.log_epoch(epoch, {
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'train_auc': train_metrics['auc'],
            'train_prec': train_metrics['prec'],
            'train_rec': train_metrics['rec'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_auc': val_metrics['auc'],
            'val_prec': val_metrics['prec'],
            'val_rec': val_metrics['rec']
        })
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['auc'])
        
        # Check for improvement
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            # Save best model
            logger.save_model(model, 'best_model.pt')
            logger.log_message(f"New best model with val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            logger.log_message(f"No improvement for {patience_counter} epochs (best val AUC: {best_val_auc:.4f})")
        
        # Early stopping
        if patience_counter >= patience:
            logger.log_message(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(logger.experiment_dir, 'best_model.pt')))
    
    # Evaluate on all datasets
    logger.log_message("Evaluating best model...")
    train_metrics, train_y_true, train_y_pred, train_y_pred_proba = evaluate_model(model, data_loaders['train'], device)
    val_metrics, val_y_true, val_y_pred, val_y_pred_proba = evaluate_model(model, data_loaders['val'], device)
    test_metrics, test_y_true, test_y_pred, test_y_pred_proba = evaluate_model(model, data_loaders['test'], device)
    
    # Log evaluation results
    logger.log_evaluation_results('train', train_metrics, train_y_true, train_y_pred, train_y_pred_proba)
    logger.log_evaluation_results('val', val_metrics, val_y_true, val_y_pred, val_y_pred_proba)
    logger.log_evaluation_results('test', test_metrics, test_y_true, test_y_pred, test_y_pred_proba)
    
    # Save final model
    logger.save_model(model, 'final_model.pt')
    
    # Finalize experiment
    logger.finalize()
    
    return model, logger.history, (train_metrics, val_metrics, test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN model with importance-based arranged features')
    parser.add_argument('--data_dir', type=str, default='split_data', help='Directory containing the split data files')
    parser.add_argument('--importance_file', type=str, default='feature_importance.json', help='Path to feature importance file')
    parser.add_argument('--width', type=int, default=None, help='Width of the 2D feature matrix')
    parser.add_argument('--height', type=int, default=None, help='Height of the 2D feature matrix')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for L2 regularization')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--experiment', type=str, default='importance_cnn', help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Set target shape if provided
    target_shape = None
    if args.width is not None and args.height is not None:
        target_shape = (args.height, args.width)
    
    # Train the model
    model, history, metrics = train_importance_arrangement_model(
        data_dir=args.data_dir,
        importance_file=args.importance_file,
        target_shape=target_shape,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        experiment_name=args.experiment
    )
    
    print("Training completed!")