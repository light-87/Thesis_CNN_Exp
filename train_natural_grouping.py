# train_natural_grouping.py
import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datatable import fread

from models.cnn_model import CNN 
from feature_arrangers.natural_grouping_arranger import NaturalGroupingArranger
from logger import ExperimentLogger

def load_data(train_file, val_file, test_file):
    """
    Load data from CSV files using datatable for faster loading
    """
    print("Loading training data...")
    train_data = fread(train_file).to_pandas()
    
    print("Loading validation data...")
    val_data = fread(val_file).to_pandas()
    
    print("Loading test data...")
    test_data = fread(test_file).to_pandas()
    
    # Separate features and target
    X_train = train_data.drop(['Header', 'Position', 'target'], axis=1)
    y_train = train_data['target']
    
    X_val = val_data.drop(['Header', 'Position', 'target'], axis=1)
    y_val = val_data['target']
    
    X_test = test_data.drop(['Header', 'Position', 'target'], axis=1)
    y_test = test_data['target']
    
    feature_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    return feature_dict

def prepare_data(feature_dict, arranger, batch_size=64):
    """
    Prepare data for model training
    """
    # Arrange features
    X_train_arr = arranger.arrange_features(feature_dict['X_train'].values)
    X_val_arr = arranger.arrange_features(feature_dict['X_val'].values)
    X_test_arr = arranger.arrange_features(feature_dict['X_test'].values)
    
    # Convert targets to tensors
    y_train_tensor = torch.from_numpy(feature_dict['y_train'].values).float()
    y_val_tensor = torch.from_numpy(feature_dict['y_val'].values).float()
    y_test_tensor = torch.from_numpy(feature_dict['y_test'].values).float()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_arr, y_train_tensor)
    val_dataset = TensorDataset(X_val_arr, y_val_tensor)
    test_dataset = TensorDataset(X_test_arr, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return data_loaders

def train_model(model, data_loaders, optimizer, criterion, device, num_epochs, logger, early_stopping_patience=10):
    """
    Train the CNN model
    """
    start_time = time.time()
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in data_loaders['train']:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track stats
            train_loss += loss.item() * inputs.size(0)
            train_preds.extend((outputs > 0.5).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(data_loaders['train'].dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loaders['val']:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                
                # Track stats
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(data_loaders['val'].dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Log progress
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    time_elapsed = time.time() - start_time
    logger.info(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    return model, history

def evaluate_model(model, data_loader, device, logger, dataset_name=""):
    """
    Evaluate the model
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            
            # Store predictions and targets
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_probs)
    
    # Log results
    logger.info(f"Evaluation on {dataset_name} set:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train CNN with natural grouping feature arrangement')
    parser.add_argument('--train_file', type=str, default='split_data/train_data.csv', help='Path to training data')
    parser.add_argument('--val_file', type=str, default='split_data/val_data.csv', help='Path to validation data')
    parser.add_argument('--test_file', type=str, default='split_data/test_data.csv', help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--experiment_name', type=str, default='natural_cnn_v1', help='Experiment name')
    
    args = parser.parse_args()
    
    # Set experiment name with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.experiment_name}_{timestamp}"
    
    # Setup logger
    log_dir = os.path.join('logs', experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = ExperimentLogger(experiment_name, os.path.join(log_dir, 'train.log'))
    
    # Log experiment parameters
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Parameters: {vars(args)}")
    
    # Load data
    logger.info("Loading data...")
    feature_dict = load_data(args.train_file, args.val_file, args.test_file)
    
    # Create feature arranger
    logger.info("Creating feature arranger...")
    feature_columns = feature_dict['X_train'].columns.tolist()
    arranger = NaturalGroupingArranger(feature_columns)
    
    # Get arranged data shape
    sample_X = arranger.arrange_features(feature_dict['X_train'].iloc[:2].values)
    _, channels, height, width = sample_X.shape
    
    # Prepare data
    logger.info("Preparing data...")
    data_loaders = prepare_data(feature_dict, arranger, args.batch_size)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = CNN(
        input_channels=channels,
        height=height,
        width=width,
        dropout_rate=args.dropout
    ).to(device)
    logger.info(f"Model architecture:\n{model}")
    
    # Setup loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Train model
    logger.info("Starting training...")
    model, history = train_model(
        model=model,
        data_loaders=data_loaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        logger=logger
    )
    
    # Save training history
    with open(os.path.join(log_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    # Evaluate model
    logger.info("Evaluating best model...")
    train_metrics = evaluate_model(model, data_loaders['train'], device, logger, "train")
    val_metrics = evaluate_model(model, data_loaders['val'], device, logger, "val")
    test_metrics = evaluate_model(model, data_loaders['test'], device, logger, "test")
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(log_dir, 'final_model.pt'))
    logger.info(f"Model saved to {os.path.join(log_dir, 'final_model.pt')}")
    
    # Save feature arranger
    torch.save(arranger, os.path.join(log_dir, 'feature_arranger.pt'))
    
    logger.info(f"Experiment {experiment_name} completed")
    print("Training completed!")

if __name__ == "__main__":
    main()