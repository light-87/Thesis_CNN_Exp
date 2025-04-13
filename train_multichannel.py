# train_multichannel.py
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

from feature_arrangers.multichannel_arranger import MultichannelArranger
from models.multi_channel_cnn import MultichannelCNN
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

def normalize_features(feature_dict):
    """
    Normalize features using mean and standard deviation from the training set
    """
    from sklearn.preprocessing import StandardScaler
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(feature_dict['X_train'])
    
    # Transform validation and test data
    X_val_scaled = scaler.transform(feature_dict['X_val'])
    X_test_scaled = scaler.transform(feature_dict['X_test'])
    
    # Create a new dictionary with normalized features
    normalized_dict = {
        'X_train': X_train_scaled,
        'y_train': feature_dict['y_train'].values,
        'X_val': X_val_scaled,
        'y_val': feature_dict['y_val'].values,
        'X_test': X_test_scaled,
        'y_test': feature_dict['y_test'].values
    }
    
    return normalized_dict

def prepare_data(feature_dict, arranger, batch_size=64):
    """
    Prepare data for model training using the feature arranger
    """
    # Arrange features
    X_train_arr = arranger.arrange_features(feature_dict['X_train'])
    X_val_arr = arranger.arrange_features(feature_dict['X_val'])
    X_test_arr = arranger.arrange_features(feature_dict['X_test'])
    
    # Convert targets to tensors
    y_train_tensor = torch.from_numpy(feature_dict['y_train']).float().unsqueeze(1)
    y_val_tensor = torch.from_numpy(feature_dict['y_val']).float().unsqueeze(1)
    y_test_tensor = torch.from_numpy(feature_dict['y_test']).float().unsqueeze(1)
    
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
    
    return data_loaders, X_train_arr.shape

def train_model(model, data_loaders, optimizer, criterion, device, num_epochs, logger, early_stopping_patience=10):
    """
    Train the CNN model
    """
    start_time = time.time()
    best_val_auc = 0
    best_model_state = None
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': [],
        'train_prec': [],
        'val_prec': [],
        'train_rec': [],
        'val_rec': []
    }
    
    for epoch in range(num_epochs):
        logger.log_message(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_probs = []
        train_targets = []
        
        for inputs, targets in data_loaders['train']:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add L2 regularization if applicable
            if hasattr(model, 'get_l2_regularization_loss'):
                l2_loss = model.get_l2_regularization_loss()
                loss += l2_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track stats
            train_loss += loss.item() * inputs.size(0)
            train_probs.extend(outputs.detach().cpu().numpy())
            train_preds.extend((outputs > 0.5).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(data_loaders['train'].dataset)
        train_preds = np.array(train_preds).flatten()
        train_probs = np.array(train_probs).flatten()
        train_targets = np.array(train_targets).flatten()
        
        train_acc = accuracy_score(train_targets, train_preds)
        train_auc = roc_auc_score(train_targets, train_probs)
        train_prec = precision_score(train_targets, train_preds)
        train_rec = recall_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loaders['val']:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track stats
                val_loss += loss.item() * inputs.size(0)
                val_probs.extend(outputs.cpu().numpy())
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(data_loaders['val'].dataset)
        val_preds = np.array(val_preds).flatten()
        val_probs = np.array(val_probs).flatten()
        val_targets = np.array(val_targets).flatten()
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_probs)
        val_prec = precision_score(val_targets, val_preds)
        val_rec = recall_score(val_targets, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_prec'].append(train_prec)
        history['val_prec'].append(val_prec)
        history['train_rec'].append(train_rec)
        history['val_rec'].append(val_rec)
        
        # Log progress
        logger.log_message(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        logger.log_message(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            logger.log_message(f"  New best model with validation AUC: {val_auc:.4f}")
        else:
            epochs_no_improve += 1
            logger.log_message(f"  No improvement for {epochs_no_improve} epochs")
            
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            logger.log_message(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    time_elapsed = time.time() - start_time
    logger.log_message(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
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
            outputs = model(inputs)
            
            # Store predictions and targets
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Flatten arrays
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_probs)
    
    # Log results
    logger.log_message(f"Evaluation on {dataset_name} set:")
    logger.log_message(f"  Accuracy: {accuracy:.4f}")
    logger.log_message(f"  Precision: {precision:.4f}")
    logger.log_message(f"  Recall: {recall:.4f}")
    logger.log_message(f"  F1 Score: {f1:.4f}")
    logger.log_message(f"  ROC AUC: {roc_auc:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    return metrics, all_targets, all_preds, all_probs

def main():
    parser = argparse.ArgumentParser(description='Train multi-channel CNN for phosphorylation site prediction')
    parser.add_argument('--train_file', type=str, default='split_data/train_data.csv', help='Path to training data')
    parser.add_argument('--val_file', type=str, default='split_data/val_data.csv', help='Path to validation data')
    parser.add_argument('--test_file', type=str, default='split_data/test_data.csv', help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--experiment_name', type=str, default='multichannel_cnn', help='Experiment name')
    
    args = parser.parse_args()
    
    # Set experiment name with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.experiment_name}_{timestamp}"
    
    # Setup logger
    logger = ExperimentLogger(experiment_name)
    
    # Log experiment parameters
    logger.log_message(f"Starting experiment: {experiment_name}")
    logger.log_config(vars(args))
    
    # Load data
    logger.log_message("Loading data...")
    feature_dict = load_data(args.train_file, args.val_file, args.test_file)
    
    # Normalize features
    logger.log_message("Normalizing features...")
    norm_feature_dict = normalize_features(feature_dict)
    
    # Create feature arranger
    logger.log_message("Creating multi-channel feature arranger...")
    feature_columns = feature_dict['X_train'].columns.tolist()
    arranger = MultichannelArranger(feature_columns)
    
    # Prepare data
    logger.log_message("Preparing data...")
    data_loaders, input_shape = prepare_data(norm_feature_dict, arranger, args.batch_size)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_message(f"Using device: {device}")
    
    # Get necessary model parameters from input shape
    n_samples, n_channels, max_height, max_width = input_shape
    
    # Create model
    logger.log_message("Creating multi-channel CNN model...")
    model = MultichannelCNN(
        input_channels=n_channels,
        input_height=max_height,
        input_width=max_width,
        filters_per_channel=[16, 32],
        shared_filters=[64, 128],
        dense_units=[256, 128, 64],
        dropout_rate=args.dropout,
        l2_reg=args.weight_decay
    ).to(device)
    
    # Log model summary
    logger.log_model_summary(model)
    
    # Setup loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    logger.log_message("Starting training...")
    model, history = train_model(
        model=model,
        data_loaders=data_loaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        logger=logger,
        early_stopping_patience=args.patience
    )
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(logger.experiment_dir, 'history.csv'), index=False)
    
    # Generate plots from training history
    logger._generate_plots()
    
    # Evaluate model
    logger.log_message("Evaluating best model...")
    train_metrics, train_targets, train_preds, train_probs = evaluate_model(
        model, data_loaders['train'], device, logger, "train"
    )
    val_metrics, val_targets, val_preds, val_probs = evaluate_model(
        model, data_loaders['val'], device, logger, "validation"
    )
    test_metrics, test_targets, test_preds, test_probs = evaluate_model(
        model, data_loaders['test'], device, logger, "test"
    )
    
    # Log evaluation results
    logger.log_evaluation_results('train', train_metrics, train_targets, train_preds, train_probs)
    logger.log_evaluation_results('val', val_metrics, val_targets, val_preds, val_probs)
    logger.log_evaluation_results('test', test_metrics, test_targets, test_preds, test_probs)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(logger.experiment_dir, 'final_model.pt'))
    logger.log_message(f"Model saved to {os.path.join(logger.experiment_dir, 'final_model.pt')}")
    
    # Save feature arranger
    torch.save(arranger, os.path.join(logger.experiment_dir, 'feature_arranger.pt'))
    
    # Finalize experiment
    logger.finalize()
    logger.log_message(f"Experiment {experiment_name} completed")
    print("Training completed!")

if __name__ == "__main__":
    main()