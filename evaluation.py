import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from typing import Dict, List, Optional
import torch
import pandas as pd

def evaluate_model(model, data_loader, device='cuda'):
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

def print_evaluation_report(metrics, y_true, y_pred, dataset_name='test'):
    """
    Print evaluation metrics and plots.
    
    Args:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
        dataset_name: Name of the dataset being evaluated
    """
    # Print results
    print(f"\nEvaluation on {dataset_name} set:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['auc']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({dataset_name} set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_training_history(history):
    """
    Plot training history curves.
    
    Args:
        history: Training history dataframe with columns for each metric
    """
    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Plot additional metrics if available
    if 'train_auc' in history.columns:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_auc'])
        plt.plot(history['val_auc'])
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_prec'])
        plt.plot(history['val_prec'])
        plt.title('Model Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 3, 3)
        plt.plot(history['train_rec'])
        plt.plot(history['val_rec'])
        plt.title('Model Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()