import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch

class ExperimentLogger:
    """
    Logger for phosphorylation site prediction experiments.
    Handles logging, saving metrics, and creating visualizations.
    """
    
    def __init__(self, experiment_name, log_dir='logs'):
        """
        Initialize logger with experiment name and directory
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs and results
        """
        self.experiment_name = experiment_name
        
        # Create timestamp for unique experiment ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set up logging
        self.log_file = os.path.join(self.experiment_dir, 'experiment.log')
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize history dataframe
        self.history = None
        
        # Log experiment start
        self.logger.info(f"Experiment {self.experiment_id} started")
    
    def log_config(self, config):
        """
        Log experiment configuration
        
        Args:
            config: Dictionary with experiment configuration
        """
        self.logger.info("Experiment configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save configuration to JSON file
        config_file = os.path.join(self.experiment_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def log_message(self, message, level='info'):
        """
        Log a message
        
        Args:
            message: Message to log
            level: Logging level (info, warning, error, critical)
        """
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
    
    def log_epoch(self, epoch, metrics):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary with metrics for train and validation
        """
        # Add epoch number to metrics
        metrics['epoch'] = epoch + 1
        
        # Log metrics
        self.logger.info(f"Epoch {epoch+1} - "
                         f"Train Loss: {metrics['train_loss']:.4f}, "
                         f"Train Acc: {metrics['train_acc']:.4f}, "
                         f"Train AUC: {metrics['train_auc']:.4f}, "
                         f"Val Loss: {metrics['val_loss']:.4f}, "
                         f"Val Acc: {metrics['val_acc']:.4f}, "
                         f"Val AUC: {metrics['val_auc']:.4f}")
        
        # Initialize history dataframe if not already created
        if self.history is None:
            self.history = pd.DataFrame([metrics])
        else:
            self.history = pd.concat([self.history, pd.DataFrame([metrics])], ignore_index=True)
        
        # Save history after each epoch
        self._save_history()
    
    def log_evaluation_results(self, dataset, metrics, y_true, y_pred, y_pred_proba=None):
        """
        Log evaluation results for a dataset
        
        Args:
            dataset: Dataset name (train, val, test)
            metrics: Dictionary with evaluation metrics
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        """
        # Log metrics
        self.logger.info(f"Evaluation on {dataset} set:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Save metrics to JSON file
        metrics_file = os.path.join(self.experiment_dir, f'{dataset}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Generate and save confusion matrix
        self._save_confusion_matrix(dataset, y_true, y_pred)
        
        # Save classification report
        self._save_classification_report(dataset, y_true, y_pred)
        
        # Save predictions
        self._save_predictions(dataset, y_true, y_pred, y_pred_proba)
    
    def log_model_summary(self, model):
        """
        Log model summary
        
        Args:
            model: PyTorch model
        """
        # Get model summary
        summary = []
        self.logger.info("Model summary:")
        summary.append("Model summary:")
        
        # Get number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary.append(f"Total parameters: {total_params}")
        summary.append(f"Trainable parameters: {trainable_params}")
        
        self.logger.info(f"  Total parameters: {total_params}")
        self.logger.info(f"  Trainable parameters: {trainable_params}")
        
        # Save model summary to file
        summary_file = os.path.join(self.experiment_dir, 'model_summary.txt')
        with open(summary_file, 'w') as f:
            for line in summary:
                f.write(f"{line}\n")
    
    def save_model(self, model, filename):
        """
        Save PyTorch model
        
        Args:
            model: PyTorch model
            filename: Filename for the model
        """
        model_path = os.path.join(self.experiment_dir, filename)
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def finalize(self):
        """
        Finalize experiment logging and generate plots
        """
        # Generate and save plots
        self._generate_plots()
        
        # Log experiment end
        self.logger.info(f"Experiment {self.experiment_id} completed")
    
    def _save_history(self):
        """Save training history to CSV file"""
        history_file = os.path.join(self.experiment_dir, 'history.csv')
        self.history.to_csv(history_file, index=False)
    
    def _save_confusion_matrix(self, dataset, y_true, y_pred):
        """
        Generate and save confusion matrix
        
        Args:
            dataset: Dataset name
            y_true: True labels
            y_pred: Predicted labels
        """
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({dataset} set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        plot_file = os.path.join(self.experiment_dir, f'{dataset}_confusion_matrix.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_classification_report(self, dataset, y_true, y_pred):
        """
        Save classification report
        
        Args:
            dataset: Dataset name
            y_true: True labels
            y_pred: Predicted labels
        """
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Save report to JSON file
        report_file = os.path.join(self.experiment_dir, f'{dataset}_classification_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
    
    def _save_predictions(self, dataset, y_true, y_pred, y_pred_proba=None):
        """
        Save predictions
        
        Args:
            dataset: Dataset name
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        """
        # Create dataframe with predictions
        predictions = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred
        })
        
        # Add probabilities if available
        if y_pred_proba is not None:
            predictions['probability'] = y_pred_proba
        
        # Save predictions to CSV file
        predictions_file = os.path.join(self.experiment_dir, f'{dataset}_predictions.csv')
        predictions.to_csv(predictions_file, index=False)
    
    def _generate_plots(self):
        """Generate and save plots from training history"""
        if self.history is None or len(self.history) == 0:
            self.logger.warning("No training history available for plotting")
            return
        
        # Plot accuracy and loss
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'])
        plt.plot(self.history['val_acc'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'])
        plt.plot(self.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Save figure
        plot_file = os.path.join(self.experiment_dir, 'accuracy_loss.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot additional metrics if available
        if 'train_auc' in self.history.columns:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(self.history['train_auc'])
            plt.plot(self.history['val_auc'])
            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.subplot(1, 3, 2)
            plt.plot(self.history['train_prec'])
            plt.plot(self.history['val_prec'])
            plt.title('Model Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.subplot(1, 3, 3)
            plt.plot(self.history['train_rec'])
            plt.plot(self.history['val_rec'])
            plt.title('Model Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # Save figure
            plot_file = os.path.join(self.experiment_dir, 'additional_metrics.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()