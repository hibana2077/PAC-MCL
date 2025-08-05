"""
Training utilities including early stopping, learning rate scheduling, and checkpointing
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import os
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import shutil
import logging
import time
from collections import defaultdict


class EarlyStopping:
    """
    Early stopping utility to stop training when metric stops improving
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
            restore_best_weights: Whether to restore best model weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_weights = None
        self.patience_counter = 0
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights:
                # Restore best weights
                model.load_state_dict(self.best_weights)
        
        return self.early_stop
    
    def get_state(self) -> Dict[str, Any]:
        """Get early stopping state for checkpointing"""
        return {
            'best_score': self.best_score,
            'patience_counter': self.patience_counter,
            'early_stop': self.early_stop
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load early stopping state from checkpoint"""
        self.best_score = state['best_score']
        self.patience_counter = state['patience_counter']
        self.early_stop = state['early_stop']


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with warmup
    """
    
    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 max_epochs: int,
                 eta_min: float = 0.0,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + np.cos(np.pi * progress)) / 2
                    for base_lr in self.base_lrs]


class LearningRateScheduler:
    """
    Learning rate scheduler factory and utilities
    """
    
    @staticmethod
    def create_scheduler(scheduler_type: str,
                        optimizer,
                        **kwargs):
        """
        Create learning rate scheduler
        
        Args:
            scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'warmup_cosine')
            optimizer: PyTorch optimizer
            **kwargs: Scheduler-specific arguments
        """
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, **kwargs)
        
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            return StepLR(optimizer, **kwargs)
        
        elif scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            return ReduceLROnPlateau(optimizer, **kwargs)
        
        elif scheduler_type == 'warmup_cosine':
            return WarmupCosineScheduler(optimizer, **kwargs)
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class ModelCheckpoint:
    """
    Model checkpointing utility with support for best model saving
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 filename_template: str = 'checkpoint_epoch_{epoch:03d}.pth',
                 save_best: bool = True,
                 save_last: bool = True,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_top_k: int = 3,
                 verbose: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            filename_template: Template for checkpoint filenames
            save_best: Whether to save best model
            save_last: Whether to save last model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for monitored metric
            save_top_k: Number of best checkpoints to keep
            verbose: Whether to print checkpoint info
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename_template = filename_template
        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose = verbose
        
        self.best_score = None
        self.best_checkpoints = []  # List of (score, filepath) tuples
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer,
                       scheduler,
                       epoch: int,
                       metrics: Dict[str, float],
                       additional_info: Optional[Dict[str, Any]] = None):
        """
        Save checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Dictionary of metrics
            additional_info: Additional information to save
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / 'last_checkpoint.pth'
            torch.save(checkpoint, last_path)
            if self.verbose:
                print(f"Saved last checkpoint: {last_path}")
        
        # Check if this is the best checkpoint
        if self.save_best and self.monitor in metrics:
            current_score = metrics[self.monitor]
            
            if self.best_score is None or self.is_better(current_score, self.best_score):
                self.best_score = current_score
                
                # Save best checkpoint
                best_path = self.checkpoint_dir / 'best_checkpoint.pth'
                torch.save(checkpoint, best_path)
                
                if self.verbose:
                    print(f"New best {self.monitor}: {current_score:.6f} - Saved: {best_path}")
            
            # Manage top-k checkpoints
            if self.save_top_k > 0:
                checkpoint_path = self.checkpoint_dir / self.filename_template.format(epoch=epoch)
                torch.save(checkpoint, checkpoint_path)
                
                self.best_checkpoints.append((current_score, checkpoint_path))
                
                # Sort and keep only top-k
                self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
                
                if len(self.best_checkpoints) > self.save_top_k:
                    # Remove worst checkpoint
                    _, worst_path = self.best_checkpoints.pop()
                    if worst_path.exists():
                        worst_path.unlink()
                        if self.verbose:
                            print(f"Removed old checkpoint: {worst_path}")
    
    def load_checkpoint(self,
                       filepath: str,
                       model: nn.Module,
                       optimizer=None,
                       scheduler=None,
                       map_location=None) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            map_location: Device mapping for loading
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.verbose:
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded checkpoint from epoch {epoch}: {filepath}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        return str(last_path) if last_path.exists() else None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        best_path = self.checkpoint_dir / 'best_checkpoint.pth'
        return str(best_path) if best_path.exists() else None


class TrainingTracker:
    """
    Track training progress and metrics
    """
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = defaultdict(list)
        self.epoch_times = []
        
        # Setup logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger for training"""
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def log_epoch(self,
                  epoch: int,
                  epoch_time: float,
                  train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float]):
        """Log epoch results"""
        self.epoch_times.append(epoch_time)
        
        # Store metrics
        for key, value in train_metrics.items():
            self.metrics_history[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            self.metrics_history[f'val_{key}'].append(value)
        
        # Log to file
        self.logger.info(f"Epoch {epoch:03d} - Time: {epoch_time:.2f}s")
        self.logger.info(f"Train metrics: {train_metrics}")
        self.logger.info(f"Val metrics: {val_metrics}")
        
        # Save metrics to JSON
        metrics_file = self.log_dir / 'metrics_history.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for key, values in self.metrics_history.items():
                serializable_metrics[key] = [float(v) for v in values]
            
            json.dump({
                'metrics': serializable_metrics,
                'epoch_times': [float(t) for t in self.epoch_times]
            }, f, indent=2)
    
    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Tuple[float, int]:
        """Get best value and epoch for a metric"""
        if metric_name not in self.metrics_history:
            return None, -1
        
        values = self.metrics_history[metric_name]
        if mode == 'min':
            best_value = min(values)
            best_epoch = values.index(best_value)
        else:
            best_value = max(values)
            best_epoch = values.index(best_value)
        
        return best_value, best_epoch
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        try:
            import matplotlib.pyplot as plt
            
            # Get unique metric names (without train/val prefix)
            metric_names = set()
            for key in self.metrics_history.keys():
                if key.startswith(('train_', 'val_')):
                    metric_names.add(key.split('_', 1)[1])
            
            # Plot each metric
            fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metric_names):
                ax = axes[i]
                
                train_key = f'train_{metric}'
                val_key = f'val_{metric}'
                
                if train_key in self.metrics_history:
                    ax.plot(self.metrics_history[train_key], label=f'Train {metric}')
                
                if val_key in self.metrics_history:
                    ax.plot(self.metrics_history[val_key], label=f'Val {metric}')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} over training')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.log_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping metric plots")


if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")
    
    # Create dummy model and optimizer for testing
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test EarlyStopping
    print("\n=== Testing EarlyStopping ===")
    early_stopping = EarlyStopping(patience=3, mode='min')
    
    # Simulate training with improving then worsening loss
    losses = [1.0, 0.8, 0.6, 0.5, 0.7, 0.9, 1.1, 1.2]
    
    for epoch, loss in enumerate(losses):
        should_stop = early_stopping(loss, model)
        print(f"Epoch {epoch}: loss={loss:.2f}, patience={early_stopping.patience_counter}, stop={should_stop}")
        if should_stop:
            break
    
    # Test LearningRateScheduler
    print("\n=== Testing LearningRateScheduler ===")
    scheduler = LearningRateScheduler.create_scheduler(
        'warmup_cosine',
        optimizer,
        warmup_epochs=5,
        max_epochs=20
    )
    
    lrs = []
    for epoch in range(20):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    print(f"Learning rates over 20 epochs: {[f'{lr:.6f}' for lr in lrs[:10]]}...")
    
    # Test ModelCheckpoint
    print("\n=== Testing ModelCheckpoint ===")
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = ModelCheckpoint(
            checkpoint_dir=temp_dir,
            save_top_k=2,
            monitor='val_loss',
            mode='min'
        )
        
        # Simulate saving checkpoints
        for epoch in range(5):
            metrics = {'val_loss': 1.0 - 0.1 * epoch, 'val_acc': 0.5 + 0.1 * epoch}
            checkpoint.save_checkpoint(model, optimizer, scheduler, epoch, metrics)
        
        print(f"Saved checkpoints to: {temp_dir}")
        print(f"Best checkpoint: {checkpoint.get_best_checkpoint()}")
        print(f"Latest checkpoint: {checkpoint.get_latest_checkpoint()}")
    
    # Test TrainingTracker
    print("\n=== Testing TrainingTracker ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = TrainingTracker(log_dir=temp_dir)
        
        # Simulate logging epochs
        for epoch in range(3):
            train_metrics = {'loss': 1.0 - 0.1 * epoch, 'accuracy': 0.6 + 0.1 * epoch}
            val_metrics = {'loss': 1.1 - 0.1 * epoch, 'accuracy': 0.55 + 0.1 * epoch}
            
            tracker.log_epoch(epoch, epoch_time=10.5, 
                             train_metrics=train_metrics, val_metrics=val_metrics)
        
        # Test best metric retrieval
        best_val_loss, best_epoch = tracker.get_best_metric('val_loss', mode='min')
        print(f"Best val_loss: {best_val_loss:.3f} at epoch {best_epoch}")
        
        print(f"Tracking logs saved to: {temp_dir}")
    
    print("\nAll training utilities tests passed!")
